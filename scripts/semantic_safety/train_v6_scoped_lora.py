"""Train comparable VLM-only, AE-only, or joint LoRA on replay-verified v6 motor data."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
import torch

from motor_lora import frozen_parameter_digest, load_policy, make_input, save_adapter, tensor_batch
from rollout_common import ROOT, write_json
from v6_scoped_lora import install, loss_matrix, prefix, VLM, AE


class Demonstrations:
    def __init__(self, manifest, policy, seed):
        self.rows=[json.loads(line) for line in manifest.read_text().splitlines()]
        assert self.rows and all(row["accepted"] for row in self.rows)
        self.native=[row for row in self.rows if row.get("data_type")=="native_rehearsal"]
        self.groups={}
        for row in self.rows:
            if row.get("data_type")!="native_rehearsal":self.groups.setdefault(row["condition_index"],[]).append(row)
        self.keys=sorted(self.groups);self.policy=policy;self.rng=np.random.default_rng(seed)
        self.pool=ThreadPoolExecutor(max_workers=4)

    def read(self,request):
        from v6_lora_data import instruction
        row,step,variant=request
        with h5py.File(ROOT/row.get("training_cache",row["path"]),"r") as data:
            spec=json.loads(data.attrs["spec_json"])
            prompt=spec["instruction"] if row.get("data_type")=="native_rehearsal" else instruction(spec,variant)
            actions=data["action_chunks"][step] if row.get("data_type")=="native_rehearsal" else data["actions"][step:step+10]
            return make_input(self.policy,data["observations/main_rgb_raw"][step],
                              data["observations/wrist_rgb_raw"][step],data["observations/state"][step],
                              prompt,actions)

    def batch(self,size):
        requests=[]
        for _ in range(size):
            if self.native and self.rng.random()<.1:
                row=self.native[int(self.rng.integers(len(self.native)))];step=int(self.rng.integers(row["chunks"]))
                requests.append((row,step,0));continue
            rows=self.groups[self.keys[int(self.rng.integers(len(self.keys)))]]
            row=rows[int(self.rng.integers(len(rows)))];step=int(self.rng.integers(row["steps"]-9))
            requests.append((row,step,int(self.rng.integers(4))))
        return list(self.pool.map(self.read,requests))


def gradient_groups(model):
    values={"vision":0.,"language":0.,"ae":0.};missing=[]
    for name,p in model.named_parameters():
        if not p.requires_grad:assert p.grad is None;continue
        if p.grad is None:missing.append(name);continue
        group="vision" if "vision_model" in name or "multi_modal_projector" in name else "language" if name.startswith(VLM) else "ae"
        values[group]+=float(p.grad.detach().square().sum())
    return dict(norms={k:v**.5 for k,v in values.items()},no_gradient=missing)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--scope",choices=["vlm","ae","all"],required=True)
    p.add_argument("--steps",type=int,default=6000);p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--learning-rate",type=float,default=2e-4);p.add_argument("--seed",type=int,default=20261111)
    p.add_argument("--save-every",type=int,default=1000);p.add_argument("--fixed-batch",action="store_true")
    p.add_argument("--pilot",action="store_true");p.add_argument("--resume",type=Path)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/"training.jsonl").exists():raise FileExistsError(args.out)
    torch.set_num_threads(8);torch.manual_seed(args.seed);np.random.seed(args.seed)
    policy,_=load_policy();model=policy._model;lora=install(model,args.scope)
    if args.resume:
        package=torch.load(args.resume,map_location="cpu",weights_only=False)
        assert package["metadata"]["lora"]==lora
        named=dict(model.named_parameters())
        with torch.no_grad():
            for name,value in package["state_dict"].items():named[name].copy_(value)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=args.learning_rate,weight_decay=.01)
    data=Demonstrations(args.manifest,policy,args.seed)
    fixed=data.batch(args.batch_size if args.fixed_batch else 2)
    fixed_obs,fixed_actions=tensor_batch(fixed)
    torch.manual_seed(99127);fixed_noise=model.sample_noise(fixed_actions.shape,"cuda")
    fixed_time=torch.full((len(fixed),),.5,device="cuda")
    before=frozen_parameter_digest(model)
    with torch.no_grad():
        hidden_before=prefix(model,fixed_obs,gradients=False)[3].cpu().clone()
        fixed_before=loss_matrix(model,fixed_obs,fixed_actions,args.scope,noise=fixed_noise,time=fixed_time).mean().item()
    metadata=dict(base_checkpoint="pi05_libero_pytorch",lora=lora,seed=args.seed,steps_planned=args.steps,
                  batch_size=args.batch_size,learning_rate_peak=args.learning_rate,optimizer="AdamW",weight_decay=.01,
                  gradient_clip=1.,manifest=str(args.manifest),manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  frozen_parameter_sha256_before=before,training_episodes=len(data.rows),conditions=data.keys,
                  normalization="unchanged official pi05_libero statistics",image_augmentation=False,
                  loss="flow matching over all 32 dimensions; differentiable prefix KV for VLM/joint",
                  sampling="90% motor: uniform condition/episode/10-step start; 10% native rehearsal: uniform episode/decision chunk" if data.native else "uniform motor condition/episode/chunk",pilot=args.pilot,
                  native_rehearsal_episodes=len(data.native),native_rehearsal_fraction=.1 if data.native else 0.,
                  prompt_sampling="uniform four equivalent motor-prefix expressions, independently of geometry and side",
                  student_privileged_state_inputs=False,safety_labels_in_training=False,
                  task_scope="motor prefixes: grasp, lift and transport, not terminal hazardous interaction",
                  resume_adapter=str(args.resume) if args.resume else None)
    snapshot=args.out/"source_snapshot";snapshot.mkdir(exist_ok=True)
    sources=[Path(__file__),Path(__file__).with_name("v6_scoped_lora.py"),Path(__file__).with_name("motor_lora.py"),
             Path(__file__).with_name("v6_lora_data.py"),Path(__file__).with_name("semantic_v6_diagnostics.py")]
    metadata["scripts"]={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    for path in sources:shutil.copy2(path,snapshot/path.name)
    write_json(args.out/"config.json",metadata)
    start=time.monotonic();warmup=min(200,max(5,args.steps//10));prefetch=ThreadPoolExecutor(max_workers=1)
    next_batch=prefetch.submit(data.batch,args.batch_size);first_gradient=None
    with (args.out/"training.jsonl").open("a",buffering=1) as log:
        for step in range(1,args.steps+1):
            items=fixed if args.fixed_batch else next_batch.result()
            if not args.fixed_batch and step<args.steps:next_batch=prefetch.submit(data.batch,args.batch_size)
            observation,actions=tensor_batch(items)
            # Make minibatch diffusion draws identical across the three adapter scopes.
            torch.manual_seed(args.seed+step)
            factor=min(1.,step/warmup)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-warmup)/max(1,args.steps-warmup))))
            optimizer.param_groups[0]["lr"]=args.learning_rate*factor
            optimizer.zero_grad(set_to_none=True)
            matrix=loss_matrix(model,observation,actions,args.scope)
            loss=matrix.mean();active=matrix[:,:,:7].mean()
            if not torch.isfinite(loss):raise FloatingPointError(f"Nonfinite loss at {step}")
            loss.backward()
            if step in [1,2,args.steps]:
                gradient=gradient_groups(model)
                if first_gradient is None:first_gradient=gradient
                for group in (["vision","language"] if args.scope=="vlm" else ["ae"] if args.scope=="ae" else ["vision","language","ae"]):
                    assert gradient["norms"][group]>0,(step,group,gradient)
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
            row=dict(step=step,loss=float(loss.detach()),active7_loss=float(active.detach()),gradient_norm=float(norm),
                     learning_rate=optimizer.param_groups[0]["lr"],elapsed_seconds=time.monotonic()-start,
                     cuda_peak_gb=torch.cuda.max_memory_allocated()/1e9)
            log.write(json.dumps(row)+"\n")
            if step==1 or step%20==0:print(json.dumps(row),flush=True)
            if step%args.save_every==0 or step==args.steps:
                with torch.no_grad():
                    held=loss_matrix(model,fixed_obs,fixed_actions,args.scope,noise=fixed_noise,time=fixed_time).mean().item()
                checkpoint=args.out/f"adapter_step_{step:05d}.pt"
                save_adapter(checkpoint,model,{**metadata,"step":step,"fixed_train_loss":held})
                write_json(args.out/"progress.json",{**row,"checkpoint":str(checkpoint),"fixed_train_loss":held})
    after=frozen_parameter_digest(model)
    with torch.no_grad():
        hidden_after=prefix(model,fixed_obs,gradients=False)[3].cpu()
        fixed_after=loss_matrix(model,fixed_obs,fixed_actions,args.scope,noise=fixed_noise,time=fixed_time).mean().item()
    audit=dict(all_original_weights_unchanged=before==after,frozen_parameter_sha256_after=after,
               fixed_input_vlm_hidden_exact=torch.equal(hidden_before,hidden_after),
               fixed_train_loss_before=fixed_before,fixed_train_loss_after=fixed_after,
               first_gradient=first_gradient,last_gradient=gradient,training_seconds=time.monotonic()-start,
               trainable_parameters=lora["trainable_parameters"])
    write_json(args.out/"freeze_audit.json",audit)
    assert audit["all_original_weights_unchanged"]
    if args.scope=="ae":assert audit["fixed_input_vlm_hidden_exact"]
    print("TRAINING_COMPLETE",json.dumps(audit),flush=True)


if __name__=="__main__":main()
