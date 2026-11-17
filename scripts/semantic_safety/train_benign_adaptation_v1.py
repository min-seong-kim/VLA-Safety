"""Matched benign-object/skill LoRA; training engine retained from the audited v5 experiment."""

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

from motor_lora import frozen_parameter_digest, make_input, save_adapter, tensor_batch
from benign_policy_v1 import load_policy, make_upright_input
from rollout_common import ROOT, write_json
from train_v6_scoped_lora import gradient_groups
from v6_scoped_lora import install, loss_matrix, prefix


class Demonstrations:
    def __init__(self,manifest,policy,seed):
        self.rows=[json.loads(line) for line in manifest.read_text().splitlines()]
        assert self.rows and all(r["accepted"] and r["split"]=="train" for r in self.rows)
        self.native={};self.groups={}
        for row in self.rows:
            if row["data_type"]=="native_rehearsal":
                self.native.setdefault((row.get("suite","libero_spatial"),row["task_id"]),[]).append(row)
            else:
                self.groups.setdefault(row["condition_index"],[]).append(row)
        assert self.native and self.groups
        self.keys=sorted(self.groups);self.native_keys=sorted(self.native)
        self.verb_keys={}
        for key,rows in self.groups.items():self.verb_keys.setdefault(rows[0]["verb"],[]).append(key)
        self.other_verbs=sorted(set(self.verb_keys)-{"place"})
        assert self.other_verbs and self.verb_keys.get("place")
        self.policy,self.seed=policy,seed
        self.pool=ThreadPoolExecutor(max_workers=8)

    def read(self,request):
        row,step=request
        with h5py.File(ROOT/row.get("training_cache",row["path"]),"r") as data:
            actions=data["action_chunks"][step] if row["data_type"]=="native_rehearsal" else data["actions"][step:step+10]
            prompt=row["instruction"]
            input_builder=make_input if row["data_type"]=="native_rehearsal" else make_upright_input
            value=input_builder(self.policy,data["observations/main_rgb_raw"][step],
                data["observations/wrist_rgb_raw"][step],data["observations/state"][step],prompt,actions)
            assert np.asarray(value["tokenized_prompt_mask"]).sum()<len(value["tokenized_prompt_mask"]),prompt
            return value

    def batch(self,size,step):
        # Per-update RNG makes all scopes and interrupted/resumed runs draw identical examples.
        rng=np.random.default_rng(self.seed+step*100003)
        requests=[];native_count=0
        for _ in range(size):
            native=bool(rng.random()<.1)
            if native:keys,groups=self.native_keys,self.native
            else:
                verb="place" if rng.random()<.5 else self.other_verbs[int(rng.integers(len(self.other_verbs)))]
                keys,groups=self.verb_keys[verb],self.groups
            rows=groups[keys[int(rng.integers(len(keys)))]]
            row=rows[int(rng.integers(len(rows)))]
            offset=int(rng.integers(row["chunks"] if native else row["steps"]-9))
            requests.append((row,offset));native_count+=int(native)
        return list(self.pool.map(self.read,requests)),native_count


def reconcile_training_log(folder, completed_step):
    """Keep only durable updates, retaining an exact copy of any interrupted log."""
    path=folder/"training.jsonl"
    if not path.exists():
        assert completed_step==0,"Checkpoint has no matching training log"
        return
    original=path.read_bytes()
    retained=[]
    for line in original.splitlines(keepends=True):
        try:
            row=json.loads(line)
        except (ValueError,UnicodeDecodeError):
            break
        if row["step"]>completed_step:
            break
        assert row["step"]==len(retained)+1,"Training log has a gap before the durable checkpoint"
        retained.append(line if line.endswith(b"\n") else line+b"\n")
    assert len(retained)==completed_step,"Training log ends before the durable checkpoint"
    durable=b"".join(retained)
    if durable!=original:
        archive=folder/"interrupted_logs";archive.mkdir(exist_ok=True)
        (archive/f"training_{time.time_ns()}.jsonl").write_bytes(original)
        temporary=path.with_suffix(".tmp.jsonl")
        temporary.write_bytes(durable);temporary.replace(path)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--scope",choices=["ae","vlm","all"],required=True)
    p.add_argument("--steps",type=int,default=16000)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--micro-batch-size",type=int,default=16)
    p.add_argument("--learning-rate",type=float,default=2e-4)
    p.add_argument("--save-every",type=int,default=4000)
    p.add_argument("--recovery-every",type=int,default=200)
    p.add_argument("--seed",type=int,default=20261116)
    p.add_argument("--pilot",action="store_true")
    p.add_argument("--init-adapter",type=Path)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    assert args.recovery_every>0 and args.save_every>0
    if (args.out/"freeze_audit.json").exists():
        prior=json.loads((args.out/"config.json").read_text())
        assert prior["steps_planned"]==args.steps and prior["lora"]["scope"]==args.scope
        assert prior["manifest_sha256"]==hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        assert json.loads((args.out/"freeze_audit.json").read_text())["all_original_weights_unchanged"]
        print("ALREADY_COMPLETE",str(args.out),flush=True)
        return
    torch.set_num_threads(8);torch.manual_seed(args.seed);np.random.seed(args.seed)
    policy,_=load_policy()
    model=policy._model
    architecture=install(model,args.scope)
    phase_start=0
    if args.init_adapter:
        initial=torch.load(args.init_adapter,map_location="cpu",weights_only=False)
        assert initial["metadata"]["lora"]==architecture
        named=dict(model.named_parameters())
        with torch.no_grad():
            for name,value in initial["state_dict"].items():named[name].copy_(value)
        phase_start=initial["metadata"].get("cumulative_step",initial["metadata"]["step"])
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=args.learning_rate,weight_decay=.01)
    data=Demonstrations(args.manifest,policy,args.seed)
    fixed,_=data.batch(2,-1)
    fixed_obs,fixed_actions=tensor_batch(fixed)
    torch.manual_seed(99127)
    fixed_noise=model.sample_noise(fixed_actions.shape,"cuda")
    fixed_time=torch.full((len(fixed),),.5,device="cuda")
    before=frozen_parameter_digest(model)
    with torch.no_grad():
        hidden_before=prefix(model,fixed_obs,gradients=False)[3].cpu().clone()
        fixed_before=loss_matrix(model,fixed_obs,fixed_actions,args.scope,noise=fixed_noise,time=fixed_time).mean().item()
    meta=dict(base_checkpoint="pi05_libero_pytorch",lora=architecture,seed=args.seed,
        steps_planned=args.steps,batch_size=args.batch_size,micro_batch_size=args.micro_batch_size,
        gradient_accumulation="sample-weighted microbatches; one optimizer update per effective batch",
        learning_rate_peak=args.learning_rate,
        manifest=str(args.manifest),manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        frozen_parameter_sha256_before=before,normalization="unchanged official pi05_libero",
        training_episodes=len(data.rows),conditions=data.keys,native_tasks=[list(k) for k in data.native_keys],
        native_rehearsal_fraction=.1,prompt_sampling="disjoint benign object/skill instructions; native original LIBERO instruction",
        sampling="10% native suite-task/episode/chunk; 90% benign with 50% place / 50% uniform other verb then condition/episode/10-step start",
        optimizer="AdamW",weight_decay=.01,gradient_clip=1.,rank=16,alpha=32,dropout=0,
        loss="flow matching over 32 dimensions; differentiable VLM prefix KV",pilot=args.pilot,
        student_privileged_pose_inputs=False,safety_labels_in_training=False,image_augmentation=False,
        task_scope="verified place and release, contact stroke, cover and press motor proxies; no actual cutting or cleaning effect",
        initialization_adapter=str(args.init_adapter) if args.init_adapter else None,
        initialization_adapter_sha256=hashlib.sha256(args.init_adapter.read_bytes()).hexdigest() if args.init_adapter else None,
        phase_start_cumulative_step=phase_start,
        code_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in [Path(__file__).name,"v6_scoped_lora.py","motor_lora.py","benign_policy_v1.py"]},
        recovery_every=args.recovery_every,
        resume="complete optimizer+adapter state; update-index deterministic sampling and diffusion noise")
    checkpoint_state=args.out/"training_state.pt"
    start_step=0;native_seen=0;elapsed_before=0.;first_gradient=None;gradient=None
    if checkpoint_state.exists():
        state=torch.load(checkpoint_state,map_location="cpu",weights_only=False)
        assert state["metadata"]==meta
        named=dict(model.named_parameters())
        with torch.no_grad():
            for name,value in state["adapter"].items():named[name].copy_(value)
        optimizer.load_state_dict(state["optimizer"])
        start_step=state["step"];native_seen=state["native_seen"]
        elapsed_before=state["elapsed_seconds"];first_gradient=state["first_gradient"]
        gradient=state["last_gradient"]
    elif (args.out/"training.jsonl").exists():
        raise FileExistsError("Training log exists without a resumable state: "+str(args.out))
    reconcile_training_log(args.out,start_step)
    snapshot=args.out/"source_snapshot";snapshot.mkdir(exist_ok=True)
    for name in [Path(__file__).name,"v6_scoped_lora.py","motor_lora.py","benign_policy_v1.py"]:
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    write_json(args.out/"config.json",meta)

    def save_recovery(step,elapsed):
        state=dict(metadata=meta,step=step,adapter={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},
            optimizer=optimizer.state_dict(),native_seen=native_seen,elapsed_seconds=elapsed,first_gradient=first_gradient,
            last_gradient=gradient)
        temporary=checkpoint_state.with_suffix(".tmp.pt")
        torch.save(state,temporary);temporary.replace(checkpoint_state)

    if not checkpoint_state.exists():
        save_recovery(0,0.)
    prefetch=ThreadPoolExecutor(max_workers=1)
    next_batch=prefetch.submit(data.batch,args.batch_size,start_step+1)
    started=time.monotonic();warmup=300;elapsed=elapsed_before
    with (args.out/"training.jsonl").open("a",buffering=1) as log:
        for step in range(start_step+1,args.steps+1):
            items,native_count=next_batch.result()
            if step<args.steps:next_batch=prefetch.submit(data.batch,args.batch_size,step+1)
            torch.manual_seed(args.seed+step)
            noise=model.sample_noise((len(items),*np.asarray(items[0]["actions"]).shape),"cuda")
            diffusion_time=model.sample_time(len(items),"cuda")
            factor=min(1.,step/warmup)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-warmup)/max(1,args.steps-warmup))))
            optimizer.param_groups[0]["lr"]=args.learning_rate*factor
            optimizer.zero_grad(set_to_none=True)
            loss_value=0.;active_value=0.
            for offset in range(0,len(items),args.micro_batch_size):
                stop=min(len(items),offset+args.micro_batch_size)
                observation,actions=tensor_batch(items[offset:stop])
                matrix=loss_matrix(model,observation,actions,args.scope,
                                  noise=noise[offset:stop],time=diffusion_time[offset:stop])
                weight=(stop-offset)/len(items)
                loss=matrix.mean()*weight
                if not torch.isfinite(loss):raise FloatingPointError(f"Nonfinite loss at {step}")
                loss_value+=float(loss.detach())
                active_value+=float(matrix[:,:,:7].mean().detach())*weight
                loss.backward()
                del matrix,loss,observation,actions
            if step in [start_step+1,args.steps]:
                gradient=gradient_groups(model)
                if first_gradient is None:first_gradient=gradient
                for group in (["ae"] if args.scope=="ae" else ["vision","language"] if args.scope=="vlm" else ["vision","language","ae"]):
                    assert gradient["norms"][group]>0,(step,group,gradient)
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            optimizer.step();native_seen+=native_count
            elapsed=elapsed_before+time.monotonic()-started
            row=dict(step=step,loss=loss_value,active7_loss=active_value,
                gradient_norm=float(norm),learning_rate=optimizer.param_groups[0]["lr"],elapsed_seconds=elapsed,
                native_samples_seen=native_seen,native_sample_fraction=native_seen/(step*args.batch_size),
                cuda_peak_gb=torch.cuda.max_memory_allocated()/1e9)
            log.write(json.dumps(row)+"\n")
            if step==start_step+1 or step%200==0:print(json.dumps(row),flush=True)
            if step%args.save_every==0 or step==args.steps:
                with torch.no_grad():
                    held=loss_matrix(model,fixed_obs,fixed_actions,args.scope,noise=fixed_noise,time=fixed_time).mean().item()
                path=args.out/f"adapter_step_{step:05d}.pt"
                save_adapter(path,model,{**meta,"step":step,"cumulative_step":phase_start+step,"fixed_train_loss":held})
                save_recovery(step,elapsed)
                write_json(args.out/"progress.json",{**row,"checkpoint":str(path),"fixed_train_loss":held})
            elif step%args.recovery_every==0:
                save_recovery(step,elapsed)
                write_json(args.out/"progress.json",{**row,"recovery_checkpoint":str(checkpoint_state)})
    after=frozen_parameter_digest(model)
    with torch.no_grad():
        hidden_after=prefix(model,fixed_obs,gradients=False)[3].cpu()
        fixed_after=loss_matrix(model,fixed_obs,fixed_actions,args.scope,noise=fixed_noise,time=fixed_time).mean().item()
    audit=dict(all_original_weights_unchanged=before==after,fixed_input_vlm_hidden_exact=torch.equal(hidden_before,hidden_after),
        fixed_train_loss_before=fixed_before,fixed_train_loss_after=fixed_after,first_gradient=first_gradient,
        last_gradient=gradient,training_seconds=elapsed,native_sample_fraction=native_seen/(args.steps*args.batch_size),
        trainable_parameters=architecture["trainable_parameters"],completed_steps=args.steps)
    assert audit["all_original_weights_unchanged"]
    if args.scope=="ae":assert audit["fixed_input_vlm_hidden_exact"]
    write_json(args.out/"freeze_audit.json",audit)
    print("TRAINING_COMPLETE",json.dumps(audit),flush=True)


if __name__=="__main__":
    main()
