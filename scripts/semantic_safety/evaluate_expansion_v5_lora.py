"""Batched official-policy rollouts on frozen v5 motor proxies or native LIBERO tasks."""

import argparse
import hashlib
import json
import multiprocessing as mp
from multiprocessing.connection import Client
from pathlib import Path
import random
import time
import traceback

import cv2
import h5py
import numpy as np

from expansion_v5_motor import DATA, ROOT, load_jsonl, make_environment, sha, write_jsonl
from evaluate_v6_lora import AUTHKEY, append, receive, request
from rollout_common import bootstrap, observation_payload, write_json

SUITES = ["libero_spatial","libero_object","libero_goal","libero_10"]


def native_environment(item,resolution=224):
    bootstrap()
    import torch
    from libero.libero import benchmark
    from libero.libero.envs.env_wrapper import ControlEnv
    suite=benchmark.get_benchmark_dict()[item["suite"]]()
    task=suite.get_task(item["task_id"])
    base=ROOT/"third_party/libero/libero/libero"
    np.random.seed(0);random.seed(0)
    wrapper=ControlEnv(bddl_file_name=str(base/"bddl_files"/task.problem_folder/task.bddl_file),
        camera_heights=max(16,resolution),camera_widths=max(16,resolution),use_camera_obs=resolution>0,
        has_offscreen_renderer=resolution>0,has_renderer=False,hard_reset=False,ignore_done=True)
    wrapper.seed(0);wrapper.env.reset()
    states=torch.load(base/"init_files"/task.problem_folder/task.init_states_file,weights_only=False)
    obs=wrapper.set_init_state(states[item["initial_state_id"]])
    for _ in range(10):
        obs,_,_,_=wrapper.env.step([0.]*6+[-1.])
    return wrapper,obs,task.language


def native_items(split):
    state_ids=range(2,6) if split=="rehearsal" else [7] if split=="validation" else range(2)
    return [dict(kind="native",suite=suite,task_id=i,initial_state_id=j,
                 item_id=f"{split}_{suite}_{i:02d}_{j:02d}",split=split,
                 noise_seed=72000000+SUITES.index(suite)*10000+i*100+j)
            for suite in SUITES for i in range(10) for j in state_ids]


class Episode:
    def __init__(self,item,folder,horizon,run_digest):
        self.item,self.folder,self.horizon,self.run_digest=item,folder,horizon,run_digest
        folder.mkdir(parents=True,exist_ok=True)
        self.native=item.get("kind")=="native"
        if self.native:
            self.wrapper,self.obs,self.instruction=native_environment(item)
            self.oracle=None
        else:
            self.wrapper,self.obs,self.oracle=make_environment(item,224)
            self.instruction=item["instruction"]
        self.env=self.wrapper.env
        self.steps=0;self.success=False;self.started=time.monotonic();self.unstable=False
        self.metric={} if self.native else self.oracle.measure(self.env)
        self.initial_success=bool(self.env._check_success()) if self.native else self.metric["proxy_success"]
        assert not self.initial_success
        self.data=h5py.File(folder/"trajectory.h5","w")
        self.data.attrs["item_json"]=json.dumps(item,ensure_ascii=False)
        self.data.attrs["instruction"]=self.instruction
        self.data.attrs["run_digest"]=run_digest
        self.data.attrs["alignment"]="decision input at t -> executed actions[t:t+10]; sim state stored before and after every action"
        self.events=(folder/"events.jsonl").open("w",buffering=1)
        self.video=cv2.VideoWriter(str(folder/"views.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),5,(448,224))
        (folder/"model.xml").write_text(self.env.sim.model.get_xml())
        self.initial_image_sha=hashlib.sha256(observation_payload(self.obs)["image"].tobytes()).hexdigest()
        self.record_state();self.frame();self.save_image("initial.png")

    def save_image(self,name):
        cv2.imwrite(str(self.folder/name),np.ascontiguousarray(self.obs["agentview_image"][::-1,::-1,::-1]))

    def record_state(self):
        payload=observation_payload(self.obs)
        state=self.env.sim.get_state().flatten()
        assert np.isfinite(state).all()
        append(self.data,"observations/state",payload["state"])
        append(self.data,"observations/sim_state",state)

    def frame(self):
        image=np.concatenate([self.obs["agentview_image"][::-1,::-1],self.obs["robot0_eye_in_hand_image"][::-1,::-1]],axis=1)
        self.video.write(np.ascontiguousarray(image[:,:,::-1]))

    def packet(self):
        payload=observation_payload(self.obs)
        for key,name in [("image","main_rgb_raw"),("wrist_image","wrist_rgb_raw"),("state","state")]:
            append(self.data,"policy_inputs/"+name,payload[key])
        append(self.data,"chunk_start_steps",np.int32(self.steps))
        return dict(observation=payload,instruction=self.instruction,seed=self.item["noise_seed"]+self.steps)

    def advance(self,chunk):
        append(self.data,"predicted_action_chunks",chunk)
        for raw in chunk[:10]:
            action=np.asarray(raw,dtype=np.float32).copy()
            action[:6]=np.clip(action[:6],-1,1)
            action[6]=1. if action[6]>0 else -1.
            append(self.data,"actions/executed",action)
            previous_time=self.env.sim.data.time
            self.obs,_,_,_=self.env.step(action)
            self.steps+=1
            self.unstable |= (self.env.sim.data.time<=previous_time or not np.isfinite(self.env.sim.data.qvel).all()
                              or float(np.max(np.abs(self.env.sim.data.qvel)))>100.)
            if self.native:
                self.success=bool(self.env._check_success())
                self.metric=dict(native_success=self.success)
            else:
                self.metric=self.oracle.measure(self.env)
                self.success=self.metric["proxy_success"]
            if self.unstable:
                self.success=False
                self.metric.update(physics_unstable=True)
            self.events.write(json.dumps(dict(control_step=self.steps,**self.metric))+"\n")
            self.record_state()
            if self.steps%4==0:self.frame()
            if self.steps>=self.horizon or self.success or self.unstable:break
        self.data.flush()
        return self.steps>=self.horizon or self.success or self.unstable

    def finish(self):
        self.save_image("final.png")
        self.video.release();self.data.close();self.events.close();self.wrapper.close()
        result=dict(item_id=self.item["item_id"],source_scene_id=self.item.get("source_scene_id"),
            condition_index=self.item.get("condition_index"),pair_group=self.item.get("pair_group"),
            suite=self.item.get("suite"),task_id=self.item.get("task_id"),initial_state_id=self.item.get("initial_state_id"),
            success=self.success,native=self.native,skill=self.item.get("skill"),steps=self.steps,
            physics_stable=not self.unstable,horizon=self.horizon,
            instruction=self.instruction,run_digest=self.run_digest,initial_image_sha256=self.initial_image_sha,
            initial_success=self.initial_success,final_metric=self.metric,seconds=time.monotonic()-self.started,
            full_instruction_success=self.success if self.native else None,harm_observed=None)
        write_json(self.folder/"summary.json",result)
        return result


def worker(connection):
    bootstrap();cv2.setNumThreads(1)
    episode=None
    try:
        while True:
            message=connection.recv()
            if message["op"]=="close":break
            if message["op"]=="start":
                episode=Episode(message["item"],Path(message["folder"]),message["horizon"],message["run_digest"])
                done=False
            else:
                done=episode.advance(message["actions"])
            if done:
                summary=episode.finish();episode=None
                connection.send(dict(done=True,summary=summary))
            else:
                connection.send(dict(done=False,step=episode.steps,packet=episode.packet()))
    except Exception:
        connection.send(dict(error=traceback.format_exc()))
    finally:
        if episode is not None:
            episode.video.release();episode.data.close();episode.events.close();episode.wrapper.close()
        connection.close()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--socket",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--items",type=Path)
    p.add_argument("--native",choices=["rehearsal","validation","test"])
    p.add_argument("--adapter",type=Path)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--horizon",type=int,default=460)
    p.add_argument("--limit",type=int)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    items=native_items(args.native) if args.native else load_jsonl(args.items)
    if args.limit:items=items[:args.limit]
    source=Path(__file__)
    run=dict(items=items,adapter=str(args.adapter) if args.adapter else None,
        adapter_sha256=sha(args.adapter) if args.adapter else None,horizon=args.horizon,batch_size=args.batch_size,
        code={p.name:sha(p) for p in ([source] if args.native else [source,source.with_name("expansion_v5_motor.py")])},
        asset_manifest_sha256=None if args.native else sha(DATA/"assets/manifest.json"),chunk_steps=10,
        scope="native task success or explicitly declared v5 motor proxy",evaluation_not_teacher_filtered=True)
    digest=hashlib.sha256(json.dumps(run,sort_keys=True).encode()).hexdigest()
    run["digest"]=digest
    if (args.out/"run.json").exists():assert json.loads((args.out/"run.json").read_text())["digest"]==digest
    write_json(args.out/"run.json",run)
    pending=[]
    for item in items:
        receipt=args.out/"episodes"/item["item_id"]/"summary.json"
        if receipt.exists():assert json.loads(receipt.read_text())["run_digest"]==digest
        else:pending.append(item)
    context=mp.get_context("spawn");workers=[]
    for _ in range(min(args.batch_size,len(pending))):
        parent,child=context.Pipe();proc=context.Process(target=worker,args=(child,),daemon=True)
        proc.start();child.close();workers.append((parent,proc))
    completed=len(items)-len(pending)
    try:
        with Client(str(args.socket),family="AF_UNIX",authkey=AUTHKEY) as connection:
            assert receive(connection)["ready"]
            hello=request(connection,dict(op="activate",adapter=str(args.adapter) if args.adapter else None))
            write_json(args.out/"model.json",hello["metadata"])
            for offset in range(0,len(pending),args.batch_size):
                group=pending[offset:offset+args.batch_size]
                for (parent,_),item in zip(workers,group):
                    parent.send(dict(op="start",item=item,folder=str(args.out/"episodes"/item["item_id"]),
                        horizon=args.horizon if item.get("suite")!="libero_10" else max(600,args.horizon),run_digest=digest))
                active={i:receive(workers[i][0]) for i in range(len(group))}
                while active:
                    indices=list(active)
                    response=request(connection,dict(op="predict",items=[active[i]["packet"] for i in indices]))
                    write_json(args.out/"inference_audit.json",response["reference_audit"])
                    for i,actions in zip(indices,response["actions"]):
                        workers[i][0].send(dict(op="advance",actions=actions))
                    for i in indices:
                        state=receive(workers[i][0])
                        if state["done"]:
                            completed+=1;summary=state["summary"];del active[i]
                            print(json.dumps(dict(completed=completed,total=len(items),item=summary["item_id"],success=summary["success"])),flush=True)
                        else:active[i]=state
                write_json(args.out/"progress.json",dict(completed=completed,total=len(items),updated_unix=time.time()))
            request(connection,dict(op="disconnect"))
        rows=[json.loads((args.out/"episodes"/item["item_id"]/"summary.json").read_text()) for item in items]
        write_jsonl(args.out/"episodes.jsonl",rows)
        write_json(args.out/"completion.json",dict(complete=True,episodes=len(rows),successes=sum(r["success"] for r in rows)))
    finally:
        for parent,proc in workers:
            if proc.is_alive():
                try:parent.send(dict(op="close"))
                except (BrokenPipeError,EOFError):pass
                proc.join(timeout=10)
                if proc.is_alive():proc.terminate();proc.join(timeout=5)
            parent.close()


if __name__=="__main__":
    main()
