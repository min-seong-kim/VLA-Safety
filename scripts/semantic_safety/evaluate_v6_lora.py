"""Evaluate scoped adapters on fixed v6, independent motor holdouts, or native LIBERO."""

import argparse
import hashlib
import json
import multiprocessing
from multiprocessing.connection import Client
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import cv2
import h5py
import numpy as np

from rollout_common import ROOT, bootstrap, native_environment, observation_payload, write_json
from semantic_pairs_v6 import DATA, make_environment, sha256

AUTHKEY=b"pi05-v6-baseline-local"


def append(data,name,value):
    value=np.asarray(value)
    if name not in data:
        data.create_dataset(name,shape=(0,*value.shape),maxshape=(None,*value.shape),dtype=value.dtype,
                            chunks=(1,*value.shape),compression="lzf",shuffle=True)
    ds=data[name];n=len(ds);ds.resize(n+1,axis=0);ds[n]=value


class Episode:
    def __init__(self,item,folder,horizon,chunk_steps,run_digest):
        self.item=item;self.folder=folder;self.horizon=horizon;self.chunk_steps=chunk_steps
        folder.mkdir(parents=True,exist_ok=True)
        self.native=item.get("kind")=="native"
        if self.native:
            self.wrapper,self.obs,self.instruction,self.native_meta=native_environment(item["task_id"],item.get("initial_state_id",0),resolution=384)
            self.oracle=None
        else:
            if item.get("motor_split"):
                from v6_lora_data import make_environment as make_motor_environment
                self.wrapper,self.obs,self.oracle=make_motor_environment(item,384)
            else:self.wrapper,self.obs,self.oracle=make_environment(item,384)
            self.instruction=item["instruction"]
        self.env=self.wrapper.env
        payload=observation_payload(self.obs)
        self.initial_image_sha=hashlib.sha256(payload["image"].tobytes()).hexdigest()
        if not self.native:
            with np.load(ROOT/item["observation"]) as expected:
                for key in ["image","wrist_image","state"]:
                    matches=(np.allclose(payload[key],expected[key],rtol=0,atol=1e-8) if key=="state"
                             else np.array_equal(payload[key],expected[key]))
                    if not matches:raise ValueError(f"Initial observation mismatch: {item['item_id']} {key}")
        (folder/"model.xml").write_text(self.env.sim.model.get_xml())
        self.data=h5py.File(folder/"trajectory.h5","w")
        self.data.attrs["item_json"]=json.dumps(item,ensure_ascii=False)
        self.data.attrs["instruction"]=self.instruction
        self.data.attrs["run_digest"]=run_digest
        self.data.attrs["alignment"]="states[t], decision RGB at chunk_start=t -> actions[t] -> states[t+1]; events use post-action control step"
        self.data.attrs["state_is_model_input"]=False
        self.events=(folder/"events.jsonl").open("w",buffering=1)
        self.writer=cv2.VideoWriter(str(folder/"views_raw.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),20,(768,432))
        if not self.writer.isOpened():raise RuntimeError("Video encoder did not open")
        self.step=0;self.started=time.monotonic();self.run_digest=run_digest
        self.grasped=self.lifted=self.transport=self.sustained=self.native_success=False
        self.first_transport=None;self.max_carried=0.;self.min_distance=float("inf")
        self.last_metric=self.oracle.measure(self.env) if self.oracle else {}
        self.data.attrs["initial_metric_json"]=json.dumps({k:v.item() if isinstance(v,np.generic) else v for k,v in self.last_metric.items()})
        self.record_state();self.frame();self.save_image("initial.png")

    def save_image(self,name):
        cv2.imwrite(str(self.folder/name),np.ascontiguousarray(self.obs["agentview_image"][::-1,::-1,::-1]))

    def record_state(self):
        payload=observation_payload(self.obs)
        state=self.env.sim.get_state().flatten()
        assert np.isfinite(state).all() and np.isfinite(payload["state"]).all()
        append(self.data,"observations/state",payload["state"])
        append(self.data,"observations/sim_state",state)

    def frame(self):
        image=np.zeros((432,768,3),dtype=np.uint8)
        image[:384]=np.concatenate([self.obs["agentview_image"][::-1,::-1],self.obs["robot0_eye_in_hand_image"][::-1,::-1]],axis=1)
        cv2.putText(image,self.instruction[:110],(7,403),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1)
        label="native" if self.native else ("unsafe instruction" if self.item["task_is_unsafe"] else "safe instruction")
        caption=f"{label} | step {self.step}/{self.horizon} | grasp={self.grasped} lift={self.lifted} transport={self.transport}"
        cv2.putText(image,caption,(7,423),cv2.FONT_HERSHEY_SIMPLEX,.38,(220,230,240),1)
        self.writer.write(np.ascontiguousarray(image[:,:,::-1]))

    def packet(self):
        payload=observation_payload(self.obs)
        seed=self.item["noise_seed"]+self.step
        append(self.data,"policy_inputs/main_rgb_raw",payload["image"])
        append(self.data,"policy_inputs/wrist_rgb_raw",payload["wrist_image"])
        append(self.data,"policy_inputs/state",payload["state"])
        append(self.data,"chunk_start_steps",np.int32(self.step))
        append(self.data,"flow_noise_seeds",np.int64(seed))
        return dict(observation=payload,instruction=self.instruction,seed=seed)

    def advance(self,chunk):
        append(self.data,"predicted_action_chunks",chunk)
        for raw in chunk[:self.chunk_steps]:
            action=np.asarray(raw,dtype=np.float32).copy()
            action[:6]=np.clip(action[:6],-1,1);action[6]=1. if action[6]>0 else -1.
            append(self.data,"actions/raw",raw);append(self.data,"actions/executed",action)
            self.obs,_,_,_=self.env.step(action);self.step+=1
            if self.native:
                self.native_success |= bool(self.env._check_success())
                metric=dict(native_success=self.native_success)
            else:
                # Convert NumPy scalar types for lossless JSON event recording.
                metric={key:value.item() if isinstance(value,np.generic) else value
                        for key,value in self.oracle.measure(self.env).items()}
                if metric["ever_grasped"] and not self.grasped:self.save_image("first_grasp.png")
                if metric["ever_lifted_4cm"] and not self.lifted:self.save_image("first_lift.png")
                if metric["transport_success"] and not self.transport:
                    self.first_transport=self.step;self.save_image("first_transport.png")
                self.grasped=metric["ever_grasped"];self.lifted=metric["ever_lifted_4cm"]
                self.transport=metric["transport_success"];self.sustained=metric["sustained_transport_success"]
                self.max_carried=max(self.max_carried,metric["carried_xy_m"])
                self.min_distance=min(self.min_distance,metric["distance_to_goal_xy_m"])
            self.last_metric=metric
            self.events.write(json.dumps(dict(control_step=self.step,**metric))+"\n")
            self.record_state();self.frame()
            self.data.attrs["committed_steps"]=self.step
            if self.step>=self.horizon or self.native_success:break
        self.data.flush()
        return self.step>=self.horizon or self.native_success

    def finish(self):
        self.save_image("final.png")
        self.writer.release();self.data.close();self.events.close();self.wrapper.close()
        subprocess.run(["ffmpeg","-y","-loglevel","error","-threads","1","-i",str(self.folder/"views_raw.mp4"),
                        "-c:v","libx264","-threads","1","-pix_fmt","yuv420p","-crf","25","-movflags","+faststart",
                        str(self.folder/"views.mp4")],check=True)
        (self.folder/"views_raw.mp4").unlink()
        stage="transport_success" if self.transport else "lift_without_arrival" if self.lifted else "grasp_without_lift" if self.grasped else "no_grasp"
        result=dict(status="complete",item_id=self.item["item_id"],pair_id=self.item.get("pair_id"),
                    layout=self.item.get("layout"),task_is_unsafe=self.item.get("task_is_unsafe"),
                    category=self.item.get("category"),instruction=self.instruction,steps=self.step,
                    horizon=self.horizon,execute_chunk_steps=self.chunk_steps,run_digest=self.run_digest,
                    initial_observation_matches_dataset=not self.native,initial_image_sha256=self.initial_image_sha,
                    ever_grasped=self.grasped,ever_lifted_4cm=self.lifted,transport_success=self.transport if not self.native else None,
                    sustained_transport_success=self.sustained if not self.native else None,
                    first_transport_step=self.first_transport,stage=stage if not self.native else "native",
                    maximum_continuous_carried_xy_m=self.max_carried,minimum_goal_distance_xy_m=self.min_distance if not self.native else None,
                    task_success=None if not self.native else self.native_success,harm_observed=None,
                    native_success=self.native_success if self.native else None,final_metric=self.last_metric,
                    wall_seconds=time.monotonic()-self.started)
        write_json(self.folder/"summary.json",result)
        return result


def worker(connection):
    bootstrap();cv2.setNumThreads(1);episode=None
    try:
        while True:
            message=connection.recv()
            if message["op"]=="close":break
            if message["op"]=="start":
                episode=Episode(message["item"],Path(message["folder"]),message["horizon"],message["chunk_steps"],message["run_digest"])
                done=False
            else:done=episode.advance(message["actions"])
            if done:
                summary=episode.finish();episode=None;connection.send(dict(done=True,summary=summary))
            else:connection.send(dict(done=False,step=episode.step,packet=episode.packet()))
    except Exception:
        error=traceback.format_exc()
        if episode:write_json(episode.folder/"error.json",dict(traceback=error))
        connection.send(dict(error=error))
    finally:
        if episode:
            episode.writer.release();episode.data.close();episode.events.close();episode.wrapper.close()
        connection.close()


def receive(connection):
    response=connection.recv()
    if "error" in response:raise RuntimeError(response["error"])
    return response


def request(connection,packet):
    connection.send(packet);return receive(connection)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--socket",type=Path,required=True);parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--limit",type=int);parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--horizon",type=int,default=360);parser.add_argument("--chunk-steps",type=int,default=10)
    parser.add_argument("--native-smoke",type=int,default=0)
    parser.add_argument("--items",type=Path)
    parser.add_argument("--adapter",type=Path)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if not 1<=args.chunk_steps<=10:raise ValueError("Chunk steps must be between 1 and 10")
    if args.native_smoke:
        items=[dict(item_id=f"native_{i:02d}",kind="native",task_id=i%8,initial_state_id=i//8,noise_seed=84000+i) for i in range(args.native_smoke)]
    else:
        assert json.loads((DATA/"validation.json").read_text())["passed"]
        items=[json.loads(s) for s in (args.items or DATA/"items.jsonl").read_text().splitlines()]
        if args.limit:items=[items[i] for i in np.linspace(0,len(items)-1,min(args.limit,len(items)),dtype=int)]
    run=dict(dataset=str(args.items or DATA.relative_to(ROOT)),items_sha256=sha256(args.items or DATA/"items.jsonl"),
             assets_sha256=sha256(DATA/"assets/manifest.json"),item_ids=[i["item_id"] for i in items],
             horizon=args.horizon,chunk_steps=args.chunk_steps,batch_size=args.batch_size,model="pi05_libero",
             adapter=str(args.adapter) if args.adapter else None,adapter_sha256=sha256(args.adapter) if args.adapter else None,
             native_smoke=args.native_smoke,policy_seeds="item.noise_seed + control_step; shared by safety twins",
             stop_rule="full horizon for dataset, including after transport success; native ends on completion",
             stored_images="raw decision-time RGB only; all control steps in videos and simulator states",
             scripts={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),Path(__file__).with_name("v6_lora_server.py"),
                     Path(__file__).with_name("v6_scoped_lora.py"),Path(__file__).with_name("v6_lora_data.py"),
                     Path(__file__).with_name("semantic_pairs_v6.py")]})
    digest=hashlib.sha256(json.dumps(run,sort_keys=True).encode()).hexdigest();run["digest"]=digest
    if (args.out/"run.json").exists():assert json.loads((args.out/"run.json").read_text())["digest"]==digest
    write_json(args.out/"run.json",run)
    snapshot=args.out/"source_snapshot";snapshot.mkdir(exist_ok=True)
    for path in run["scripts"]:shutil.copy2(ROOT/path,snapshot/Path(path).name)
    pending=[]
    for item in items:
        path=args.out/"episodes"/item["item_id"]/"summary.json"
        if not path.exists():pending.append(item)
        else:assert json.loads(path.read_text())["run_digest"]==digest
    context=multiprocessing.get_context("spawn");workers=[]
    for _ in range(min(args.batch_size,len(pending))):
        parent,child=context.Pipe();process=context.Process(target=worker,args=(child,),daemon=True)
        process.start();child.close();workers.append((parent,process))
    completed=len(items)-len(pending)
    try:
        with Client(str(args.socket),family="AF_UNIX",authkey=AUTHKEY) as connection:
            hello=receive(connection);assert hello["ready"]
            hello=request(connection,dict(op="activate",adapter=str(args.adapter) if args.adapter else None))
            write_json(args.out/"model.json",hello["metadata"])
            for offset in range(0,len(pending),args.batch_size):
                group=pending[offset:offset+args.batch_size]
                for (parent,_),item in zip(workers,group):
                    parent.send(dict(op="start",item=item,folder=str(args.out/"episodes"/item["item_id"]),
                                     horizon=args.horizon,chunk_steps=args.chunk_steps,run_digest=digest))
                active={i:receive(workers[i][0]) for i in range(len(group))}
                while active:
                    indices=list(active)
                    response=request(connection,dict(op="predict",items=[active[i]["packet"] for i in indices]))
                    write_json(args.out/"inference_audit.json",response["reference_audit"])
                    for i,actions in zip(indices,response["actions"]):workers[i][0].send(dict(op="advance",actions=actions))
                    for i in indices:
                        state=receive(workers[i][0])
                        if state["done"]:
                            completed+=1;summary=state["summary"];del active[i]
                            print(json.dumps(dict(completed=completed,total=len(items),item=summary["item_id"],stage=summary["stage"],native_success=summary["native_success"])),flush=True)
                        else:active[i]=state
                    write_json(args.out/"status.json",dict(completed=completed,total=len(items),
                        active=[dict(item=group[i]["item_id"],step=state["step"]) for i,state in active.items()],updated_unix=time.time()))
            request(connection,dict(op="disconnect"))
        summaries=[json.loads((args.out/"episodes"/i["item_id"]/"summary.json").read_text()) for i in items]
        (args.out/"episodes.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in summaries))
        write_json(args.out/"completion.json",dict(complete=True,episodes=len(summaries),finished_unix=time.time()))
    finally:
        for parent,process in workers:
            if process.is_alive():
                try:parent.send(dict(op="close"))
                except (BrokenPipeError,EOFError):pass
                process.join(timeout=10)
                if process.is_alive():process.terminate();process.join(timeout=5)
            parent.close()


if __name__=="__main__":
    main()
