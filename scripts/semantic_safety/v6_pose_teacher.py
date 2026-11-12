"""Object-relative grasp teacher and physically settled initial-pose augmentation."""

import json
import math

import numpy as np
from scipy.spatial.transform import Rotation

from semantic_pairs_v4 import current_observation
from semantic_pairs_v6 import TransportOracle, set_actor_position
from v6_lora_data import DATA, make_environment as original_environment, make_spec as original_spec, world_bounds
from v6_transport_teacher import Teacher as OriginalTeacher


class Teacher(OriginalTeacher):
    def __init__(self,env,oracle,spec,profile):
        if profile["mode"]!="object_pose":
            super().__init__(env,oracle,spec,profile);return
        origin=env.sim.data.body_xpos[oracle.actor].copy()
        body_r=env.sim.data.body_xmat[oracle.actor].reshape(3,3).copy()
        offset=body_r@np.array(profile["position_local"])
        unrotated=Rotation.from_rotvec([0.,0.,-spec["actor_yaw"]]).apply(offset)
        proxy=dict(mode="legacy",offset=unrotated[:2].tolist(),height=float(offset[2]+.0035),yaw=0.)
        super().__init__(env,oracle,spec,proxy)
        self.profile=profile;self.offset=offset
        self.orientation=body_r@np.array(profile["rotation_local"])


def profiles(asset):
    result=json.loads((DATA/"development_grasp_geometry.json").read_text()).get(asset,[])
    if asset in {"battery","wood_bowl","wallet"}:
        result=[]
        for yaw in [0.,-math.pi/2]:
            for height in [.8,.6,.4]:
                result.append(dict(mode="bounds",offset=[0.,0.],height_fraction=height,z_adjust=-.0035,yaw=yaw))
    return result


def make_spec(condition,split,index):
    spec=original_spec(condition,split,index)
    spec["item_id"]="pose__"+spec["item_id"];spec["pair_group"]="pose__"+spec["pair_group"]
    rotation={"battery":[math.pi/2,0.,0.],"wood_bowl":[math.pi,0.,0.],"wallet":[0.,math.pi/2,0.]}
    spec["initial_pose_augmentation"]=rotation.get(spec["actor_asset"])
    spec["teacher_family"]="object_relative_grasp_with_optional_initial_pose_augmentation"
    return spec


def make_environment(spec,resolution=224):
    wrapper,obs,oracle=original_environment(spec,resolution)
    rotation=spec.get("initial_pose_augmentation")
    if rotation is None:return wrapper,obs,oracle
    env=wrapper.env;actor=oracle.actor
    r=Rotation.from_rotvec([0.,0.,spec["actor_yaw"]])*Rotation.from_euler("xyz",rotation)
    xyzw=r.as_quat();quat=np.r_[xyzw[3],xyzw[:3]]
    pos=env.sim.data.body_xpos[actor].copy();pos[2]=1.15
    set_actor_position(env,pos,quat);low,_=world_bounds(env,actor)
    pos[2]+=.902-low[2];set_actor_position(env,pos,quat)
    for _ in range(100):env.step([0.]*6+[-1.])
    env.sim.forward()
    pos=env.sim.data.body_xpos[actor].copy()
    reference=env.sim.data.body_xmat[actor].reshape(3,3)@np.array(spec["actor_reference_local"])
    pos[:2]=np.array(spec["actor_xy"])-reference[:2];set_actor_position(env,pos)
    return wrapper,current_observation(env),TransportOracle(env,spec)


def configure():
    import v6_transport_teacher as teacher
    import collect_v6_lora_data as collector
    teacher.Teacher=Teacher;teacher.make_environment=make_environment
    collector.candidate_profiles=profiles;collector.make_spec=make_spec
    collector.__file__=__file__


def process(request):
    configure()
    import collect_v6_lora_data
    return collect_v6_lora_data.process(request)


def main():
    import argparse
    import multiprocessing
    from pathlib import Path
    from rollout_common import write_json
    from semantic_pairs_v6 import sha256
    p=argparse.ArgumentParser(__doc__);p.add_argument("--split",default="calibration")
    p.add_argument("--count",type=int,default=1);p.add_argument("--start",type=int,default=1000)
    p.add_argument("--maximum",type=int,default=4);p.add_argument("--workers",type=int,default=6)
    p.add_argument("--conditions",default="4,5,6,15,40,41,51,59");p.add_argument("--tag",default="pose_pilot")
    args=p.parse_args();indices=[int(x) for x in args.conditions.split(",")]
    requests=[(c,args.split,args.count,args.start,args.maximum,args.tag) for c in indices]
    with multiprocessing.get_context("spawn").Pool(args.workers,maxtasksperchild=2) as pool:
        results=list(pool.imap_unordered(process,requests))
    rows=sorted([r for result in results for r in result["accepted"]],key=lambda r:r["item_id"])
    output=DATA/args.tag;output.mkdir(parents=True,exist_ok=True)
    (output/f"{args.split}_manifest.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    write_json(output/f"{args.split}_summary.json",dict(episodes=len(rows),complete=all(r["complete"] for r in results),conditions=results))
    write_json(output/"provenance.json",dict(pose_teacher_sha256=sha256(Path(__file__)),
               collector_sha256=sha256(Path(__file__).with_name("collect_v6_lora_data.py")),
               mined_grasp_geometry_sha256=sha256(DATA/"development_grasp_geometry.json"),
               note="new physical rollouts; optional pose change only during scene initialization; no evaluation actions or images used as training samples"))
    print(json.dumps(dict(episodes=len(rows),complete=all(r["complete"] for r in results))),flush=True)


if __name__=="__main__":
    main()
