"""Collect front-region transports with additional reachable training layouts."""

import argparse
import json
import multiprocessing
from pathlib import Path

import numpy as np

from collect_v6_fast import configure as fast_configure
from rollout_common import write_json
from semantic_pairs_v6 import sha256
from v6_lora_data import DATA, make_spec as base_spec
from v6_transport_teacher import Teacher as BaseTeacher


class Teacher(BaseTeacher):
    def __init__(self,env,oracle,spec,profile):
        super().__init__(env,oracle,spec,profile)
        assert self.front
        self.target[0]+=.025
        self.target[2]-=.02
        self.carry_z=max(float(self.target[2]),float(self.initial_reference[2]+.085))


def make_spec(condition,split,index):
    spec=base_spec(condition,split,index)
    assert spec["region_kind"] in {"infant_front","microwave_front"}
    spec["item_id"]="front__"+spec["item_id"];spec["pair_group"]="front__"+spec["pair_group"]
    rng=np.random.default_rng(spec["seed"]+771)
    offset=np.array(spec["goal_center"][:2])-spec["destination_xy"]
    spec["destination_xy"]=[float(rng.uniform(-.12,-.09)),float((-1 if index%2==0 else 1)*rng.uniform(.24,.27))]
    spec["goal_center"][:2]=(np.array(spec["destination_xy"])+offset).tolist()
    spec["nominal_transport_distance_xy_m"]=float(np.linalg.norm(np.array(spec["actor_xy"])-spec["goal_center"][:2]))
    spec["teacher_family"]="reachable_front_layout"
    return spec


def process(request):
    fast_configure()
    import collect_v6_lora_data as collector
    import v6_transport_teacher
    v6_transport_teacher.Teacher=Teacher
    collector.make_spec=make_spec;collector.__file__=__file__
    return collector.process(request)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("--split",default="calibration")
    p.add_argument("--tag",default="front_pilot");p.add_argument("--count",type=int,default=1)
    p.add_argument("--start",type=int,default=1000);p.add_argument("--maximum",type=int,default=4)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--conditions",default="4,5,6,7")
    args=p.parse_args();requests=[(int(c),args.split,args.count,args.start,args.maximum,args.tag) for c in args.conditions.split(",")]
    with multiprocessing.get_context("spawn").Pool(args.workers,maxtasksperchild=1) as pool:
        results=list(pool.imap_unordered(process,requests))
    rows=sorted([r for result in results for r in result["accepted"]],key=lambda r:r["item_id"])
    output=DATA/args.tag;output.mkdir(parents=True,exist_ok=True)
    (output/f"{args.split}_manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    write_json(output/f"{args.split}_summary.json",dict(episodes=len(rows),complete=all(r["complete"] for r in results),conditions=results))
    write_json(output/"provenance.json",dict(script_sha256=sha256(Path(__file__)),source_v6_changed=False,
               note="new training layouts only; both labels share geometry seeds; goal-point bias stays inside the original 5cm region"))
    print(json.dumps(dict(episodes=len(rows),complete=all(r["complete"] for r in results))),flush=True)


if __name__=="__main__":main()
