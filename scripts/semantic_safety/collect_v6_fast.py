"""Skip camera rendering only in unrecorded teacher trials and physics-only replays."""

import argparse
import json
import multiprocessing
from pathlib import Path

import numpy as np

from rollout_common import bootstrap, write_json
from semantic_pairs_v6 import sha256
from v6_lora_data import DATA


def configure(pose=False):
    bootstrap()
    import v6_transport_teacher as teacher
    if pose:
        import v6_pose_teacher
        v6_pose_teacher.configure()
    physical_factory=teacher.make_environment

    def environment(spec,resolution=224):
        if resolution!=64:return physical_factory(spec,resolution)
        import libero.libero.envs as envs
        original=envs.OffScreenRenderEnv
        def factory(**kwargs):
            kwargs["use_camera_obs"]=False
            return original(**kwargs)
        envs.OffScreenRenderEnv=factory
        try:wrapper,obs,oracle=physical_factory(spec,resolution)
        finally:envs.OffScreenRenderEnv=original
        # These placeholders are used only by the unrecorded trial's unused initial
        # payload. Recorded demonstrations always take the unmodified 224px branch.
        obs["agentview_image"]=np.zeros((64,64,3),dtype=np.uint8)
        obs["robot0_eye_in_hand_image"]=np.zeros((64,64,3),dtype=np.uint8)
        return wrapper,obs,oracle
    teacher.make_environment=environment


def process(request):
    pose,values=request;configure(pose)
    import collect_v6_lora_data
    return collect_v6_lora_data.process(values)


def replay_one(folder):
    configure()
    from v6_transport_teacher import replay
    return dict(folder=str(folder),**replay(Path(folder)))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("--split",default="train")
    p.add_argument("--count",type=int,default=48);p.add_argument("--start",type=int,default=0)
    p.add_argument("--maximum",type=int,default=96);p.add_argument("--workers",type=int,default=12)
    p.add_argument("--conditions");p.add_argument("--tag",default="episodes");p.add_argument("--pose",action="store_true")
    p.add_argument("--verify-only",action="store_true")
    args=p.parse_args()
    if args.verify_only:
        folders=sorted((DATA/"pilot16/calibration").glob("c*/*/trajectory.h5"))
        folders += sorted((DATA/"episodes/train").glob("c*/*/trajectory.h5"))[:2]
        folders=[str(path.parent) for path in folders if (path.parent/"receipt.json").exists()][:16]
        assert len(folders)==16
        with multiprocessing.get_context("spawn").Pool(4,maxtasksperchild=1) as pool:
            rows=list(pool.imap_unordered(replay_one,folders))
        write_json(DATA/"render_skip_replay_audit.json",dict(passed=all(r["accepted"] for r in rows),episodes=len(rows),results=rows))
        assert all(r["accepted"] for r in rows);print("VERIFIED",len(rows),flush=True);return
    indices=list(range(66)) if args.conditions is None else [int(x) for x in args.conditions.split(",")]
    requests=[(args.pose,(c,args.split,args.count,args.start,args.maximum,args.tag)) for c in indices]
    with multiprocessing.get_context("spawn").Pool(args.workers,maxtasksperchild=1) as pool:
        results=list(pool.imap_unordered(process,requests))
    rows=sorted([r for result in results for r in result["accepted"]],key=lambda r:r["item_id"])
    output=DATA/args.tag;output.mkdir(parents=True,exist_ok=True)
    (output/f"{args.split}_manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    write_json(output/f"{args.split}_summary.json",dict(episodes=len(rows),steps=sum(r["steps"] for r in rows),
               complete=all(r["complete"] for r in results),conditions=results))
    write_json(output/"render_optimization.json",dict(wrapper_sha256=sha256(Path(__file__)),
               note="Camera sensors skipped only for unrecorded 64px teacher trials and physics replays; all training RGB is genuinely rendered at 224px. Resumed cache can include earlier standard-rendering trajectories.",
               pose_teacher=args.pose))
    print(json.dumps(dict(episodes=len(rows),complete=all(r["complete"] for r in results))),flush=True)


if __name__=="__main__":main()
