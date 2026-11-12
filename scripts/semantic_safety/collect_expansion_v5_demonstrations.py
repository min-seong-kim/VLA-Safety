"""Calibrate and collect replay-verified demonstrations without policy-success filtering of holdouts."""

import argparse
import json
import hashlib
import multiprocessing as mp
from pathlib import Path
import shutil
import time
import traceback

import h5py
import numpy as np

from expansion_v5_motor import DATA, RUN, ROOT, conditions, make_spec, sha, write_jsonl
from expansion_v5_teacher import candidates, replay, rollout
from rollout_common import write_json


def signature():
    return {**{name:sha(Path(__file__).with_name(name)) for name in [
        "expansion_v5_motor.py","expansion_v5_teacher.py","collect_expansion_v5_demonstrations.py"]},
        "physical_assets":sha(DATA/"assets/manifest.json")}


def process(request):
    condition,mode,count,attempt_limit,profile_limit,tag,warm_start=request
    source=conditions()[condition]
    base=DATA/tag
    selection=base/"profiles"/f"c{condition:03d}.json"
    if mode=="calibrate":
        if selection.exists():
            cached=json.loads(selection.read_text())
            if cached["signature"]==signature() and (cached["success"] or len(cached["trials"])>=profile_limit):
                return cached
        spec=make_spec(condition,"calibration",0)
        trials=[];chosen=None
        proposals=[];previous_folders=[]
        if warm_start:
            for earlier in warm_start.split(","):
                old=DATA/earlier/"profiles"/f"c{condition:03d}.json"
                if old.exists():
                    old=json.loads(old.read_text())
                    if old["success"]:
                        if old["profile"] not in proposals:proposals.append(old["profile"])
                        if abs(old["profile"]["tilt"])>.1:
                            retained={**old["profile"],"reorient_after_lift":False}
                            if retained not in proposals:proposals.append(retained)
                        previous_folders.append(DATA/earlier/"calibration"/f"c{condition:03d}")
        proposals.extend(p for p in candidates(spec)[:profile_limit] if p not in proposals)
        for profile in proposals:
            try:
                result=rollout(spec,profile)
            except Exception:
                trials.append(dict(error=traceback.format_exc(),profile=profile))
                break
            trials.append(result)
            if result["passed"]:
                chosen=profile
                break
        result=dict(condition=condition,source_scene_id=source["source_scene_id"],actor=source["actor_kind"],
            success=chosen is not None,profile=chosen,trials=trials,signature=signature())
        if chosen is not None:
            folder=base/"calibration"/f"c{condition:03d}"
            reused=None
            for earlier in previous_folders:
                if not (earlier/"trajectory.h5").exists() or sha(earlier/"model.xml")!=trials[-1]["model_xml_sha256"]:continue
                with h5py.File(earlier/"trajectory.h5") as old:
                    same=(json.loads(old.attrs["spec_json"])==spec and json.loads(old.attrs["profile_json"])==chosen and
                          hashlib.sha256(np.asarray(old["actions"][:],dtype=np.float32).tobytes()).hexdigest()==trials[-1]["action_sha256"])
                if same:
                    shutil.copytree(earlier,folder,dirs_exist_ok=True)
                    reused=str(earlier.relative_to(ROOT));break
            rendered=trials[-1] if reused else rollout(spec,chosen,folder=folder,resolution=224)
            check=replay(folder)
            assert rendered["passed"] and check["accepted"],(condition,rendered,check)
            result["replay_verified"]=True
            if reused:result["reuse_audit"]=dict(source=reused,current_teacher_actions_exact=True,
                model_xml_exact=True,spec_exact=True,independently_replayed_under_current_code=True)
        write_json(selection,result)
        print(json.dumps({k:result[k] for k in ["condition","source_scene_id","actor","success"]}),flush=True)
        return result
    assert selection.exists(),f"Calibrate condition {condition} first"
    calibration=json.loads(selection.read_text())
    assert calibration["signature"]==signature(),"Calibration code changed; recalibrate before collection"
    accepted=[];attempts=[]
    if not calibration["success"]:
        return dict(condition=condition,accepted=[],attempts=[],missing_reason="calibration_failed")
    for index in range(attempt_limit):
        spec=make_spec(condition,"train",index)
        folder=base/"train"/f"c{condition:03d}"/spec["item_id"]
        receipt=folder/"receipt.json"
        if receipt.exists():
            row=json.loads(receipt.read_text())
            assert row["signature"]==signature()
        else:
            profiles=[calibration["profile"]]
            profiles.extend(p for p in candidates(spec)[:6] if p not in profiles)
            chosen=None;trials=[]
            for profile in profiles:
                result=rollout(spec,profile)
                trials.append(dict(passed=result["passed"],stage=result["stage"],final_metric=result["final_metric"]))
                if result["passed"]:
                    chosen=profile
                    break
            row=dict(item_id=spec["item_id"],condition_index=condition,source_scene_id=source["source_scene_id"],
                pair_group=spec["pair_group"],split="train",accepted=False,signature=signature(),skill=source["skill"],
                data_type="expansion_v5",instruction=source["instruction"])
            if chosen is not None:
                rendered=rollout(spec,chosen,folder=folder,resolution=224)
                assert rendered["passed"]
                check=replay(folder)
                assert check["accepted"],check
                row.update(accepted=True,steps=rendered["steps"],path=str((folder/"trajectory.h5").relative_to(ROOT)),
                    h5_sha256=check["trajectory_sha256"],replay=str((folder/"replay_validation.json").relative_to(ROOT)))
            write_json(folder/"attempts.json",trials)
            write_json(receipt,row)
        attempts.append(row)
        if row["accepted"]:
            accepted.append(row)
        if len(accepted)>=count:
            break
        if len(attempts)>=10 and not accepted:
            break
    result=dict(condition=condition,accepted=accepted,attempts=attempts,target=count,complete=len(accepted)>=count)
    write_json(base/"summaries"/f"c{condition:03d}.json",result)
    print(json.dumps(dict(condition=condition,accepted=len(accepted),attempted=len(attempts),target=count)),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("mode",choices=["calibrate","collect"])
    p.add_argument("--ids",help="Comma-separated condition indices")
    p.add_argument("--workers",type=int,default=12)
    p.add_argument("--count",type=int,default=48)
    p.add_argument("--attempt-limit",type=int,default=160)
    p.add_argument("--profile-limit",type=int,default=80)
    p.add_argument("--tag",default="collection_r1")
    p.add_argument("--warm-start-tag",help="Propose earlier successful profiles; always rerun and independently replay them")
    args=p.parse_args()
    indices=list(map(int,args.ids.split(","))) if args.ids else list(range(len(conditions())))
    requests=[(i,args.mode,args.count,args.attempt_limit,args.profile_limit,args.tag,args.warm_start_tag) for i in indices]
    start=time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results=list(pool.imap_unordered(process,requests))
    results.sort(key=lambda r:r["condition"])
    folder=DATA/args.tag
    summary=dict(mode=args.mode,conditions=len(results),seconds=time.time()-start,
        signature=signature(),requested_indices=indices)
    if args.mode=="calibrate":
        summary.update(success=sum(r["success"] for r in results),
            failed=[r["condition"] for r in results if not r["success"]])
    else:
        rows=[row for r in results for row in r["accepted"]]
        write_jsonl(folder/"train_manifest.jsonl",rows)
        summary.update(episodes=len(rows),control_steps=sum(r["steps"] for r in rows),
            coverage={r["condition"]:len(r["accepted"]) for r in results},
            all_accepted_replayed=True,complete=all(r.get("complete",False) for r in results))
    write_json(folder/(args.mode+"_summary.json"),summary)
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
