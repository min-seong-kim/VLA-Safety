"""Collect physical demonstrations and accept only independently replayed successes."""

import argparse
import json
import multiprocessing
from pathlib import Path
import traceback

from rollout_common import ROOT, bootstrap, write_json
from semantic_pairs_v6 import sha256
from v6_lora_data import DATA, RUN, conditions, freeze_design, make_spec
from v6_transport_teacher import candidate_profiles, replay, rollout


def process(request):
    bootstrap()
    condition,split,count,start,maximum,tag=request
    actor=conditions()[condition]["actor_asset"]
    folder=DATA/tag/split/f"c{condition:02d}";folder.mkdir(parents=True,exist_ok=True)
    selected=DATA/"profile_search"/actor/"selected.json"
    profiles=[]
    if selected.exists():profiles.append(json.loads(selected.read_text())["profile"])
    # Candidate search is only for generating labels, never a model or a test-score filter.
    for profile in candidate_profiles(actor):
        if profile not in profiles:profiles.append(profile)
    signature={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name("v6_lora_data.py"),
                                        Path(__file__).with_name("v6_transport_teacher.py")]}
    accepted=[];attempts=[]
    for index in range(start,start+maximum):
        spec=make_spec(condition,split,index);target=folder/spec["item_id"]
        receipt=target/"receipt.json"
        if receipt.exists():
            row=json.loads(receipt.read_text());assert row["signature"]==signature
            attempts.append(row)
            if row["accepted"]:accepted.append(row)
            if len(accepted)>=count:break
            continue
        target.mkdir(parents=True,exist_ok=True);trials=[];chosen=None
        for profile in profiles[:12]:
            try:trial=rollout(spec,profile)
            except Exception:
                trial=dict(passed=False,error=traceback.format_exc());trials.append(trial);break
            trials.append(trial)
            if trial["passed"]:chosen=profile;break
        row=dict(item_id=spec["item_id"],condition_index=condition,asset=actor,split=split,
                 accepted=False,signature=signature,pair_group=spec["pair_group"],source_item_id=spec["source_item_id"])
        if chosen is not None:
            result=rollout(spec,chosen,folder=target,resolution=224)
            assert result["passed"], "Rendering resolution must not change physics"
            check=replay(target)
            assert check["accepted"], "Independent simulator replay did not reproduce successful demonstration"
            row.update(accepted=True,steps=result["steps"],path=str((target/"trajectory.h5").relative_to(ROOT)),
                       profile=chosen,h5_sha256=sha256(target/"trajectory.h5"))
            accepted.append(row)
        write_json(target/"attempts.json",trials);write_json(receipt,row);attempts.append(row)
        print(json.dumps(dict(condition=condition,split=split,index=index,accepted=row["accepted"],collected=len(accepted),target=count)),flush=True)
        if len(accepted)>=count:break
        if len(attempts)>=8 and not accepted:break
    result=dict(condition=condition,split=split,asset=actor,accepted=accepted,attempted=len(attempts),requested=count,
                complete=len(accepted)>=count)
    write_json(folder/"collection_summary.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--split",default="train");parser.add_argument("--count",type=int,default=48)
    parser.add_argument("--start",type=int,default=0);parser.add_argument("--maximum",type=int,default=96)
    parser.add_argument("--workers",type=int,default=6);parser.add_argument("--conditions")
    parser.add_argument("--tag",default="episodes")
    args=parser.parse_args();freeze_design()
    indices=list(range(66)) if not args.conditions else [int(x) for x in args.conditions.split(",")]
    requests=[(c,args.split,args.count,args.start,args.maximum,args.tag) for c in indices]
    with multiprocessing.get_context("spawn").Pool(args.workers,maxtasksperchild=2) as pool:
        results=list(pool.imap_unordered(process,requests))
    rows=sorted([row for result in results for row in result["accepted"]],key=lambda r:r["item_id"])
    manifest=DATA/args.tag/f"{args.split}_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row)+"\n" for row in rows))
    write_json(DATA/args.tag/f"{args.split}_summary.json",dict(episodes=len(rows),steps=sum(r["steps"] for r in rows),
               complete=all(r["complete"] for r in results),conditions=results))
    print(json.dumps(dict(manifest=str(manifest),episodes=len(rows),complete=all(r["complete"] for r in results))),flush=True)


if __name__=="__main__":main()
