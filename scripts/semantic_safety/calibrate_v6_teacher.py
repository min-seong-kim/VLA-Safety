"""Select grasp profiles using newly randomized calibration layouts only."""

import argparse
import json
import multiprocessing
from pathlib import Path
import traceback

from rollout_common import bootstrap, write_json
from semantic_pairs_v6 import sha256
from v6_lora_data import DATA, conditions, freeze_design, make_spec
from v6_transport_teacher import candidate_profiles, rollout


def search(request):
    bootstrap()
    asset,condition=request
    folder=DATA/"profile_search"/asset;folder.mkdir(parents=True,exist_ok=True)
    signature={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name("v6_lora_data.py"),
                                        Path(__file__).with_name("v6_transport_teacher.py")]}
    final=folder/"selected.json"
    if final.exists():
        selected=json.loads(final.read_text());assert selected["signature"]==signature
        return selected
    results=[]
    for number,profile in enumerate(candidate_profiles(asset)):
        path=folder/f"candidate_{number:03d}.json"
        if path.exists():
            result=json.loads(path.read_text());assert result["signature"]==signature
        else:
            trials=[]
            for index in [0,1]:
                spec=make_spec(condition,"calibration",index)
                try:trials.append(rollout(spec,profile))
                except Exception:trials.append(dict(passed=False,error=traceback.format_exc()))
                if not trials[-1]["passed"]:break
            result=dict(asset=asset,condition=condition,profile=profile,candidate=number,trials=trials,
                        successes=sum(t["passed"] for t in trials),signature=signature)
            write_json(path,result)
        results.append(result)
        print(json.dumps(dict(asset=asset,candidate=number,successes=result["successes"])),flush=True)
        if result["successes"]==2:break
    best=max(results,key=lambda r:(r["successes"],-r["candidate"]))
    selected=dict(asset=asset,condition=condition,profile=best["profile"],calibration_successes=best["successes"],
                  candidates_tested=len(results),signature=signature,validation_or_test_used=False)
    write_json(final,selected)
    return selected


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--assets")
    args=parser.parse_args();freeze_design()
    lookup={}
    for index,item in enumerate(conditions()):lookup.setdefault(item["actor_asset"],index)
    if args.assets:lookup={a:lookup[a] for a in args.assets.split(",")}
    with multiprocessing.get_context("spawn").Pool(args.workers,maxtasksperchild=5) as pool:
        results=list(pool.imap_unordered(search,list(lookup.items())))
    profiles={r["asset"]:r["profile"] for r in results if r["calibration_successes"]>0}
    write_json(DATA/"selected_profiles.json",profiles)
    write_json(DATA/"calibration_coverage.json",dict(assets=len(results),included=len(profiles),
               results=results,excluded=[r["asset"] for r in results if not r["calibration_successes"]]))
    print(json.dumps(dict(included=len(profiles),assets=len(results),excluded=[r["asset"] for r in results if not r["calibration_successes"]])),flush=True)


if __name__=="__main__":
    main()
