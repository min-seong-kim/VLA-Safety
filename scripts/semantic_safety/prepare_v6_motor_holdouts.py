"""Freeze paired validation/test scenes without filtering on teacher or policy success."""

import argparse
import json
import multiprocessing

import numpy as np

from rollout_common import ROOT, bootstrap, observation_payload, write_json
from semantic_pairs_v6 import sha256
from v6_lora_data import DATA, make_environment, make_spec, validate_initial


def pair(request):
    bootstrap()
    split,family,index=request
    folder=DATA/"holdouts"/split/f"f{family:02d}_{index:02d}"
    folder.mkdir(parents=True,exist_ok=True)
    final=folder/"pair.json"
    if final.exists():return json.loads(final.read_text())["items"]
    rejections=[]
    for attempt in range(32):
        actual_index=index+100*attempt;items=[];valid=True
        for condition in [family*2,family*2+1]:
            spec=make_spec(condition,split,actual_index)
            spec["instruction"]=spec["source_instruction"]
            wrapper=None
            try:
                wrapper,obs,oracle=make_environment(spec,384);validate_initial(wrapper.env,oracle)
                payload=observation_payload(obs)
                target=folder/spec["item_id"];target.mkdir(exist_ok=True)
                np.savez_compressed(target/"initial.npz",**payload,sim_state=wrapper.env.sim.get_state().flatten())
                spec["observation"]=str((target/"initial.npz").relative_to(ROOT))
                spec["observation_sha256"]=sha256(target/"initial.npz")
                spec["motor_split"]=split;items.append(spec)
            except AssertionError as error:
                valid=False;rejections.append(dict(index=actual_index,condition=condition,reason=str(error)))
            finally:
                if wrapper:wrapper.close()
        if valid:
            write_json(final,dict(items=items,initial_contact_rejections=rejections,
                                 teacher_or_policy_success_used_to_select=False))
            return items
    raise RuntimeError((split,family,index,rejections))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("--split",choices=["validation","test"],required=True)
    p.add_argument("--layouts",type=int,required=True);p.add_argument("--workers",type=int,default=6)
    args=p.parse_args();requests=[(args.split,f,i) for f in range(33) for i in range(args.layouts)]
    results=[]
    with multiprocessing.get_context("spawn").Pool(args.workers,maxtasksperchild=8) as pool:
        for result in pool.imap_unordered(pair,requests):
            results.extend(result);print(json.dumps(dict(split=args.split,complete=len(results),total=len(requests)*2)),flush=True)
    results.sort(key=lambda r:r["item_id"])
    output=DATA/"holdouts"/(args.split+".jsonl")
    output.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in results))
    write_json(output.with_suffix(".manifest.json"),dict(sha256=sha256(output),scenes=len(results),
               conditions=66,filtered_on_success=False,checkpoint_selection=args.split=="validation"))


if __name__=="__main__":main()
