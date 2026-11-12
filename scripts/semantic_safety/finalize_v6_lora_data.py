"""Build pixel-identical frame-chunked training caches and freeze an audited manifest."""

import argparse
from collections import Counter
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import h5py
import numpy as np

from rollout_common import ROOT, write_json
from semantic_pairs_v6 import sha256
from v6_lora_data import DATA, conditions


def cache_one(receipt):
    row=json.loads(Path(receipt).read_text());assert row["accepted"]
    source=ROOT/row["path"];folder=DATA/"training_cache"/row["item_id"]
    folder.mkdir(parents=True,exist_ok=True);final=folder/"audit.json"
    if final.exists():
        cached=json.loads(final.read_text());assert cached["source_sha256"]==row["h5_sha256"]
        return {**row,"training_cache":str((folder/"inputs.h5").relative_to(ROOT)),"cache_audit":str(final.relative_to(ROOT))}
    assert sha256(source)==row["h5_sha256"]
    replay=json.loads((source.parent/"replay_validation.json").read_text());assert replay["accepted"]
    temporary=folder/"inputs.tmp.h5";checks={}
    with h5py.File(source) as raw,h5py.File(temporary,"w") as output:
        spec=json.loads(raw.attrs["spec_json"])
        assert spec["split"]=="train" and raw.attrs["teacher_success"]
        assert len(raw["actions"])==row["steps"]
        output.attrs["spec_json"]=raw.attrs["spec_json"]
        output.attrs["source_h5_sha256"]=row["h5_sha256"]
        for key in ["observations/main_rgb_raw","observations/wrist_rgb_raw","observations/state","actions"]:
            value=raw[key][:];assert np.isfinite(value).all()
            if "rgb" in key:
                assert value.shape==(row["steps"]+1,224,224,3) and value.dtype==np.uint8
                assert np.std(value[0].astype(float))>1
                output.create_dataset(key,data=value,chunks=(1,224,224,3),compression="lzf",shuffle=True)
            else:output.create_dataset(key,data=value)
            checks[key]=hashlib.sha256(value.tobytes()).hexdigest()
    with h5py.File(temporary) as check:
        for key,digest in checks.items():assert hashlib.sha256(check[key][:].tobytes()).hexdigest()==digest
    temporary.replace(folder/"inputs.h5")
    write_json(final,dict(passed=True,source_sha256=row["h5_sha256"],array_sha256=checks,
                         pixels_actions_states_bit_identical=True,physical_replay=replay))
    return {**row,"training_cache":str((folder/"inputs.h5").relative_to(ROOT)),"cache_audit":str(final.relative_to(ROOT))}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("--follow",action="store_true");p.add_argument("--workers",type=int,default=4)
    args=p.parse_args();processed={};sources=[DATA/"episodes",DATA/"pose_episodes",DATA/"front_episodes"]
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        while True:
            pending=[]
            for source in sources:
                for receipt in (source/"train").glob("c*/*/receipt.json"):
                    if str(receipt) in processed:continue
                    row=json.loads(receipt.read_text())
                    if row["accepted"]:pending.append(str(receipt))
            for result in pool.imap_unordered(cache_one,pending):processed[result["path"]]=result
            # Keys track physical trajectory paths, not incomplete producer directories.
            if pending:
                for receipt in pending:
                    row=json.loads(Path(receipt).read_text());processed[str(receipt)]=processed[row["path"]]
            rows={r["item_id"]:r for r in processed.values()}
            print(json.dumps(dict(cached=len(rows),collection_done=all((s/"train_summary.json").exists() for s in sources))),flush=True)
            if all((s/"train_summary.json").exists() for s in sources) or not args.follow:break
            time.sleep(20)
    rows=sorted(rows.values(),key=lambda r:r["item_id"])
    assert rows
    by_condition=Counter(r["condition_index"] for r in rows);assets=Counter(r["asset"] for r in rows)
    geometry=set();variants=Counter();pose_count=0
    for row in rows:
        with h5py.File(ROOT/row["training_cache"]) as data:
            spec=json.loads(data.attrs["spec_json"])
        key=tuple(np.round([*spec["actor_xy"],*spec["destination_xy"],spec["actor_yaw"]],8))
        geometry.add(key);variants[spec["index"]%4]+=1;pose_count+=spec.get("initial_pose_augmentation") is not None
    for split in ["validation","test"]:
        holdouts=[json.loads(s) for s in (DATA/"holdouts"/(split+".jsonl")).read_text().splitlines()]
        assert not any(tuple(np.round([*s["actor_xy"],*s["destination_xy"],s["actor_yaw"]],8)) in geometry for s in holdouts)
    manifest=DATA/"train_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row)+"\n" for row in rows))
    coverage=dict(complete=True,episodes=len(rows),control_steps=sum(r["steps"] for r in rows),
                  conditions_with_demonstrations=len(by_condition),conditions_total=66,by_condition=dict(by_condition),
                  actor_assets=dict(assets),instruction_variants=dict(variants),pose_augmented_episodes=pose_count,
                  missing_conditions=[dict(condition=i,actor=r["actor_asset"],destination=r["destination_asset"])
                                      for i,r in enumerate(conditions()) if not by_condition[i]],
                  all_accepted_demos_replayed=True,cache_arrays_bit_identical=True,no_exact_holdout_geometry_overlap=True,
                  manifest_sha256=sha256(manifest),collection_complete=all((s/"train_summary.json").exists() for s in sources))
    coverage["collection_tags"]=[s.name for s in sources]
    write_json(DATA/"training_data_audit.json",coverage);print("TRAINING_DATA_READY",json.dumps(coverage),flush=True)


if __name__=="__main__":main()
