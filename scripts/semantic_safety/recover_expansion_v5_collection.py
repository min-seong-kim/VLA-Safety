"""Fill missing training demonstrations using broader grasp search and unchanged success checks."""

import argparse
import fcntl
import json
import multiprocessing as mp
from pathlib import Path
import time

from collect_expansion_v5_demonstrations import signature
from expansion_v5_motor import DATA, ROOT, RUN, conditions, make_spec, sha, write_jsonl
from expansion_v5_teacher import replay, rollout
from rollout_common import write_json
from search_expansion_v5_recovery import proposals


def read_receipt(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        # Another worker may have just archived a rejected attempt. Accepted
        # receipts are never moved, and new receipts are published atomically.
        return None


def accepted_rows(base, condition):
    rows = []
    for path in sorted((base / "train" / f"c{condition:03d}").glob("*/receipt.json")):
        row = read_receipt(path)
        if row is None:
            continue
        assert row["signature"] == signature()
        if row["accepted"]:
            rows.append(row)
    return rows


def warm_profiles(base, spec):
    definitions = conditions()
    rows = []
    for path in sorted((RUN / "diagnostics/perturbed_grasp_search").rglob("c*.json")):
        probe = json.loads(path.read_text())
        source = definitions[probe["condition"]]
        if (probe["signature"] == signature() and probe["result"]["passed"]
                and source["actor_asset"] == spec["actor_asset"] and source["skill"] == spec["skill"]):
            if probe["profile"] not in rows:
                rows.append(probe["profile"])
    for source in definitions:
        if source["actor_asset"] != spec["actor_asset"] or source["skill"] != spec["skill"]:
            continue
        folder = base / "train" / f"c{source['condition_index']:03d}"
        for path in sorted(folder.glob("*/receipt.json")):
            receipt = read_receipt(path)
            if receipt is None:
                continue
            profile = receipt.get("recovery_profile")
            if receipt["accepted"] and profile and profile not in rows:
                rows.append(profile)
    return rows


def collect_one(request):
    tag, condition, index, profile_limit, wide = request
    base = DATA / tag
    source = conditions()[condition]
    spec = make_spec(condition, "train", index)
    folder = base / "train" / f"c{condition:03d}" / spec["item_id"]
    receipt = folder / "receipt.json"
    calibration = json.loads((base / "profiles" / f"c{condition:03d}.json").read_text())
    assert calibration["success"] and calibration["signature"] == signature()
    profiles = warm_profiles(base, spec)
    limit=profile_limit if wide or not profiles else min(profile_limit,len(profiles)+4)
    profiles.extend(p for p in proposals(spec, calibration["profile"]) if p not in profiles)
    if receipt.exists():
        row = json.loads(receipt.read_text())
        assert row["signature"] == signature()
        if row["accepted"] or row.get("recovery_profile_limit",0)>=limit:
            return row
        archive=folder/"recovery_archive"/str(time.time_ns());archive.mkdir(parents=True)
        receipt.rename(archive/"receipt.json")
        if (folder/"attempts.json").exists():
            (folder/"attempts.json").rename(archive/"attempts.json")
    previous = folder / "recovery_trials.json"
    trials = json.loads(previous.read_text()) if previous.exists() else []
    chosen = None
    for profile in profiles[:limit]:
        prior = next((r for r in trials if r["profile"] == profile), None)
        if prior is not None:
            assert prior["signature"] == signature()
            result = prior["result"]
        else:
            result = rollout(spec, profile)
            trials.append(dict(profile=profile, result=result, signature=signature()))
            write_json(previous, trials)
        if result["passed"]:
            chosen = profile
            break
    row = dict(item_id=spec["item_id"], condition_index=condition, source_scene_id=source["source_scene_id"],
               pair_group=spec["pair_group"], split="train", accepted=False, signature=signature(),
               skill=source["skill"], data_type="expansion_v5", instruction=source["instruction"],
               recovery_profile=chosen, recovery_script_sha256=sha(__file__),
               recovery_profile_limit=limit,
               recovery_proposal_script_sha256=sha(Path(__file__).with_name("search_expansion_v5_recovery.py")),
               recovery_reason="Default grasp was not robust to training layout perturbations",
               unchanged_physics_controller_success_checks=True)
    if chosen is not None:
        rendered = rollout(spec, chosen, folder=folder, resolution=224)
        assert rendered["passed"], rendered
        check = replay(folder)
        assert check["accepted"], check
        row.update(accepted=True, steps=rendered["steps"], path=str((folder / "trajectory.h5").relative_to(ROOT)),
                   h5_sha256=check["trajectory_sha256"], replay=str((folder / "replay_validation.json").relative_to(ROOT)))
    write_json(folder / "attempts.json", [dict(passed=t["result"]["passed"], stage=t["result"]["stage"],
        final_metric=t["result"]["final_metric"]) for t in trials])
    write_json(receipt, row)
    print(json.dumps(dict(condition=condition, index=index, accepted=row["accepted"], profiles=len(trials))), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--tag", default="collection_r10")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--attempt-limit", type=int, default=240)
    parser.add_argument("--profile-limit", type=int, default=160)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--ids", help="Restrict collection; do not replace the global manifest or summary")
    args = parser.parse_args()
    base = DATA / args.tag
    lock=(base/"recovery.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX)
    for name in [Path(__file__).name,"search_expansion_v5_recovery.py"]:
        path=Path(__file__).with_name(name)
        snapshot=base/"recovery_source_snapshot"/sha(path)/name
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
    indices = list(map(int, args.ids.split(","))) if args.ids else list(range(len(conditions())))
    started = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for wide in [False,True]:
            while True:
                tasks = []
                for condition in indices:
                    remaining = args.count - len(accepted_rows(base, condition))
                    if remaining <= 0:
                        continue
                    folder = base / "train" / f"c{condition:03d}"
                    available=[]
                    for index in range(args.attempt_limit):
                        receipt=folder/make_spec(condition,"train",index)["item_id"]/"receipt.json"
                        row=json.loads(receipt.read_text()) if receipt.exists() else None
                        if row is None or (not row["accepted"] and ("recovery_script_sha256" not in row or
                                (wide and row.get("recovery_profile_limit",0)<args.profile_limit))):
                            available.append(index)
                    tasks.extend((args.tag,condition,index,args.profile_limit,wide)
                                 for index in available[:min(4,remaining)])
                if not tasks:
                    break
                for _ in pool.imap_unordered(collect_one,tasks):
                    pass
    coverage = {i:len(accepted_rows(base, i)) for i in indices}
    result = dict(complete=all(n >= args.count for n in coverage.values()), coverage=coverage,
                  seconds=time.time()-started, signature=signature(), unchanged_success_criteria=True)
    if args.ids:
        write_json(base / ("recovery_subset_" + args.ids.replace(",", "_") + ".json"), result)
    else:
        assert result["complete"], {i:n for i,n in coverage.items() if n < args.count}
        rows = []
        for condition in indices:
            accepted = accepted_rows(base, condition)[:args.count]
            rows.extend(accepted)
            receipts = [json.loads(p.read_text()) for p in sorted((base / "train" / f"c{condition:03d}").glob("*/receipt.json"))]
            write_json(base / "summaries" / f"c{condition:03d}.json", dict(condition=condition, accepted=accepted,
                attempts=receipts, target=args.count, complete=True))
        old = base / "collect_summary.json"
        if old.exists():
            write_json(base / "collection_summary_before_recovery.json", json.loads(old.read_text()))
        write_jsonl(base / "train_manifest.jsonl", rows)
        write_json(old, dict(mode="collect", conditions=len(indices), requested_indices=indices,
            episodes=len(rows), control_steps=sum(r["steps"] for r in rows), coverage=coverage,
            all_accepted_replayed=True, recovery_used=True, **{k:v for k,v in result.items() if k!="coverage"}))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
