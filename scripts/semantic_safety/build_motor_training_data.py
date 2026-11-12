"""Collect and independently replay demonstrations with fixed seeds and coverage."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path

from motor_skill_env import ASSETS, DATA, make_spec, write_protocol
from rollout_common import ROOT, bootstrap, write_json


def collect_one(request):
    from collect_motor_demonstrations import record_teacher
    from validate_motor_demonstrations import validate

    bootstrap()
    spec, profile, folder = request
    path = folder / "summary.json"
    if path.exists():
        summary = json.loads(path.read_text())
    else:
        summary = record_teacher(spec, profile, folder, save_video=spec["index"] < 1)
    replay_path = folder / "replay_validation.json"
    if summary["success"]:
        result = json.loads(replay_path.read_text()) if replay_path.exists() else validate(folder)
        summary["action_replay_validation"] = result
        write_json(path, summary)
    else:
        result = {"accepted": False, "reason": "teacher failed success predicate"}
    return {"episode_id": spec["episode_id"], "asset": spec["asset"], "split": spec["split"],
            "path": str((folder / "trajectory.h5").resolve().relative_to(ROOT)),
            "steps": summary["steps"], "teacher_success": summary["success"],
            "accepted": result["accepted"], "replay": result}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=DATA / "demonstrations")
    args = parser.parse_args()
    bootstrap()
    write_protocol()
    profiles = json.loads(args.profiles.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.jsonl"
    rows = []
    requests = []
    for asset, profile in sorted(profiles.items()):
        for index in range(args.count):
            spec = make_spec(asset, "train", index)
            requests.append((spec, profile, args.out / spec["episode_id"]))
    frozen = {"assets": sorted(profiles), "attempts_per_asset": args.count, "profiles": profiles,
              "collection_attempts_planned": len(requests), "split": "train", "validation_indices": [0, 1],
              "test_indices": list(range(6)), "eval_horizon": 360, "eval_execute_chunk_steps": 10,
              "validation_checkpoint_candidates": [500, 1500, 3000],
              "checkpoint_selection": "validation task success, then validation grasp/lift, then earlier checkpoint",
              "asset_xml_sha256": {a: hashlib.sha256((ASSETS / f"{a}.xml").read_bytes()).hexdigest() for a in profiles},
              "held_out_test_used_for_training_or_selection": False,
              "teacher_failed_attempts": "preserved, excluded only from imitation training; no replacement seeds"}
    freeze_path = args.out / "frozen_design.json"
    if freeze_path.exists():
        assert json.loads(freeze_path.read_text()) == frozen
    write_json(freeze_path, frozen)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(collect_one, request): request for request in requests}
        with manifest.open("w", buffering=1) as log:
            for future in as_completed(futures):
                spec, profile, folder = futures[future]
                try:
                    row = future.result()
                except Exception as error:
                    row = {"episode_id": spec["episode_id"], "asset": spec["asset"], "split": "train",
                           "accepted": False, "error": repr(error)}
                    write_json(folder / "collection_error.json", row)
                log.write(json.dumps(row) + "\n")
                rows.append(row)
                print(json.dumps({"completed": len(rows), "planned": len(requests), "episode_id": spec["episode_id"],
                                  "accepted": row["accepted"]}), flush=True)
    rows.sort(key=lambda r: r["episode_id"])
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    coverage = {asset: {"accepted": sum(r["accepted"] for r in rows if r["asset"] == asset),
                         "attempted": sum(r["asset"] == asset for r in rows)} for asset in profiles}
    write_json(args.out / "coverage.json", {"assets": coverage, "accepted": sum(r["accepted"] for r in rows),
                                           "attempts": len(rows), "complete": True})


if __name__ == "__main__":
    main()
