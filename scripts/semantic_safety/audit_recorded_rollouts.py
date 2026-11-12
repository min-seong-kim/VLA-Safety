"""Verify complete episodes and create timestep-level probe indices without label leakage."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np

from rollout_common import ROOT, write_json


def audit_episode(folder, check_finite=True):
    summary = json.loads((folder / "summary.json").read_text())
    meta = json.loads((folder / "episode.json").read_text())
    steps = summary["steps"]
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    assert len(events) == steps
    signatures = {name: {"bytes": (folder / name).stat().st_size, "mtime_ns": (folder / name).stat().st_mtime_ns}
                  for name in ["features.h5", "trajectory.h5", "episode.json", "events.jsonl"]}
    previous_path = folder / "audit.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        if (previous.get("status") == "passed" and previous.get("file_signatures") == signatures and
                (previous.get("finite_features_checked") or not check_finite)):
            with h5py.File(folder / "trajectory.h5") as trajectory:
                times = trajectory["observations/sim_time"][:]
            if summary.get("feature_timestep_alignment") != "passed post-run audit":
                summary["feature_timestep_alignment"] = "passed post-run audit"
                write_json(folder / "summary.json", summary)
            return summary, meta, events, times
    feature_shapes = {}
    with h5py.File(folder / "features.h5") as features, h5py.File(folder / "trajectory.h5") as trajectory:
        assert int(features.attrs["committed_steps"]) == steps
        assert int(trajectory.attrs["committed_steps"]) == steps
        assert trajectory["observations/main_rgb_raw"].shape[0] == steps + 1
        assert trajectory["observations/wrist_rgb_raw"].shape[0] == steps + 1
        assert np.array_equal(features["output/submitted_action"][:], trajectory["actions/submitted"][:])
        assert np.array_equal(features["output/action_chunk"][:], trajectory["actions/predicted_chunk"][:])
        assert np.array_equal(features["input/raw_state"][:], trajectory["observations/state"][:-1])
        for step in range(steps):
            recorded = json.loads(features["step_metadata"][step])
            digest = hashlib.sha256(trajectory["observations/main_rgb_raw"][step].tobytes()).hexdigest()
            assert recorded["main_observation_sha256"] == digest, (folder, step, "image/feature mismatch")
            assert recorded["step"] == events[step]["step"] == step
            assert recorded["instruction"] == meta["instruction"]
            assert recorded["model_input_contains_safety_label"] is False
        for name, dataset in features["pooled"].items():
            assert dataset.shape[0] == steps
            feature_shapes[name] = list(dataset.shape)
            if check_finite:
                for start in range(0, steps, 16):
                    assert np.isfinite(dataset[start:start + 16]).all(), (folder, name, "nonfinite")
        for key in ["observations/qpos", "observations/qvel", "observations/state", "actions/submitted"]:
            assert np.isfinite(trajectory[key][:]).all(), (folder, key, "nonfinite")
        times = trajectory["observations/sim_time"][:]
        assert np.allclose(np.diff(times), .05, atol=1e-6)
        assert features["initial_full_tokens/vlm_hidden"].shape[1] == features["pooled/vlm_hidden"].shape[2]
        if meta.get("suite") == "semantic_safety_v3":
            assert meta["initial_metrics"]["risk_proxy"] is False
            assert all(v >= 4 for v in meta["initial_visible_pixels_agentview"].values())
    report = {"status": "passed", "steps": steps, "all_step_image_hashes_match": True,
              "all_step_states_actions_match": True, "finite_features_checked": check_finite,
              "feature_shapes": feature_shapes, "audit_time_unix": time.time()}
    report["file_signatures"] = signatures
    write_json(folder / "audit.json", report)
    summary["feature_timestep_alignment"] = "passed post-run audit"
    write_json(folder / "summary.json", summary)
    return summary, meta, events, times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--skip-finite", action="store_true")
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    completed = []
    failures = []
    pairs = defaultdict(list)
    index_path = args.run_root / "probe_index.jsonl.gz"
    temporary = index_path.with_suffix(".tmp.gz")
    with gzip.open(temporary, "wt") as index:
        for path in sorted(args.run_root.glob("*/episodes/*/summary.json")):
            folder = path.parent
            try:
                summary, meta, events, times = audit_episode(folder, not args.skip_finite)
                completed.append(summary)
                if meta.get("canonical_scene_id"):
                    pairs[meta["canonical_scene_id"]].append({"model": meta["model"],
                         "unsafe": meta["task_is_unsafe"], "image_hash": meta["initial_observation_sha256"],
                         "state_hash": meta["initial_state_sha256"], "instruction": meta["instruction"]})
                previous_risk = False if meta.get("suite") == "semantic_safety_v3" else None
                for step, event in enumerate(events):
                    row = {"model": meta["model"], "episode_id": summary["episode_id"], "timestep": step,
                           "sim_time_before_action": float(times[step]), "suite": meta["suite"],
                           "task_is_unsafe": meta.get("task_is_unsafe"),
                           "family_id": meta.get("family_id"), "mechanism_group": meta.get("mechanism_group"),
                           "split_group": meta.get("family_id", meta.get("source_bddl")),
                           "canonical_scene_id": meta.get("canonical_scene_id"),
                           "risk_proxy_before_action": previous_risk, "risk_proxy_after_action": event["risk_proxy"],
                           "task_success_after_action": event["task_success"], "harm_observed": None,
                           "feature_h5": str((folder / "features.h5").relative_to(ROOT)),
                           "trajectory_h5": str((folder / "trajectory.h5").relative_to(ROOT)),
                           "feature_row": step, "observation_row": step}
                    index.write(json.dumps(row) + "\n")
                    previous_risk = event["risk_proxy"]
                print(json.dumps({"audited": len(completed), "episode": str(folder.relative_to(ROOT)), "steps": summary["steps"]}), flush=True)
            except Exception as error:
                failures.append({"episode": str(folder.relative_to(ROOT)), "error": repr(error)})
                print("AUDIT_FAILED", json.dumps(failures[-1]), flush=True)
    temporary.replace(index_path)
    pairing = {}
    for key, rows in pairs.items():
        consistent = len({r["image_hash"] for r in rows}) == 1 and len({r["state_hash"] for r in rows}) == 1
        pairing[key] = {"episodes": len(rows), "same_initial_image_and_state": consistent,
                        "model_label_combinations": sorted([r["model"] + ":" + str(int(r["unsafe"])) for r in rows])}
        if not consistent:
            failures.append({"scene": key, "error": "initial observations or states differ across paired episodes/models"})
    by_model = defaultdict(lambda: {"episodes": 0, "steps": 0, "task_success_proxy": 0, "risk_proxy_episodes": 0})
    for summary in completed:
        group = by_model[summary["model"]]
        group["episodes"] += 1
        group["steps"] += summary["steps"]
        group["task_success_proxy"] += summary["task_success"] is True
        group["risk_proxy_episodes"] += summary["risk_proxy_ever"] is True
    write_json(args.run_root / "audit_summary.json", {"status": "passed" if not failures else "failed",
               "audited_complete_episodes": len(completed), "timestep_records": sum(r["steps"] for r in completed),
               "by_model": dict(by_model), "pairing": pairing, "failures": failures,
               "native_tasks_are_not_automatically_labeled_safe": True,
               "index_path": str(index_path.relative_to(ROOT))})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
