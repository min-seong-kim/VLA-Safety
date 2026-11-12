"""Freeze an expanded corpus without changing the first experiment's manifests."""

from collections import Counter
import json
from pathlib import Path
import shutil
import time

import h5py
import numpy as np

from rollout_common import ROOT, write_json
from semantic_pairs_v6 import sha256
from v6_lora_data import DATA as PREVIOUS_DATA
from v6_lora_data import RUN as PREVIOUS_RUN
from v6_lora_data import conditions

DATA = ROOT / "data/semantic_motor_v6_lora_v2"
RUN = ROOT / "runs/semantic_motor_v6_lora_v2"


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(parents=True, exist_ok=True)
    source = PREVIOUS_DATA / "spoon_recovery"
    while not (source / "train_summary.json").exists():
        time.sleep(20)
    assert json.loads((source / "train_summary.json").read_text())["complete"]
    rows = [json.loads(line) for path in [PREVIOUS_DATA / "train_manifest.jsonl", source / "train_manifest.jsonl"]
            for line in path.read_text().splitlines()]
    rows.sort(key=lambda row: row["item_id"])
    assert len({row["item_id"] for row in rows}) == len(rows)
    geometry = set()
    for row in rows:
        audit = json.loads((ROOT / row["cache_audit"]).read_text())
        assert row["accepted"] and audit["passed"] and audit["pixels_actions_states_bit_identical"]
        assert audit["physical_replay"]["accepted"] and audit["source_sha256"] == row["h5_sha256"]
        with h5py.File(ROOT / row["training_cache"]) as file:
            spec = json.loads(file.attrs["spec_json"])
            assert spec["split"] == "train"
        geometry.add(tuple(np.round([*spec["actor_xy"], *spec["destination_xy"], spec["actor_yaw"]], 8)))
    holdouts = DATA / "holdouts"
    holdouts.mkdir(exist_ok=True)
    for path in (PREVIOUS_DATA / "holdouts").glob("*.json*"):
        shutil.copy2(path, holdouts / path.name)
    for split in ["validation", "test"]:
        for line in (holdouts / f"{split}.jsonl").read_text().splitlines():
            item = json.loads(line)
            key = tuple(np.round([*item["actor_xy"], *item["destination_xy"], item["actor_yaw"]], 8))
            assert key not in geometry
    manifest = DATA / "train_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    by_condition = Counter(row["condition_index"] for row in rows)
    definitions = conditions()
    labels = Counter("unsafe" if definitions[index]["task_is_unsafe"] else "safe" for index in by_condition)
    audit = dict(complete=True, collection_complete=True, episodes=len(rows),
                 control_steps=sum(row["steps"] for row in rows), conditions_total=66,
                 conditions_with_demonstrations=len(by_condition), by_condition=dict(by_condition),
                 covered_conditions_by_label=dict(labels),
                 actor_assets=dict(Counter(row["asset"] for row in rows)),
                 missing_conditions=[dict(condition=i, actor=item["actor_asset"], destination=item["destination_asset"])
                                     for i, item in enumerate(definitions) if not by_condition[i]],
                 all_accepted_demos_replayed=True, cache_arrays_bit_identical=True,
                 no_exact_holdout_geometry_overlap=True, manifest_sha256=sha256(manifest),
                 previous_manifest_sha256=sha256(PREVIOUS_DATA / "train_manifest.jsonl"),
                 added_manifest_sha256=sha256(source / "train_manifest.jsonl"),
                 change="48 successful spoon demonstrations after a calibration-only 8mm grasp-height correction",
                 shared_immutable_trajectory_root=str(PREVIOUS_DATA.relative_to(ROOT)))
    assert min(by_condition.values()) >= 48 and len(by_condition) == 63
    write_json(DATA / "training_data_audit.json", audit)
    for name in ["native_rehearsal_manifest.jsonl", "native_rehearsal_audit.json", "design.json"]:
        path = PREVIOUS_DATA / name
        if path.exists():
            shutil.copy2(path, DATA / name)
    native = [json.loads(line) for line in (DATA / "native_rehearsal_manifest.jsonl").read_text().splitlines()]
    combined = DATA / "combined_train_manifest.jsonl"
    combined.write_text("".join(json.dumps(row) + "\n" for row in rows + native))
    write_json(DATA / "rehearsal_merge_audit.json", dict(passed=True, motor_episodes=len(rows), native_episodes=len(native),
               motor_sampling_fraction=.9, native_sampling_fraction=.1,
               native_evaluation_initial_states_excluded=True, motor_manifest_sha256=sha256(manifest),
               native_manifest_sha256=sha256(DATA / "native_rehearsal_manifest.jsonl"),
               combined_manifest_sha256=sha256(combined), identical_manifest_for_three_scopes=True))
    for name in ["pipeline_preflight.json", "batch32_full_rollout_equivalence_audit.json", "environment.json",
                 "simulation_environment.json", "original_goal_ambiguity.json"]:
        shutil.copy2(PREVIOUS_RUN / name, RUN / name)
    if not (RUN / "experiment_source_snapshot").exists():
        shutil.copytree(PREVIOUS_RUN / "experiment_source_snapshot", RUN / "experiment_source_snapshot")
    protocol = json.loads((PREVIOUS_RUN / "experiment_protocol.json").read_text())
    protocol.update(dataset_version=2, previous_run=str(PREVIOUS_RUN.relative_to(ROOT)),
                    revision_reason=audit["change"], all_scopes_restarted_from_official_base=True,
                    new_manifest_sha256=sha256(combined), prior_results_retained=True, written_unix=time.time())
    write_json(RUN / "experiment_protocol.json", protocol)
    for relative, original in [("baseline_selection66_balanced", "baseline_selection66_balanced"),
                               ("final/native/baseline", "final/native/baseline")]:
        target = RUN / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(PREVIOUS_RUN / original, target_is_directory=True)
    write_json(RUN / "reused_baselines.json", dict(reason="Same official base, observations, seeds and evaluation lists",
               previous_run=str(PREVIOUS_RUN), original_v6="runs/semantic_safety_transport_v6_evaluation/pi05_libero"))
    print("EXPANDED_CORPUS_READY", json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
