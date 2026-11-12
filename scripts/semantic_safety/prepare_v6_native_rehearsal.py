"""Replay successful native trajectories and merge a disjoint rehearsal manifest."""

import argparse
from collections import Counter
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import time

import h5py
import numpy as np

from rollout_common import ROOT, bootstrap, native_environment, write_json
from semantic_pairs_v6 import sha256
from v6_lora_data import DATA, RUN


def prepare_one(summary):
    bootstrap()
    import libero.libero.envs as envs

    source = RUN / "native_rehearsal_collection/episodes" / summary["item_id"] / "trajectory.h5"
    folder = DATA / "native_rehearsal_cache" / summary["item_id"]
    folder.mkdir(parents=True, exist_ok=True)
    source_digest = sha256(source)
    receipt = folder / "receipt.json"
    if receipt.exists():
        row = json.loads(receipt.read_text())
        assert row["source_sha256"] == source_digest and row["accepted"]
        return row
    with h5py.File(source) as raw:
        item = json.loads(raw.attrs["item_json"])
        assert item["kind"] == "native" and item["initial_state_id"] in range(2, 7)
        assert summary["native_success"] and summary["steps"] >= 20
        states = raw["observations/sim_state"][:]
        actions = raw["actions/executed"][:]
        assert len(states) == len(actions) + 1 == summary["steps"] + 1
        original = envs.OffScreenRenderEnv

        def factory(**kwargs):
            kwargs["use_camera_obs"] = False
            return original(**kwargs)

        envs.OffScreenRenderEnv = factory
        try:
            wrapper, _, instruction, _ = native_environment(item["task_id"], item["initial_state_id"], 64)
        finally:
            envs.OffScreenRenderEnv = original
        env = wrapper.env
        try:
            initial_success = bool(env._check_success())
            maximum = float(np.max(np.abs(env.sim.get_state().flatten() - states[0])))
            for step, action in enumerate(actions):
                env.step(action)
                maximum = max(maximum, float(np.max(np.abs(env.sim.get_state().flatten() - states[step + 1]))))
            success = bool(env._check_success())
        finally:
            wrapper.close()
        assert maximum < 1e-6 and success and not initial_success, (item, maximum, success, initial_success)
        assert instruction == raw.attrs["instruction"]
        starts = raw["chunk_start_steps"][:]
        valid = np.flatnonzero(starts + 10 <= len(actions))
        assert len(valid) >= 2 and np.array_equal(starts, np.arange(0, len(actions), 10))
        spec = {**item, "instruction": instruction, "split": "train", "data_type": "native_rehearsal"}
        arrays = {
            "observations/main_rgb_raw": raw["policy_inputs/main_rgb_raw"][:][valid],
            "observations/wrist_rgb_raw": raw["policy_inputs/wrist_rgb_raw"][:][valid],
            "observations/state": raw["policy_inputs/state"][:][valid],
            "action_chunks": np.stack([actions[start:start + 10] for start in starts[valid]]),
            "source_control_steps": starts[valid],
        }
    temporary = folder / "inputs.tmp.h5"
    hashes = {}
    with h5py.File(temporary, "w") as output:
        output.attrs["spec_json"] = json.dumps(spec)
        output.attrs["source_sha256"] = source_digest
        for key, value in arrays.items():
            assert np.isfinite(value).all()
            if "rgb" in key:
                assert value.dtype == np.uint8 and value.shape[1:] == (384, 384, 3)
                assert np.std(value[0].astype(float)) > 1
                output.create_dataset(key, data=value, chunks=(1, 384, 384, 3), compression="lzf", shuffle=True)
            else:
                output.create_dataset(key, data=value)
            hashes[key] = hashlib.sha256(value.tobytes()).hexdigest()
    with h5py.File(temporary) as check:
        for key, digest in hashes.items():
            assert hashlib.sha256(check[key][:].tobytes()).hexdigest() == digest
    temporary.replace(folder / "inputs.h5")
    audit = dict(passed=True, independent_physics_replay=True, state_max_abs_error=maximum,
                 final_native_success=success, initial_native_success=initial_success,
                 source_sha256=source_digest, array_sha256=hashes,
                 full_executed_chunks_only=True, no_action_padding=True,
                 excluded_native_evaluation_initial_state_ids=[0, 1])
    write_json(folder / "audit.json", audit)
    row = dict(data_type="native_rehearsal", accepted=True, split="train", item_id=item["item_id"],
               condition_index=1000 + item["task_id"], asset="libero_native",
               path=str(source.relative_to(ROOT)), training_cache=str((folder / "inputs.h5").relative_to(ROOT)),
               cache_audit=str((folder / "audit.json").relative_to(ROOT)),
               chunks=len(valid), steps=len(actions), task_id=item["task_id"],
               initial_state_id=item["initial_state_id"], source_sha256=source_digest)
    write_json(receipt, row)
    print(json.dumps(dict(replayed=item["item_id"], chunks=len(valid), state_error=maximum)), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    source = RUN / "native_rehearsal_collection"
    while not (source / "completion.json").exists():
        if not args.follow:
            raise FileNotFoundError(source / "completion.json")
        time.sleep(20)
    model = json.loads((source / "model.json").read_text())
    assert model["adapter_sha256"] is None
    summaries = [json.loads(line) for line in (source / "episodes.jsonl").read_text().splitlines()]
    successful = [row for row in summaries if row["native_success"] and row["steps"] >= 20]
    with multiprocessing.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        rows = sorted(pool.imap_unordered(prepare_one, successful), key=lambda row: row["item_id"])
    assert len(rows) >= 20 and len({row["task_id"] for row in rows}) >= 8
    manifest = DATA / "native_rehearsal_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    audit = dict(passed=True, attempted=len(summaries), accepted=len(rows),
                 rejected=[dict(item_id=row["item_id"], steps=row["steps"], success=row["native_success"])
                           for row in summaries if row not in successful],
                 chunks=sum(row["chunks"] for row in rows),
                 by_task=dict(Counter(row["task_id"] for row in rows)),
                 initial_state_ids=sorted({row["initial_state_id"] for row in rows}),
                 evaluation_initial_state_ids=[0, 1], all_independently_replayed=True,
                 exact_recorded_inputs_and_executed_actions=True, manifest_sha256=sha256(manifest))
    write_json(DATA / "native_rehearsal_audit.json", audit)
    snapshot = DATA / "native_rehearsal_cache/source_snapshot"
    snapshot.mkdir(exist_ok=True)
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    while not (DATA / "training_data_audit.json").exists():
        if not args.follow:
            print("NATIVE_REHEARSAL_READY; motor collection pending", flush=True)
            return
        time.sleep(20)
    motor_audit = json.loads((DATA / "training_data_audit.json").read_text())
    assert motor_audit["collection_complete"] and motor_audit["cache_arrays_bit_identical"]
    motor_manifest = DATA / "train_manifest.jsonl"
    assert sha256(motor_manifest) == motor_audit["manifest_sha256"]
    motor_rows = [json.loads(line) for line in motor_manifest.read_text().splitlines()]
    combined = motor_rows + rows
    assert len({row["item_id"] for row in combined}) == len(combined)
    merged = DATA / "combined_train_manifest.jsonl"
    merged.write_text("".join(json.dumps(row) + "\n" for row in combined))
    result = dict(passed=True, motor_episodes=len(motor_rows), native_episodes=len(rows),
                  motor_sampling_fraction=.9, native_sampling_fraction=.1,
                  native_evaluation_initial_states_excluded=True,
                  motor_manifest_sha256=sha256(motor_manifest), native_manifest_sha256=sha256(manifest),
                  combined_manifest_sha256=sha256(merged), identical_manifest_for_three_scopes=True)
    write_json(DATA / "rehearsal_merge_audit.json", result)
    print("COMBINED_TRAINING_DATA_READY", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
