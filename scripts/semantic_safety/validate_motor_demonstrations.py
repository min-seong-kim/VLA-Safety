"""Replay recorded actions from a fresh seeded environment and audit demonstrations."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from motor_skill_env import make_environment
from rollout_common import ROOT, bootstrap, write_json


def validate(folder):
    with h5py.File(folder / "trajectory.h5", "r") as data:
        spec = json.loads(data.attrs["spec_json"])
        actions = data["actions"][:]
        initial = data["initial_sim_state"][:]
        expected = data["observations/sim_state"][:]
    wrapper, obs, oracle = make_environment(spec)
    env = wrapper.env
    try:
        initial_error = float(np.max(np.abs(env.sim.get_state().flatten() - initial)))
        if initial_error > 1e-7:
            raise ValueError(f"Fresh deterministic reset differs from recorded initial state: {initial_error}")
        errors = []
        metric = oracle.measure(env)
        for action, target_state in zip(actions, expected[1:]):
            obs, _, _, _ = env.step(action)
            metric = oracle.measure(env)
            errors.append(float(np.max(np.abs(env.sim.get_state().flatten() - target_state))))
        result = {"accepted": bool(metric["success"] and max(errors) < 1e-6),
                  "success_on_replay": metric["success"], "initial_state_max_abs_error": initial_error,
                  "trajectory_state_max_abs_error": max(errors), "steps": len(actions),
                  "reset_method": "fresh deterministic seeded reset, no state restoration during execution",
                  "actions_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                  "final_metric": metric}
        write_json(folder / "replay_validation.json", result)
        return result
    finally:
        wrapper.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    bootstrap()
    rows = []
    for path in sorted(args.input.glob("*/summary.json")):
        summary = json.loads(path.read_text())
        if summary["success"]:
            replay_path = path.parent / "replay_validation.json"
            try:
                result = json.loads(replay_path.read_text()) if replay_path.exists() else validate(path.parent)
            except Exception as error:
                result = {"accepted": False, "error": repr(error)}
                write_json(replay_path, result)
            print(json.dumps({"episode_id": summary["episode_id"], **result}), flush=True)
            summary["action_replay_validation"] = result
            write_json(path, summary)
        else:
            result = {"accepted": False, "reason": "teacher failed success predicate"}
        rows.append({"episode_id": summary["episode_id"], "asset": summary["asset"], "split": summary["split"],
                     "path": str((path.parent / "trajectory.h5").resolve().relative_to(ROOT)),
                     "steps": summary["steps"], "teacher_success": summary["success"],
                     "accepted": result["accepted"], "replay": result})
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))


if __name__ == "__main__":
    main()
