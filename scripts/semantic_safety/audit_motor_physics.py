"""Negative oracle controls and a frozen scripted-teacher ceiling on held-out layouts."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path

import numpy as np

from motor_skill_env import make_environment, make_spec
from rollout_common import bootstrap, write_json


def run(request):
    from collect_motor_demonstrations import ScriptedTeacher

    bootstrap()
    spec, profile, mode = request
    wrapper, obs, oracle = make_environment(spec, 64)
    env = wrapper.env
    teacher = ScriptedTeacher(env, oracle, profile)
    metric = oracle.measure(env)
    max_drift = 0.
    try:
        for step in range(360):
            action = [0.] * 6 + [-1.] if mode == "noop" else teacher.action(env, obs, metric)
            obs, _, _, _ = env.step(action)
            metric = oracle.measure(env)
            max_drift = max(max_drift, float(np.linalg.norm(np.asarray(metric["actor_position"]) - oracle.initial)))
            if mode == "teacher" and metric["success"] and teacher.stage == "retract":
                break
        return {"spec": spec, "mode": mode, "success": metric["success"], "steps": step + 1,
                "max_actor_displacement_m": max_drift, "final_metric": metric}
    finally:
        wrapper.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["noop", "teacher"], default="noop")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    design = json.loads(args.design.read_text())
    split = "calibration" if args.mode == "noop" else "test"
    indices = [2] if args.mode == "noop" else design["test_indices"]
    requests = [(make_spec(asset, split, index), design["profiles"][asset], args.mode)
                for asset in design["assets"] for index in indices]
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(run, request) for request in requests]):
            row = future.result()
            results.append(row)
            write_json(args.out / (row["spec"]["episode_id"] + ".json"), row)
            print(json.dumps({"id": row["spec"]["episode_id"], "mode": args.mode,
                              "success": row["success"], "max_displacement": row["max_actor_displacement_m"]}), flush=True)
    report = {"mode": args.mode, "episodes": len(results), "successes": sum(row["success"] for row in results),
              "teacher_profiles_frozen_before_evaluation": True, "used_for_model_selection": False,
              "results": results}
    write_json(args.out / "aggregate.json", report)
    if args.mode == "noop":
        assert report["successes"] == 0


if __name__ == "__main__":
    main()
