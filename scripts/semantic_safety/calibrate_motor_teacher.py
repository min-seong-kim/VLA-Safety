"""Search grasp profiles on explicitly separated calibration layouts only."""

import argparse
import itertools
import json
import math
from pathlib import Path

from collect_motor_demonstrations import ScriptedTeacher, default_profile
from motor_skill_env import DATA, actor_assets, make_environment, make_spec
from rollout_common import bootstrap, write_json


def candidates(asset):
    base = default_profile(asset)
    yield base
    options = {
        "aerosol": ([.12, .135, .105], [[0., 0.]], [0., -math.pi / 2]),
        "kettle": ([.15, .17, .12], [[0., 0.], [-.035, 0.], [0., .035]], [0., -math.pi / 2]),
        "basin": ([.055, .035], [[.075, 0.], [-.075, 0.], [0., .075]], [-math.pi / 2, 0.]),
        "plastic_bowl": ([.055, .035], [[.075, 0.], [-.075, 0.], [0., .075]], [-math.pi / 2, 0.]),
        "wood_bowl": ([.045, .025], [[.055, 0.], [-.055, 0.], [0., .055]], [-math.pi / 2, 0.]),
        "hammer": ([.012, .02], [[-.04, 0.], [.04, 0.], [0., 0.]], [0.]),
        "power_bank": ([.0, .01], [[0., -.034], [.04, -.034]], [0., -math.pi / 2]),
        "battery": ([.0, -.005, .005], [[0., -.014]], [0., -math.pi / 2]),
        "chocolate": ([.0, -.005, .008], [[0., 0.]], [-math.pi / 2, 0.]),
        "spoon": ([.006, .012, .018], [[0., .04], [0., -.04]], [-math.pi / 2]),
        "wallet": ([.01, .0], [[.04, 0.], [0., .03]], [0., -math.pi / 2]),
        "knife": ([.018, .01], [[0., .04], [0., -.045], [0., 0.]], [-math.pi / 2]),
        "towel": ([.008, .015], [[.02, 0.], [0., .05]], [-math.pi / 2, 0.]),
    }
    if asset in options:
        heights, offsets, yaws = options[asset]
        for height, offset, yaw in itertools.product(heights, offsets, yaws):
            yield {**base, "height": height, "offset": offset, "yaw": yaw, "max_translation_action": .45}


def trial(asset, profile, index):
    spec = make_spec(asset, "calibration", index)
    wrapper, obs, oracle = make_environment(spec, resolution=64)
    env = wrapper.env
    teacher = ScriptedTeacher(env, oracle, profile)
    metric = oracle.measure(env)
    try:
        for step in range(360):
            action = teacher.action(env, obs, metric)
            obs, _, _, _ = env.step(action)
            metric = oracle.measure(env)
            if metric["success"] and teacher.stage == "retract":
                break
        return {"spec": spec, "success": metric["success"], "steps": step + 1,
                "final_metric": metric, "final_stage": teacher.stage}
    finally:
        wrapper.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--assets", default=",".join(a for a in actor_assets() if a != "rice_cake"))
    parser.add_argument("--out", type=Path, default=DATA / "profile_search")
    args = parser.parse_args()
    bootstrap()
    args.out.mkdir(parents=True, exist_ok=True)
    chosen = {}
    for asset in args.assets.split(","):
        results = []
        for candidate, profile in enumerate(candidates(asset)):
            path = args.out / f"{asset}_{candidate:03d}.json"
            if path.exists():
                result = json.loads(path.read_text())
            else:
                first = trial(asset, profile, 0)
                trials = [first]
                if first["success"]:
                    trials.append(trial(asset, profile, 1))
                result = {"asset": asset, "profile": profile, "candidate": candidate,
                          "trials": trials, "calibration_successes": sum(t["success"] for t in trials)}
                write_json(path, result)
            print(json.dumps({"asset": asset, "candidate": candidate,
                              "successes_of_2": result["calibration_successes"]}), flush=True)
            results.append(result)
            if result["calibration_successes"] == 2:
                break
        best = max(results, key=lambda r: (r["calibration_successes"], -r["candidate"]))
        if best["calibration_successes"] > 0:
            chosen[asset] = best["profile"]
        write_json(args.out / "selected_profiles.json", chosen)
    write_json(args.out / "coverage.json", {"included": sorted(chosen),
               "excluded": sorted(set(args.assets.split(",")) - set(chosen)),
               "selection": "first profile succeeding on both calibration layouts; otherwise most successes",
               "held_out_validation_or_test_used": False})


if __name__ == "__main__":
    main()
