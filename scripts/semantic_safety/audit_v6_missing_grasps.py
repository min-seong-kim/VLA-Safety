"""Bounded extra calibration of grasp heights for uncovered thin objects."""

import itertools
import json
import math
import multiprocessing
from pathlib import Path
import traceback

from collect_v6_fast import configure
from rollout_common import write_json
from semantic_pairs_v6 import sha256
from v6_lora_data import DATA, make_spec


def trial(request):
    condition, number, profile = request
    configure()
    import v6_transport_teacher as teacher

    folder = DATA / "extra_grasp_calibration" / f"c{condition:02d}" / f"profile_{number:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "result.json"
    if path.exists():
        return json.loads(path.read_text())
    results = []
    for index in [3000, 3001]:
        spec = make_spec(condition, "calibration", index)
        if condition == 4:
            from collect_v6_front import make_spec as front_spec
            from collect_v6_front import Teacher
            spec = front_spec(condition, "calibration", index)
            teacher.Teacher = Teacher
        try:
            result = teacher.rollout(spec, profile)
            if result["passed"]:
                recorded = folder / f"scene_{index}"
                check = teacher.rollout(spec, profile, folder=recorded, resolution=224)
                assert check["passed"]
                result["replay"] = teacher.replay(recorded)
                assert result["replay"]["accepted"]
        except Exception:
            result = dict(passed=False, error=traceback.format_exc())
        results.append(result)
    value = dict(condition=condition, profile_number=number, profile=profile,
                 successes=sum(result["passed"] for result in results), trials=results,
                 split="calibration", train_manifest_unchanged=True, test_used=False)
    write_json(path, value)
    print(json.dumps({key: value[key] for key in ["condition", "profile_number", "successes"]}), flush=True)
    return value


def main():
    requests = []
    profiles = [dict(mode="bounds", offset=[0., 0.], height_fraction=.5,
                     z_adjust=height, yaw=yaw, tilt=tilt)
                for height, yaw, tilt in itertools.product([-.002, .002, .006, .010, .014],
                                                           [0., -math.pi / 2], [0., -.3, .3])]
    requests += [(15, number, profile) for number, profile in enumerate(profiles)]
    from v6_transport_teacher import prior_profiles
    profiles = []
    for original, shift in itertools.product(prior_profiles("spoon")[:2], [0., .004, .008, .012]):
        profiles.append({**original, "height": original["height"] + shift})
    requests += [(4, number, profile) for number, profile in enumerate(profiles)]
    with multiprocessing.get_context("spawn").Pool(4, maxtasksperchild=1) as pool:
        results = list(pool.imap_unordered(trial, requests))
    write_json(DATA / "extra_grasp_calibration/summary.json", dict(
        profiles=len(results), episodes=sum(len(row["trials"]) for row in results),
        successful_profiles=[row for row in results if row["successes"]],
        script_sha256=sha256(Path(__file__)), train_manifest_unchanged=True,
        purpose="Check teacher grasp-height limitations on new calibration scenes; no changes to frozen training data."))


if __name__ == "__main__":
    main()
