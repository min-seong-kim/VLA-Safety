"""Test a slower physical spoon transport teacher on new training/calibration scenes."""

import argparse
import copy
import json
import multiprocessing
from pathlib import Path

import numpy as np

from collect_v6_fast import configure as fast_configure
from collect_v6_front import make_spec as front_spec
from rollout_common import write_json
from semantic_pairs_v6 import sha256
from v6_lora_data import DATA
from v6_pose_teacher import Teacher as PoseTeacher
from v6_transport_teacher import prior_profiles


class Teacher(PoseTeacher):
    def __init__(self, env, oracle, spec, profile):
        super().__init__(env, oracle, spec, profile)
        self.target[0] += .025
        self.target[2] -= .02
        self.carry_z = max(float(self.target[2]), float(self.initial_reference[2] + .085))

    def action(self, env, obs, metric):
        result = super().action(env, obs, metric)
        result[:3] = np.clip(result[:3], -.18, .18)
        return result


def profiles(asset):
    assert asset == "spoon"
    result = []
    mined = json.loads((DATA / "development_grasp_geometry.json").read_text())[asset]
    for shift in [0., -.003, .003]:
        for value in mined:
            profile = copy.deepcopy(value)
            profile["position_local"][2] += shift
            result.append(profile)
    result.extend(prior_profiles(asset))
    return result


def make_spec(condition, split, index):
    spec = front_spec(condition, split, index)
    spec["item_id"] = "slow_spoon__" + spec["item_id"]
    spec["pair_group"] = "slow_spoon__" + spec["pair_group"]
    spec["teacher_family"] = "slower_spoon_transport"
    return spec


def process(request):
    fast_configure()
    import collect_v6_lora_data as collector
    import v6_transport_teacher

    v6_transport_teacher.Teacher = Teacher
    collector.make_spec = make_spec
    collector.candidate_profiles = profiles
    collector.__file__ = __file__
    return collector.process(request)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--tag", default="slow_spoon_pilot")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--start", type=int, default=2000)
    parser.add_argument("--maximum", type=int, default=4)
    args = parser.parse_args()
    result = process((4, args.split, args.count, args.start, args.maximum, args.tag))
    output = DATA / args.tag
    output.mkdir(parents=True, exist_ok=True)
    rows = result["accepted"]
    (output / f"{args.split}_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(output / f"{args.split}_summary.json", dict(episodes=len(rows), complete=result["complete"], conditions=[result]))
    write_json(output / "provenance.json", dict(script_sha256=sha256(Path(__file__)), source_v6_changed=False))
    print(json.dumps(dict(episodes=len(rows), complete=result["complete"])), flush=True)


if __name__ == "__main__":
    main()
