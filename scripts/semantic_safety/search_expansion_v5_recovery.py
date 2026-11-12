"""Search extra grasp profiles on training layouts without altering the frozen physics."""

import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path

from collect_expansion_v5_demonstrations import signature
from expansion_v5_motor import DATA, make_spec, sha
from expansion_v5_teacher import candidates, rollout
from rollout_common import write_json


def proposals(spec, calibration):
    rows = []

    def add(profile):
        if profile not in rows:
            rows.append(profile)

    add(calibration)
    # A half-turn preserves the parallel-jaw pinch but changes arm reachability.
    for turn in [math.pi, -math.pi, math.pi / 2, -math.pi / 2]:
        add({**calibration, "yaw": calibration["yaw"] + turn})
    if spec["actor_kind"] == "mallet":
        for offset in [(0, 0), (-.25, 0), (.25, 0), (0, .25), (0, -.25)]:
            for yaw in [math.pi / 2, 0., -math.pi / 2]:
                for height in [.5, .7, .3]:
                    add({**calibration, "offset": list(offset), "yaw": yaw, "height_fraction": height})
    if spec["actor_kind"] == "mesh:bread":
        for offset in [(.32, 0), (.42, 0), (-.32, 0), (-.42, 0), (0, .32), (0, -.32)]:
            for height in [.6, .4, .8]:
                for yaw in [0., math.pi / 2]:
                    add(dict(offset=list(offset), height_fraction=height, yaw=yaw, tilt=0., z_adjust=-.002))
    for profile in candidates(spec):
        add(profile)
    return rows


def refined_bread_profiles(calibration):
    rows=[]
    for x in [.34,.38,.30,.42]:
        for height in [.6,.7,.8,.5]:
            for dz in [.002,.006,-.006]:
                rows.append({**calibration,"offset":[x,0],"height_fraction":height,"z_adjust":dz})
    return rows


def probe(request):
    condition, index, number, profile, out = request
    path = Path(out) / f"c{condition:03d}_i{index:04d}_p{number:03d}.json"
    expected = signature()
    if path.exists():
        row = json.loads(path.read_text())
        assert row["signature"] == expected and row["profile"] == profile
        return row
    result = rollout(make_spec(condition, "train", index), profile)
    row = dict(condition=condition, index=index, profile_number=number, profile=profile,
               signature=expected, search_script_sha256=sha(__file__), result=result)
    write_json(path, row)
    if result["passed"]:
        print(json.dumps(dict(condition=condition, index=index, profile_number=number, passed=True)), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--indices", default="0")
    parser.add_argument("--profiles", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tag", default="collection_r10")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--refine-bread", action="store_true")
    args = parser.parse_args()
    tasks = []
    for condition in map(int, args.ids.split(",")):
        calibration = json.loads((DATA / args.tag / "profiles" / f"c{condition:03d}.json").read_text())
        for index in map(int, args.indices.split(",")):
            rows = (refined_bread_profiles(calibration["profile"]) if args.refine_bread else
                    proposals(make_spec(condition, "train", index), calibration["profile"]))
            tasks.extend((condition, index, n, p, str(args.out)) for n, p in enumerate(rows[:args.profiles]))
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = list(pool.imap_unordered(probe, tasks))
    write_json(args.out / "summary.json", dict(trials=len(results), successes=sum(r["result"]["passed"] for r in results)))


if __name__ == "__main__":
    main()
