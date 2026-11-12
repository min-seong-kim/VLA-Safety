"""Collect replay-verified spoon demonstrations with the calibrated 8mm height correction."""

import json
import multiprocessing
from pathlib import Path
import shutil

from collect_v6_fast import configure
from collect_v6_front import Teacher, make_spec as front_spec
from finalize_v6_lora_data import cache_one
from rollout_common import ROOT, write_json
from semantic_pairs_v6 import sha256
from v6_lora_data import DATA

PROFILE = dict(mode="legacy", height=.019098405893373395, offset=[0., .03780000000000001],
               yaw=-1.5707963267948966)


def collect(shard):
    configure()
    import v6_transport_teacher as teacher
    teacher.Teacher = Teacher
    accepted = []
    for index in range(3000 + shard * 100, 3012 + shard * 100):
        spec = front_spec(4, "train", index)
        spec["item_id"] = "spoon_recovery__" + spec["item_id"]
        spec["pair_group"] = "spoon_recovery__" + spec["pair_group"]
        spec["teacher_family"] = "calibration_verified_spoon_height_plus_8mm"
        folder = DATA / "spoon_recovery/train/c04" / spec["item_id"]
        folder.mkdir(parents=True, exist_ok=True)
        receipt = folder / "receipt.json"
        if receipt.exists():
            row = json.loads(receipt.read_text())
            assert row["collector_sha256"] == sha256(Path(__file__))
            if row["accepted"]:
                accepted.append(cache_one(str(receipt)))
        else:
            result = teacher.rollout(spec, PROFILE)
            row = dict(item_id=spec["item_id"], condition_index=4, asset="spoon", split="train",
                       accepted=False, pair_group=spec["pair_group"], source_item_id=spec["source_item_id"],
                       collector_sha256=sha256(Path(__file__)))
            if result["passed"]:
                result = teacher.rollout(spec, PROFILE, folder=folder, resolution=224)
                assert result["passed"]
                replay = teacher.replay(folder)
                assert replay["accepted"]
                source = folder / "trajectory.h5"
                row.update(accepted=True, steps=result["steps"], path=str(source.relative_to(ROOT)),
                           profile=PROFILE, h5_sha256=sha256(source))
            write_json(folder / "attempt.json", result)
            write_json(receipt, row)
            if row["accepted"]:
                accepted.append(cache_one(str(receipt)))
            print(json.dumps(dict(shard=shard, index=index, accepted=row["accepted"], successes=len(accepted))), flush=True)
        if len(accepted) >= 6:
            break
    return accepted


def main():
    with multiprocessing.get_context("spawn").Pool(8, maxtasksperchild=1) as pool:
        rows = sorted([row for group in pool.imap_unordered(collect, range(8)) for row in group],
                      key=lambda row: row["item_id"])
    output = DATA / "spoon_recovery"
    (output / "train_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(output / "train_summary.json", dict(episodes=len(rows), requested=48, complete=len(rows) >= 48,
               profile=PROFILE, independent_replay_for_all=True, no_changes_to_original_assets_or_goals=True))
    snapshot = output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in [Path(__file__).name, "v6_transport_teacher.py", "collect_v6_front.py", "v6_lora_data.py"]:
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    assert len(rows) >= 48
    print("SPOON_RECOVERY_COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    main()
