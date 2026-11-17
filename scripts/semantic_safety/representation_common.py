"""Frozen inputs and family partitions shared by the representation experiment."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

from rollout_common import ROOT, write_json

RUN = ROOT / "runs/semantic_safety_representation_linear_v1"
DATA = ROOT / "data/semantic_expansion_v5_lora_v1"
BOTH = ROOT / "runs/semantic_expansion_v5_lora_v1/final/all/motor"
TEACHER = ROOT / "data/semantic_safety_representation_teacher_v1"
MODELS = ("baseline", "ae", "vlm", "all")
FLOW_STEPS = (1, 5, 10)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def immutable_json(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, f"Frozen definition changed: {path}"
    else:
        write_json(path, value)


def milestone(name, title, result):
    from experiment_notifications import notify_progress
    write_json(RUN / "milestones" / f"{name}.json", {
        "complete": True, "title": title, "result": result, "completed_utc": now()})
    return notify_progress(RUN)


def prepare():
    conditions = read_rows(DATA / "conditions.jsonl")
    specs = read_rows(DATA / "holdouts/test.jsonl")
    results = {row["item_id"]: row for row in read_rows(BOTH / "episodes.jsonl")}
    assert len(specs) == len(results) == 676
    episodes = []
    for spec in sorted(specs, key=lambda row: row["item_id"]):
        result = results[spec["item_id"]]
        assert spec["instruction"] == result["instruction"]
        episodes.append({"episode_id": spec["item_id"], "condition_index": spec["condition_index"],
            "family_id": spec["family_id"], "category": spec["category"],
            "pair_id": spec["pair_id"], "pair_group": spec["pair_group"],
            "label": int(spec["side"] == "unsafe"), "instruction": spec["instruction"],
            "noise_seed": spec["noise_seed"], "both_success": result["success"],
            "both_steps": result["steps"], "initial_image_sha256": result["initial_image_sha256"]})
    assert Counter(row["label"] for row in episodes) == {0: 338, 1: 338}
    families = Counter(row["family_id"] for row in episodes)
    assert len(families) == 35
    category = {row["family_id"]: row["category"] for row in episodes}
    rng = random.Random(20261112)
    order = list(families)
    rng.shuffle(order)
    order.sort(key=lambda family: -families[family])
    bins = [[] for _ in range(5)]
    totals = [0] * 5
    categories = [Counter() for _ in range(5)]
    for family in order:
        choices = [i for i in range(5) if len(bins[i]) < 7]
        selected = min(choices, key=lambda i: (categories[i][category[family]], totals[i], len(bins[i]), i))
        bins[selected].append(family)
        totals[selected] += families[family]
        categories[selected][category[family]] += families[family]
    folds = []
    for index in range(5):
        test, validation = (index + 4) % 5, (index + 3) % 5
        fold = {"fold": index, "test_families": sorted(bins[test]),
                "validation_families": sorted(bins[validation]),
                "train_families": sorted(f for i, block in enumerate(bins) if i not in (test, validation) for f in block)}
        parts = [set(fold[key]) for key in ("train_families", "validation_families", "test_families")]
        assert len(parts[0]) == 21 and len(parts[1]) == len(parts[2]) == 7
        assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
        folds.append(fold)
    immutable_json(RUN / "episodes.json", episodes)
    immutable_json(RUN / "splits.json", {"seed": 20261112, "group": "family_id", "folds": folds,
        "episode_counts_by_bin": totals, "labels_used_for_partition": False,
        "limitations": "Families were seen by the LoRA policies; this holds families out from probe fitting only."})
    return episodes


def samples(source, view, max_samples=None):
    import h5py
    rows = json.loads((RUN / "episodes.json").read_text())
    result = []
    for episode in rows:
        if max_samples is not None and len(result) >= max_samples:
            break
        root = BOTH / "episodes" if source == "both" else TEACHER / "episodes"
        path = root / episode["episode_id"] / "trajectory.h5"
        if source == "teacher":
            receipt = json.loads(path.with_name("receipt.json").read_text())
            assert receipt["accepted"] and receipt["replay_verified"]
        with h5py.File(path, "r") as data:
            steps = data["chunk_start_steps"][:].tolist()
            spec = json.loads(data.attrs["item_json"])
            assert spec["item_id"] == episode["episode_id"] and spec["instruction"] == episode["instruction"]
            assert len(steps) == len(data["policy_inputs/state"])
        indices = [0] if view == "initial" else range(len(steps))
        for index in indices:
            if max_samples is not None and len(result) >= max_samples:
                break
            result.append({**episode, "path": str(path.relative_to(ROOT)), "input_index": index,
                "timestep": steps[index], "decision_count": len(steps),
                "phase": index / max(1, len(steps) - 1), "source": source})
    return result


def feature_sites():
    for layer in range(18):
        yield f"vlm/layer_{layer:02d}", 2048
    for step in FLOW_STEPS:
        for layer in range(18):
            yield f"ae/flow_{step:02d}/layer_{layer:02d}", 1024
        yield f"velocity_pre/flow_{step:02d}", 1024
        yield f"velocity_post/flow_{step:02d}", 32


if __name__ == "__main__":
    print(json.dumps({"episodes": len(prepare()), "run": str(RUN)}))
