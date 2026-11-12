"""Export audited hidden states and fixed mechanism-held-out partitions for probing.

No classifier is fitted here. Normalization and dimension reduction must be fitted
on the training split only. Default sampling uses one initial state per instruction.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from rollout_common import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/semantic_rollouts_v3")
    parser.add_argument("--feature", default="vlm_hidden")
    parser.add_argument("--region", default="last_prompt")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--forward-call", type=int, default=-1)
    parser.add_argument("--timesteps", choices=["initial", "all"], default="initial")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.stride < 1:
        raise ValueError("stride must be positive")
    args.run_root = args.run_root.resolve()
    splits = json.loads((ROOT / "data/semantic_rollout_v3/probe_splits.json").read_text())["folds"][args.fold]
    assignment = {family: split for split in ["train", "validation", "test"] for family in splits[split + "_families"]}
    paths = sorted((args.run_root / args.model / "episodes").glob("*/summary.json"))
    if len(paths) != 72 and not args.allow_partial:
        raise RuntimeError(f"Need 72 complete paired episodes, found {len(paths)}")
    vectors, labels, records = [], [], []
    kind = "expert" if args.feature.startswith("action_expert") else "head" if args.feature.startswith("action_head") else "vlm"
    for path in paths:
        folder = path.parent
        audit = json.loads((folder / "audit.json").read_text())
        if audit["status"] != "passed":
            raise ValueError(f"Episode did not pass audit: {folder}")
        meta = json.loads((folder / "episode.json").read_text())
        if meta.get("task_is_unsafe") is None:
            raise ValueError("Do not automatically label native LIBERO tasks safe")
        with h5py.File(folder / "features.h5") as features:
            step_meta = json.loads(features["step_metadata"][0])
            dataset = features["pooled/" + args.feature]
            names = step_meta["region_names"].get(kind, [])
            region = names.index(args.region) if args.region in names else int(args.region)
            if "absent" in args.region:
                raise ValueError("This region is a masked/absent modality")
            steps = [0] if args.timesteps == "initial" else range(0, dataset.shape[0], args.stride)
            for step in steps:
                vector = dataset[step, args.forward_call, args.layer, 0, region, :].astype(np.float32)
                assert np.isfinite(vector).all()
                vectors.append(vector)
                labels.append(int(meta["task_is_unsafe"]))
                records.append({"episode_id": meta["episode_id"], "family_id": meta["family_id"],
                                "mechanism_group": meta["mechanism_group"], "timestep": step,
                                "partition": assignment[meta["family_id"]], "model": args.model,
                                "features": str((folder / "features.h5").relative_to(ROOT))})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=np.stack(vectors), y=np.asarray(labels, dtype=np.int64),
                        partition=np.asarray([r["partition"] for r in records]),
                        family_id=np.asarray([r["family_id"] for r in records]),
                        mechanism_group=np.asarray([r["mechanism_group"] for r in records]),
                        timestep=np.asarray([r["timestep"] for r in records]),
                        episode_id=np.asarray([r["episode_id"] for r in records]))
    write_json(args.out.with_suffix(".json"), {"model": args.model, "feature": args.feature, "region": args.region,
               "layer": args.layer, "forward_call": args.forward_call, "timestep_sampling": args.timesteps,
               "fold": args.fold, "matrix_shape": [len(vectors), len(vectors[0])], "rows": records,
               "label": "explicit instruction's authored semantic safety label, not action failure or observed harm",
               "fit_preprocessing_on_train_only": True, "all_frames_from_family_share_partition": True})
    print(json.dumps({"path": str(args.out), "shape": [len(vectors), len(vectors[0])],
                      "partitions": {s: sum(r["partition"] == s for r in records) for s in ["train", "validation", "test"]}}))


if __name__ == "__main__":
    main()
