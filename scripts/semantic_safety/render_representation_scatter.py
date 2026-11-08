"""Scatter views of existing held-out robot semantic-safety representations.

No model inference, probe fitting, or data-dependent direction search is run.
Binary LDA has one discriminant axis. The second axis is a fixed random linear
projection of the same hidden states, standardized with training scenes only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib

os.environ.setdefault("MPLCONFIGDIR", str(pathlib.Path(__file__).resolve().parents[2] / ".cache/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = ("pi05", "pi05_libero", "openvla_oft")
MODEL_LABELS = ("π0.5 base", "π0.5 LIBERO", "OpenVLA-OFT Spatial")
SEED = 20261106


def coordinates(run: pathlib.Path, model: str, condition: str, folds: list[dict]) -> dict:
    feature_path = run / "features" / model / f"{condition}.npz"
    prediction_path = run / "analysis" / model / condition / "oof_predictions.npz"
    with np.load(feature_path, allow_pickle=False) as packed:
        h = packed["h"].astype(np.float64)
        item_ids, groups, labels = packed["item_id"], packed["group_id"], packed["label"]
    with np.load(prediction_path, allow_pickle=False) as packed:
        for name, expected in (("item_id", item_ids), ("group_id", groups), ("label", labels)):
            if not np.array_equal(packed[name], expected):
                raise ValueError(f"Feature/prediction alignment changed: {model}/{condition}/{name}")
        x = packed["lda_projection"].copy()
        fold_ids = packed["fold"].copy()
    # One predeclared direction per hidden dimension, shared across conditions.
    # The vector is chosen without looking at labels or separation quality.
    direction = np.random.default_rng(SEED + h.shape[1]).standard_normal(h.shape[1])
    direction /= np.linalg.norm(direction)
    y = np.full(len(h), np.nan)
    for fold in folds:
        train = np.flatnonzero(np.isin(groups, fold["train"]))
        test = np.flatnonzero(fold_ids == fold["fold"])
        if set(groups[test]) != set(fold["test"]):
            raise ValueError("Held-out groups do not match the saved evaluation fold")
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Scene leakage in auxiliary projection normalization")
        mean = h[train].mean(axis=0)
        scale = h[train].std(axis=0)
        scale = np.where(scale == 0, 1, scale)
        train_projection = ((h[train] - mean) / scale) @ direction
        test_projection = ((h[test] - mean) / scale) @ direction
        normalization = float(train_projection.std())
        if normalization <= 1e-12:
            raise ValueError("Degenerate auxiliary projection")
        y[test] = (test_projection - train_projection.mean()) / normalization
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Incomplete or non-finite plot coordinates")
    return {"item_id": item_ids, "group_id": groups, "label": labels, "fold": fold_ids,
            "lda_score": x, "auxiliary_projection": y}


def render(run: pathlib.Path, destination: pathlib.Path, condition: str, packs: dict, results: dict,
           fold_id: int | None = None) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    selected = {}
    for model in MODELS:
        packed = packs[model]
        selected[model] = (np.ones(len(packed["label"]), dtype=bool) if fold_id is None
                           else packed["fold"] == fold_id)
    all_x = np.concatenate([packs[m]["lda_score"][selected[m]] for m in MODELS])
    all_y = np.concatenate([packs[m]["auxiliary_projection"][selected[m]] for m in MODELS])
    span_x, span_y = max(float(np.ptp(all_x)), 1), max(float(np.ptp(all_y)), 1)
    for axis, model, name in zip(axes, MODELS, MODEL_LABELS, strict=True):
        packed, selection = packs[model], selected[model]
        for label, color, description in ((0, "#2878b5", "Source non-undesirable"),
                                           (1, "#f28e2b", "Source undesirable")):
            mask = selection & (packed["label"] == label)
            axis.scatter(packed["lda_score"][mask], packed["auxiliary_projection"][mask],
                         s=31, c=color, alpha=0.76, edgecolors="white", linewidths=0.35,
                         label=f"{description} (n={int(mask.sum())})", rasterized=True)
        metrics = results[model][condition]["methods"]["lda"]
        auc = (metrics["mean_fold_auroc"] if fold_id is None else metrics["per_fold"][fold_id]["auroc"])
        auc_label = "Mean held-out-fold" if fold_id is None else f"Held-out fold {fold_id}"
        axis.set_title(f"{name}\n{auc_label} LDA AUROC = {auc:.3f}", fontsize=12)
        axis.set_xlabel("LDA discriminant score (train-normalized)", fontsize=10)
        axis.grid(alpha=0.13)
        axis.set_xlim(float(all_x.min()) - 0.06 * span_x, float(all_x.max()) + 0.06 * span_x)
        axis.set_ylim(float(all_y.min()) - 0.08 * span_y, float(all_y.max()) + 0.08 * span_y)
    axes[0].set_ylabel("Fixed auxiliary linear projection\n(train-normalized; no label fitting)", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.025))
    condition_title = ("Robot instruction prefix — no added safety question" if condition == "instruction"
                       else "Robot-scene safety question prefix")
    fig.suptitle(condition_title, fontsize=14, y=0.99)
    if fold_id is None:
        caption = ("Each dot is an out-of-fold case; every scene was excluded from its LDA training. "
                   "Cross-fitted axis scores, not one global 2-D LDA embedding.")
    else:
        caption = ("All displayed cases share one training-derived projection per model. "
                   "Fold 0 was fixed in advance, not selected for visual separation.")
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.23, top=0.82, wspace=0.1)
    stem = f"scatter_{condition}_" + ("oof" if fold_id is None else f"fold{fold_id}")
    files = []
    for extension in ("png", "pdf"):
        path = destination / f"{stem}.{extension}"
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=220)
        files.append(str(path.relative_to(run)))
    plt.close(fig)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=pathlib.Path, required=True)
    args = parser.parse_args()
    folds = json.loads((args.run / "folds.json").read_text())
    results = json.loads((args.run / "results.json").read_text())
    destination = args.run / "representation_scatter"
    destination.mkdir(exist_ok=False)
    outputs = []
    for condition in ("instruction", "query"):
        packs = {model: coordinates(args.run, model, condition, folds) for model in MODELS}
        for model, packed in packs.items():
            np.savez(destination / f"coordinates_{model}_{condition}.npz", **packed)
        outputs.extend(render(args.run, destination, condition, packs, results))
        outputs.extend(render(args.run, destination, condition, packs, results, fold_id=0))
    metadata = {
        "source_run": str(args.run), "method": "Existing held-out 1-D LDA score plus fixed auxiliary linear projection",
        "x": "Saved held-out LDA score, normalized using its own fold training scenes",
        "y": "Fixed seeded Gaussian direction on standardized hidden states; train-only normalization",
        "seed": SEED, "direction_seed": "seed + hidden dimension; same vector across prompt conditions",
        "new_model_inference": False, "new_probe_fitting": False,
        "class_dependent_coordinate_offsets": False, "direction_search_or_test_label_fitting": False,
        "oof_caveat": "Each fold has its own training normalization/LDA axis. OOF panels combine cross-fitted scores.",
        "single_projection_control": "Separate fold-0 plots show one learned projection per model; fold fixed a priori",
        "label_definition": "Original ASIMOV undesirable_groundtruth_answer; not newly verified expert harm labels",
        "source_code_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "outputs": outputs,
    }
    (destination / "projection_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"status": "complete", "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
