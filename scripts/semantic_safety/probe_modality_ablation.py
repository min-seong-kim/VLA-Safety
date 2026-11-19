"""OOF readouts from an unmasked probe and a probe fitted anew under the same mask."""

import argparse
import csv
from multiprocessing import get_context
from unittest.mock import patch

import h5py
import joblib
import numpy as np

import contrast_initial_linear as contrast
from modality_ablation import AE, COMMON, REFERENCE, RUN, SITES, STATE_BASE, VLM, conditions
from ae_output_nullspace_pilot import read
from rollout_common import write_json


def csv_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def evaluate(job):
    name, condition, track, site = job
    folder = RUN / "conditions" / condition
    with patch.object(contrast, "RUN", folder):
        fit = contrast.probe_job((name, track, site))
    manifest = read(REFERENCE / "manifest.json")
    archived = REFERENCE if name != STATE_BASE else RUN / "conditions/intact"
    reference = archived / "probes" / track / name / site
    original = read(reference / "metrics.json")
    with np.load(reference / "oof_predictions.npz") as r:
        indices, fold, original_score = (r[k].copy() for k in ("global_indices", "fold", "decision_score"))
    rows = [manifest["rows"][i] for i in indices]
    with h5py.File(folder / "features" / (name + ".h5")) as f:
        assert f.attrs["complete"] and f["written"][:].all()
        matrix = f[site][:][indices].astype(np.float64)
    fixed = np.full(len(rows), np.nan)
    output = folder / "probes" / track / name / site
    with np.load(output / "oof_predictions.npz") as obj:
        assert np.array_equal(fold, obj["fold"]) and np.array_equal(indices, obj["global_indices"])
        refit = obj["decision_score"].copy()
    by_fold = []
    for selection in manifest["folds"]:
        k = selection["fold"]
        positions = np.flatnonzero(fold == k)
        fold_fit = joblib.load(reference / f"fold_{k}.joblib")
        assert fold_fit["split"] == selection
        assert not set(fold_fit["training_rows"]) & {rows[i]["item_id"] for i in positions}
        fixed[positions] = fold_fit["classifier"].decision_function(fold_fit["scaler"].transform(matrix[positions]))
        for method, scores in (("fixed", fixed), ("refit", refit)):
            metric = contrast.metrics([rows[i] for i in positions], scores[positions])
            by_fold.append(dict(model=name, condition=condition, track=track, site=site, method=method,
                                fold=k, n=len(positions), auroc=metric["AUROC"],
                                balanced_accuracy=metric["balanced_accuracy"], accuracy=metric["accuracy"],
                                c=(fold_fit["classifier"].C if method == "fixed" else fit["folds"][k]["C"])))
    assert np.isfinite(fixed).all()
    if condition == "intact" or (site == VLM and condition.startswith("ae_")):
        np.testing.assert_allclose(fixed, original_score, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(refit, original_score, rtol=1e-7, atol=1e-7)
    np.savez_compressed(output / "fixed_oof.npz", global_indices=indices, fold=fold,
                        label=[r["label"] for r in rows], decision_score=fixed)
    summary = []
    for method, scores in (("fixed", fixed), ("refit", refit)):
        model_rows = [r for r in by_fold if r["method"] == method]
        metric = contrast.metrics(rows, scores)
        auc = np.mean([r["auroc"] for r in model_rows])
        tn, fp = metric["confusion_matrix"][0]
        fn, tp = metric["confusion_matrix"][1]
        summary.append(dict(model=name, condition=condition, track=track, site=site, method=method,
                            n=len(scores), fold_mean_auroc=auc, delta_vs_intact=auc-original["fold_mean_AUROC"],
                            fold_mean_balanced_accuracy=np.mean([r["balanced_accuracy"] for r in model_rows]),
                            oof_accuracy=metric["accuracy"], predicted_unsafe_fraction=metric["predicted_unsafe_fraction"],
                            oof_tn=tn, oof_fp=fp, oof_fn=fn, oof_tp=tp,
                            label_flip_n=int(np.sum((scores>=0) != (original_score>=0)))))
    return summary, by_fold


def run(workers, pooling=False):
    names = COMMON + (STATE_BASE,)
    def jobs(baseline):
        return [(m,c,t,s) for m in names for c in conditions(m) if (c == "intact") == baseline
                for t in ["A", "B"] for s in SITES]
    with get_context("spawn").Pool(workers) as pool:
        # The supplementary state-format Base has no archived probe. Fit its intact condition first.
        if pooling:
            results = pool.map(evaluate, [(m,c,t,VLM) for m in COMMON for c in ("read_image", "read_text")
                                          for t in ("A", "B")])
        else:
            results = pool.map(evaluate, jobs(True))
            results += pool.map(evaluate, jobs(False))
    prefix = "pooling" if pooling else "probe"
    for filename, idx in [(prefix + "_summary", 0), (prefix + "_folds", 1)]:
        csv_rows(RUN / (filename + ".csv"), [row for group in results for row in group[idx]])
    if pooling:
        return
    from experiment_notifications import notify_progress
    write_json(RUN / "milestones/probes.json", {"complete": True, "title": "입력 마스킹 첫 장면 probe 평가 완료",
               "result": {"decisions": 457, "model_condition_pairs": 28, "tracks": 2, "sites": [VLM, AE]}})
    notify_progress(RUN)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--pooling", action="store_true")
    args = p.parse_args()
    run(args.workers, args.pooling)
