"""Family-held-out logistic probes; no policy optimization or LDA fitting."""

import argparse
from collections import Counter, defaultdict
import fcntl
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback
import warnings

import h5py
import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from representation_common import MODELS, RUN, digest, feature_sites, milestone
from rollout_common import write_json

CONFIG = {"schema": 1, "method": "L2_logistic_regression", "C_grid": [0.01, 0.1, 1.0],
          "solver": "lbfgs", "max_iter": 400, "retry_max_iter": 2000, "tol": 1e-4,
          "selection_metric": "family_episode_balanced_validation_AUROC", "threshold_logit": 0.,
          "sample_weights": "equal family mass; equal episode mass within family; equal frames within episode; sum=episodes",
          "standardization": "weighted_training_only", "refit_on_validation": False}


def weights(rows):
    counts = Counter(row["episode_id"] for row in rows)
    families = defaultdict(set)
    for row in rows:
        families[row["family_id"]].add(row["episode_id"])
    value = np.asarray([1. / (len(families[row["family_id"]]) * counts[row["episode_id"]]) for row in rows])
    return value * len(counts) / value.sum()


def metrics(rows, labels, scores):
    if len(rows) == 0 or len(np.unique(labels)) < 2:
        return {"samples": len(rows), "AUROC": None, "balanced_accuracy": None}
    sample_weights = weights(rows)
    grouped = defaultdict(list)
    episode_rows = {}
    for row, score in zip(rows, scores, strict=True):
        grouped[row["episode_id"]].append(float(score))
        episode_rows[row["episode_id"]] = row
    episode_scores = {key: float(np.mean(values)) for key, values in grouped.items()}
    paired = defaultdict(dict)
    for key, score in episode_scores.items():
        row = episode_rows[key]
        paired[row["pair_group"]][row["label"]] = score
    pair_scores = [float(pair[1] > pair[0]) + .5 * float(pair[1] == pair[0])
                   for pair in paired.values() if set(pair) == {0, 1}]
    unique_rows = list(episode_rows.values())
    episode_y = np.asarray([row["label"] for row in unique_rows])
    episode_s = np.asarray([episode_scores[row["episode_id"]] for row in unique_rows])
    return {"samples": len(rows), "episodes": len(grouped), "families": len({row["family_id"] for row in rows}),
        "AUROC": float(roc_auc_score(labels, scores, sample_weight=sample_weights)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, scores >= 0, sample_weight=sample_weights)),
        "episode_AUROC": float(roc_auc_score(episode_y, episode_s, sample_weight=weights(unique_rows))),
        "paired_episode_order_accuracy": float(np.mean(pair_scores)) if pair_scores else None,
        "paired_episodes": len(pair_scores)}


def one_site(request):
    feature_root, output_root, source, view, model, site, threads = request
    with threadpool_limits(limits=threads):
        return fit_site(Path(feature_root), Path(output_root), source, view, model, site)


def fit_site(feature_root, output_root, source, view, model, site):
    output = output_root / source / view / model / site
    output.mkdir(parents=True, exist_ok=True)
    with (output / "fit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _fit_site(feature_root, output_root, source, view, model, site)


def _fit_site(feature_root, output_root, source, view, model, site):
    started = time.monotonic()
    folder = feature_root / source / view
    rows = json.loads((folder / "rows.json").read_text())
    folds = json.loads((RUN / "splits.json").read_text())["folds"]
    output = output_root / source / view / model / site
    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(folder / f"{model}.h5", "r") as features:
        assert bool(features.attrs["complete"])
        signature = digest({"rows": features.attrs["rows_digest"], "model": features.attrs["model_json"],
                            "site": site, "config": CONFIG, "folds": folds})
        if (output / "metrics.json").exists():
            cached = json.loads((output / "metrics.json").read_text())
            assert cached["signature"] == signature
            return cached
        matrix = features[site][:].astype(np.float64)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int8)
    family = np.asarray([row["family_id"] for row in rows])
    oof = np.full(len(rows), np.nan, dtype=np.float64)
    oof_fold = np.full(len(rows), -1, dtype=np.int8)
    reports = []
    for fold in folds:
        train = np.flatnonzero(np.isin(family, fold["train_families"]))
        validation = np.flatnonzero(np.isin(family, fold["validation_families"]))
        test = np.flatnonzero(np.isin(family, fold["test_families"]))
        assert len(train) and len(validation) and len(test)
        assert not set(train) & set(test) and not set(validation) & set(test)
        train_rows, validation_rows, test_rows = [[rows[i] for i in indices] for indices in (train, validation, test)]
        fold_metadata = output / f"fold_{fold['fold']}.json"
        fold_model = output / f"fold_{fold['fold']}.joblib"
        if fold_metadata.exists():
            saved = json.loads(fold_metadata.read_text())
            assert saved["signature"] == signature
            package = joblib.load(fold_model)
            assert package["signature"] == signature and package["split"] == fold
            predicted = package["classifier"].decision_function(package["scaler"].transform(matrix[test]))
            oof[test], oof_fold[test] = predicted, fold["fold"]
            reports.append(saved["result"])
            continue
        train_weights = weights(train_rows)
        scaler = StandardScaler().fit(matrix[train], sample_weight=train_weights)
        x_train, x_validation, x_test = [scaler.transform(matrix[indices]) for indices in (train, validation, test)]
        best = None
        choices = []
        for regularization in CONFIG["C_grid"]:
            classifier = LogisticRegression(C=regularization, penalty="l2", solver=CONFIG["solver"],
                max_iter=CONFIG["max_iter"], tol=CONFIG["tol"], random_state=20261112)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                classifier.fit(x_train, labels[train], sample_weight=train_weights)
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                classifier.set_params(max_iter=CONFIG["retry_max_iter"], warm_start=True)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    classifier.fit(x_train, labels[train], sample_weight=train_weights)
            converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            assert converged, f"Probe did not converge: {source}/{view}/{model}/{site}/{fold['fold']}/{regularization}"
            score = float(roc_auc_score(labels[validation], classifier.decision_function(x_validation),
                                       sample_weight=weights(validation_rows)))
            choices.append({"C": regularization, "validation_AUROC": score,
                            "iterations": int(classifier.n_iter_.max()), "converged": converged})
            if best is None or score > best[0] + 1e-12:
                best = score, classifier
        validation_score, classifier = best
        predicted = classifier.decision_function(x_test)
        assert np.isnan(oof[test]).all()
        oof[test], oof_fold[test] = predicted, fold["fold"]
        fold_result = {"fold": fold["fold"], "C": classifier.C, "validation_AUROC": validation_score,
            "train_samples": len(train), "validation_samples": len(validation), "test_samples": len(test),
            "train_episodes": len({row["episode_id"] for row in train_rows}),
            "train_weight_sum": float(train_weights.sum()), "candidates": choices,
            "test": metrics(test_rows, labels[test], predicted)}
        reports.append(fold_result)
        temporary = fold_model.with_suffix(".joblib.tmp")
        joblib.dump({"scaler": scaler, "classifier": classifier, "signature": signature, "split": fold,
                     "config": CONFIG}, temporary, compress=3)
        temporary.replace(fold_model)
        write_json(fold_metadata, {"signature": signature, "result": fold_result})
    assert np.isfinite(oof).all() and (oof_fold >= 0).all()
    by_pair = defaultdict(dict)
    for row in rows:
        by_pair[row["pair_group"]][row["label"]] = row["both_success"]
    successful_pairs = {pair for pair, sides in by_pair.items() if set(sides) == {0, 1} and all(sides.values())}
    selected = np.asarray([i for i, row in enumerate(rows) if row["pair_group"] in successful_pairs])
    phase = np.asarray([min(9, int(row["phase"] * 10)) for row in rows])
    phase_results = []
    for bin_index in np.unique(phase):
        indices = np.flatnonzero(phase == bin_index)
        per_fold = []
        for fold_index in range(5):
            selected_fold = indices[oof_fold[indices] == fold_index]
            per_fold.append(metrics([rows[i] for i in selected_fold], labels[selected_fold], oof[selected_fold]))
        valid = [value["AUROC"] for value in per_fold if value["AUROC"] is not None]
        phase_results.append({"phase_bin": int(bin_index),
            "fold_mean_AUROC": float(np.mean(valid)) if valid else None,
            "folds": per_fold, "samples": len(indices)})
    success_folds = []
    for fold_index in range(5):
        selected_fold = selected[oof_fold[selected] == fold_index]
        success_folds.append(metrics([rows[i] for i in selected_fold], labels[selected_fold], oof[selected_fold]))
    timestep_results = []
    timestep = np.asarray([row["timestep"] for row in rows])
    for step in np.unique(timestep):
        indices = np.flatnonzero(timestep == step)
        per_fold = []
        for fold_index in range(5):
            chosen = indices[oof_fold[indices] == fold_index]
            per_fold.append(metrics([rows[i] for i in chosen], labels[chosen], oof[chosen]))
        valid = [value["AUROC"] for value in per_fold if value["AUROC"] is not None]
        timestep_results.append({"timestep": int(step), "samples": len(indices), "valid_folds": len(valid),
            "fold_mean_AUROC": float(np.mean(valid)) if valid else None, "folds": per_fold})
    result = {"complete": True, "signature": signature, "source": source, "view": view,
        "model": model, "site": site, "folds": reports, "config": CONFIG,
        "fold_mean_AUROC": float(np.mean([row["test"]["AUROC"] for row in reports])),
        "fold_sd_AUROC": float(np.std([row["test"]["AUROC"] for row in reports], ddof=1)),
        "fold_mean_balanced_accuracy": float(np.mean([row["test"]["balanced_accuracy"] for row in reports])),
        "oof": metrics(rows, labels, oof),
        "both_successful_pair_subset_oof": metrics([rows[i] for i in selected], labels[selected], oof[selected]),
        "both_successful_pair_subset_folds": success_folds,
        "phase_bin_oof": phase_results, "seconds": time.monotonic() - started,
        "decision_timestep_oof": timestep_results,
        "caution": "Pooled OOF logits come from five probes; fold-mean AUROC is the primary metric."}
    np.savez_compressed(output / "oof_predictions.npz", decision_score=oof, fold=oof_fold, label=labels)
    write_json(output / "metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", choices=["both", "teacher"], required=True)
    parser.add_argument("--view", choices=["initial", "all"], required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--sites", nargs="+")
    parser.add_argument("--feature-root", type=Path, default=RUN / "features")
    parser.add_argument("--output-root", type=Path, default=RUN / "probes")
    args = parser.parse_args()
    sites = args.sites or [name for name, _ in feature_sites()]
    requests = [(str(args.feature_root), str(args.output_root), args.source, args.view, model, site, args.threads)
                for model in args.models for site in sites]
    results = []
    started = time.monotonic()
    output = args.output_root / args.source / args.view
    partial = bool(args.sites) or args.models != list(MODELS)
    prefix = ("partial_" + digest({"sites": sites, "models": args.models})[:12] + "_") if partial else ""
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=8) as pool:
        for result in pool.imap_unordered(one_site, requests):
            results.append(result)
            write_json(output / (prefix + "progress.json"), {"completed_sites": len(results), "total_sites": len(requests),
                "seconds": time.monotonic() - started, "last_model": result["model"], "last_site": result["site"]})
    summary = {"complete": True, "source": args.source, "view": args.view, "sites": len(results),
               "folds_per_site": 5, "seconds": time.monotonic() - started, "VLA_training": False, "LDA": False}
    write_json(output / (prefix + "completion.json"), summary)
    if not partial:
        milestone(f"probes_{args.source}_{args.view}",
                  f"{args.source} {args.view}: 네 모델 Linear probe 5-fold 학습·평가 완료", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        write_json(RUN / "experiment_error.json", {"stage": "linear_probes", "error": traceback.format_exc()})
        from experiment_notifications import notify_progress
        notify_progress(RUN)
        raise
