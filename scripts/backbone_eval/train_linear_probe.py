"""Train one linear probe per (model, dataset) on cached last-token features; evaluate on the same held-out test ids.

StandardScaler (fit on train only) + L2 LogisticRegression (lbfgs, max_iter 2000). C in {0.01, 0.1, 1, 10} chosen on dev
(binary: balanced accuracy; MC: accuracy; ties -> smaller C). MC: one classifier over the maximal number of option
letters; at prediction time classes beyond an item's n_choices are masked out. The backbone is never touched here
(features are read from .npz files).

Writes runs/backbone_eval_v1/probes/<model>/<dataset>/{probe.joblib, metrics.json} and
       runs/backbone_eval_v1/predictions/<model>/<dataset>.jsonl (likelihood + probe predictions on identical test ids).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATASETS, LETTERS, RUN_DIR, dump_json  # noqa: E402

C_GRID = [0.01, 0.1, 1.0, 10.0]


def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def masked_argmax(proba: np.ndarray, classes: np.ndarray, n_choices: np.ndarray) -> np.ndarray:
    """proba [n, len(classes)] -> predicted class id, only among classes < n_choices[i]."""
    pred = np.zeros(len(proba), np.int32)
    for i in range(len(proba)):
        valid = classes < n_choices[i]
        p = np.where(valid, proba[i], -np.inf)
        pred[i] = classes[int(np.argmax(p))]
    return pred


def metrics(y, pred, task_type, n_choices=None):
    m = {"acc": float(np.mean(pred == y)), "n": int(len(y))}
    if task_type == "binary":
        m["balanced_acc"] = float(balanced_accuracy_score(y, pred)) if len(set(y)) > 1 else None
        m["macro_f1"] = float(f1_score(y, pred, average="macro")) if len(set(y)) > 1 else None
        m["f1_yes"] = float(f1_score(y, pred, pos_label=1)) if 1 in set(y) else None
        m["yes_rate"] = float(np.mean(pred == 1))
        # safety-style auxiliaries (yes = unsafe for the safety sets; harmless for POPE/MME)
        m["recall_yes"] = float(np.mean(pred[y == 1] == 1)) if (y == 1).any() else None
        m["fpr_no"] = float(np.mean(pred[y == 0] == 1)) if (y == 0).any() else None
    else:
        m["chance"] = float(np.mean(1.0 / n_choices))
        m["n_by_k"] = {int(k): int(v) for k, v in zip(*np.unique(n_choices, return_counts=True), strict=True)}
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--run_dir", default=str(RUN_DIR))
    ap.add_argument("--max_iter", type=int, default=2000)
    ap.add_argument("--feature_run_dir", help="read cached features from another run; write results only under --run_dir")
    args = ap.parse_args()
    run = pathlib.Path(args.run_dir)
    feature_run = pathlib.Path(args.feature_run_dir) if args.feature_run_dir else run
    for model in args.models:
        for ds in args.datasets:
            fdir = feature_run / "features" / model / ds
            if not (fdir / "train.npz").exists() or not (fdir / "test.npz").exists():
                continue
            task_type = DATASETS[ds][1]
            tr, dv, te = load(fdir / "train.npz"), load(fdir / "dev.npz") if (fdir / "dev.npz").exists() else None, load(fdir / "test.npz")
            out = run / "probes" / model / ds
            out.mkdir(parents=True, exist_ok=True)
            rep = {"model": model, "dataset": ds, "task_type": task_type, "n_train": int(len(tr["label"])),
                   "feature_run_dir": str(feature_run),
                   "n_dev": int(len(dv["label"])) if dv else 0, "n_test": int(len(te["label"])),
                   "train_label_dist": {int(k): int(v) for k, v in zip(*np.unique(tr["label"], return_counts=True), strict=True)}}
            classes_needed = np.arange(2) if task_type == "binary" else np.arange(int(max(tr["n_choices"].max(), te["n_choices"].max())))
            if len(np.unique(tr["label"])) < 2:
                rep["status"] = "held: single-class train split"
                dump_json(out / "metrics.json", rep)
                continue
            missing = sorted(set(classes_needed.tolist()) - set(np.unique(tr["label"]).tolist()))
            rep["classes_missing_in_train"] = missing
            scaler = StandardScaler().fit(tr["h"])
            Xtr, Xte = scaler.transform(tr["h"]), scaler.transform(te["h"])
            Xdv = scaler.transform(dv["h"]) if dv else None
            best = None
            rep["dev_selection"] = []
            for C in C_GRID:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", ConvergenceWarning)
                    clf = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=args.max_iter, n_jobs=1)
                    clf.fit(Xtr, tr["label"])
                    converged = (not any(issubclass(x.category, ConvergenceWarning) for x in w)
                                 and int(np.max(clf.n_iter_)) < args.max_iter)
                if dv is not None and len(dv["label"]) > 0:
                    proba = clf.predict_proba(Xdv)
                    pred = masked_argmax(proba, clf.classes_, dv["n_choices"]) if task_type == "mc" else (proba[:, list(clf.classes_).index(1)] >= 0.5).astype(int)
                    score = balanced_accuracy_score(dv["label"], pred) if task_type == "binary" and len(set(dv["label"])) > 1 else np.mean(pred == dv["label"])
                else:
                    score = float("nan")
                rep["dev_selection"].append({"C": C, "dev_score": float(score), "converged": converged, "n_iter": int(np.max(clf.n_iter_))})
                if best is None or (score > best[0] + 1e-12):
                    best = (score, C, clf, converged)
            score, C, clf, converged = best
            rep.update(selected_C=C, dev_score=float(score), converged=converged, classes_=clf.classes_.tolist(),
                       selection_metric="balanced_acc" if task_type == "binary" else "acc", max_iter=args.max_iter,
                       solver="lbfgs", n_jobs=1)
            proba = clf.predict_proba(Xte)
            if task_type == "binary":
                pos = list(clf.classes_).index(1)
                pred = (proba[:, pos] >= 0.5).astype(int)
            else:
                pred = masked_argmax(proba, clf.classes_, te["n_choices"])
            rep["test_probe"] = metrics(te["label"], pred, task_type, te["n_choices"])
            rep["test_likelihood"] = metrics(te["label"], te["pred_ll"], task_type, te["n_choices"]) if (te["pred_ll"] >= 0).all() else {"status": "unsupported"}
            maj = int(np.bincount(tr["label"]).argmax())
            rep["test_majority_baseline"] = metrics(te["label"], np.full_like(te["label"], maj), task_type, te["n_choices"])
            # image-shuffle control with the already-trained probe
            sh = fdir / "test_imgshuf.npz"
            if sh.exists():
                s = load(sh)
                Xs = scaler.transform(s["h"])
                ps = clf.predict_proba(Xs)
                pred_s = (ps[:, list(clf.classes_).index(1)] >= 0.5).astype(int) if task_type == "binary" else masked_argmax(ps, clf.classes_, s["n_choices"])
                # compare with the same items' normal-image predictions
                idx = {i: k for k, i in enumerate(te["item_id"])}
                sel = np.array([idx[i] for i in s["item_id"]])
                rep["imgshuf"] = {"n": int(len(s["label"])), "probe_acc": float(np.mean(pred_s == s["label"])),
                                  "probe_acc_same_items_normal": float(np.mean(pred[sel] == te["label"][sel])),
                                  "ll_acc": float(np.mean(s["pred_ll"] == s["label"])) if (s["pred_ll"] >= 0).all() else None,
                                  "ll_acc_same_items_normal": float(np.mean(te["pred_ll"][sel] == te["label"][sel])) if (te["pred_ll"] >= 0).all() else None}
            joblib.dump({"scaler": scaler, "clf": clf, "C": C, "classes": clf.classes_.tolist()}, out / "probe.joblib")
            rep["status"] = "ok"
            dump_json(out / "metrics.json", rep)
            pdir = run / "predictions" / model
            pdir.mkdir(parents=True, exist_ok=True)
            with (pdir / f"{ds}.jsonl").open("w") as f:
                for i in range(len(te["label"])):
                    f.write(json.dumps({"item_id": str(te["item_id"][i]), "label": int(te["label"][i]), "pred_ll": int(te["pred_ll"][i]),
                                        "pred_probe": int(pred[i]), "cand_logp": [None if np.isnan(x) else float(x) for x in te["cand_logp"][i]],
                                        "cand_mass": float(te["cand_mass"][i]), "top1_id": int(te["top1_id"][i]),
                                        "margin": float(te["margin"][i]) if np.isfinite(te["margin"][i]) else None,
                                        "probe_proba": [float(x) for x in proba[i]]}) + "\n")
            print(f"[{model}/{ds}] C={C} dev={score:.3f} test probe={rep['test_probe']['acc']:.3f} "
                  f"ll={rep['test_likelihood'].get('acc', float('nan')):.3f} chance={rep['test_probe'].get('chance', 0.5):.3f}", flush=True)


if __name__ == "__main__":
    main()
