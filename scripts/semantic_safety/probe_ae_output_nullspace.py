"""Group-held-out component probes and frozen-readout tests for the last AE projection."""

import argparse
import csv
from unittest.mock import patch

import h5py
import joblib
import numpy as np

import contrast_initial_linear as contrast
from ae_output_nullspace_pilot import MODELS, POST, PRE, REFERENCE, RUN, array_sha, maxabs, read
from rollout_common import write_json


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def score(fit, x):
    return fit["classifier"].decision_function(fit["scaler"].transform(x.astype(np.float64)))


def component_features(name, q, w, bias, tolerance):
    with h5py.File(REFERENCE / "features" / (name + ".h5")) as h:
        assert h.attrs["complete"] and h["written"][:].all()
        full, post = h[PRE][:].astype(np.float64), h[POST][:].astype(np.float64)
        sig = contrast.digest([str(h.attrs["signature"]), array_sha(w), array_sha(q), "raw_fp64_row_and_null_v1"])
    out = (full @ q) @ q.T
    null = full - out
    numerical = dict(reconstruction_max_abs=maxabs(full - (out + null)), null_times_W_max_abs=maxabs(null @ w.T),
                     out_minus_full_velocity_max_abs=maxabs(out @ w.T - full @ w.T),
                     archived_post_minus_effective_projection_max_abs=maxabs(post - (full @ w.T + bias)))
    assert max(numerical[k] for k in ("reconstruction_max_abs", "null_times_W_max_abs",
                                      "out_minus_full_velocity_max_abs")) <= tolerance["algebra_absolute"]
    budget = tolerance["native_floor"] + tolerance["native_relative"] * np.maximum(1, np.abs(post).max(axis=1))
    numerical["archived_projection_within_budget"] = bool(np.all(np.max(np.abs(post - (full @ w.T+bias)), axis=1) <= budget))
    path = RUN / "features" / (name + ".h5")
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        with h5py.File(path) as h:
            assert h.attrs["signature"] == sig and h.attrs["complete"]
            np.testing.assert_array_equal(h["h_out"][:], out)
            np.testing.assert_array_equal(h["h_null"][:], null)
    else:
        tmp = path.with_suffix(".tmp")
        with h5py.File(tmp, "w") as h:
            h["h_out"] = out
            h["h_null"] = null
            h.attrs.update(signature=sig, complete=True)
        tmp.replace(path)
    return full, out, null, numerical


def analyze(name):
    protocol, manifest = read(RUN / "protocol.json"), read(RUN / "manifest.json")
    live = RUN / "intervention" / name
    audit = read(live / "numerics.json")
    with np.load(live / "readout.npz") as z:
        q, w, bias = z["Q"], z["W"], z["bias"]
        full_panel, out_panel, panel_indices = z["panel_full"], z["panel_out"], z["global_indices"]
    assert array_sha(w) == audit["effective_W_sha256"]
    full, out, null, numerics = component_features(name, q, w, bias, protocol["tolerance"])
    assert full.shape == (457, 1024)
    folder = RUN / "analysis" / name
    fold_records, sample_records, summaries, fixed_checks, panel_records = [], [], [], [], []
    for track in ("A", "B"):
        indices = np.array([i for i, r in enumerate(manifest["rows"]) if r["track"] == track])
        rows = [manifest["rows"][i] for i in indices]
        groups = np.array([r["group_id"] for r in rows])
        labels = np.array([r["label"] for r in rows])
        folds = np.full(len(rows), -1, dtype=int)
        for f in manifest["folds"]:
            mask = np.isin(groups, f["test_groups"])
            assert np.all(folds[mask] == -1)
            folds[mask] = f["fold"]
        assert (folds >= 0).all() and len(rows) == protocol["tracks"][track]
        scores, chosen = {}, {}
        for label, site in (("full", PRE), ("row_refit", "h_out"), ("null_refit", "h_null"), ("post", POST)):
            if site in ("h_out", "h_null"):
                with patch.object(contrast, "RUN", RUN):
                    result = contrast.probe_job((name, track, site))
                source = RUN
            else:
                source = REFERENCE
                result = read(source / "probes" / track / name / site / "metrics.json")
            location = source / "probes" / track / name / site
            with np.load(location / "oof_predictions.npz") as z:
                for key, expected in (("label", labels), ("fold", folds), ("global_indices", indices)):
                    np.testing.assert_array_equal(z[key], expected)
                scores[label] = z["decision_score"]
            chosen[label] = [f["C"] for f in result["folds"]]
        scores["row_fixed"] = np.empty(len(rows))
        chosen["row_fixed"] = chosen["full"]
        full_folder = REFERENCE / "probes" / track / name / PRE
        for f in manifest["folds"]:
            fold = f["fold"]
            test = folds == fold
            gi = indices[test]
            fit = joblib.load(full_folder / f"fold_{fold}.joblib")
            assert fit["split"] == f and fit["training_rows"] == [r["item_id"] for r in rows if r["group_id"] in f["train_groups"]]
            np.testing.assert_allclose(score(fit, full[gi]), scores["full"][test], rtol=1e-9, atol=1e-9)
            fixed_out = score(fit, out[gi])
            scores["row_fixed"][test] = fixed_out
            delta = fixed_out - scores["full"][test]
            raw_coef = fit["classifier"].coef_[0] / fit["scaler"].scale_
            np.testing.assert_allclose(delta, -null[gi] @ raw_coef, rtol=1e-9, atol=1e-9)
            fixed_checks.append(dict(track=track, fold=fold, replay_max_abs=maxabs(score(fit, full[gi])-scores["full"][test]),
                                     logit_identity_max_abs=maxabs(delta+null[gi] @ raw_coef)))
        for label, logits in scores.items():
            metrics = []
            for f in manifest["folds"]:
                k = f["fold"]
                test = np.flatnonzero(folds == k)
                value = contrast.metrics([rows[i] for i in test], logits[test])
                metrics.append(value)
                fold_records.append(dict(model=name, track=track, condition=label, fold=k, C=chosen[label][k],
                                         n=len(test), AUROC=value["AUROC"], BA=value["balanced_accuracy"],
                                         TN=value["confusion_matrix"][0][0], FP=value["confusion_matrix"][0][1],
                                         FN=value["confusion_matrix"][1][0], TP=value["confusion_matrix"][1][1]))
            oof = contrast.metrics(rows, logits)
            (tn, fp), (fn, tp) = oof["confusion_matrix"]
            summaries.append(dict(model=name, track=track, condition=label, n=len(rows),
                AUROC=np.mean([x["AUROC"] for x in metrics]), AUROC_SD=np.std([x["AUROC"] for x in metrics], ddof=1),
                BA=np.mean([x["balanced_accuracy"] for x in metrics]), BA_SD=np.std([x["balanced_accuracy"] for x in metrics], ddof=1),
                TN=tn, FP=fp, FN=fn, TP=tp, OOF_accuracy=oof["accuracy"], predicted_unsafe=fp+tp))
        for i, row in enumerate(rows):
            full_score, out_score = scores["full"][i], scores["row_fixed"][i]
            sample_records.append(dict(model=name, track=track, item_id=row["item_id"], group=row["group_id"],
                global_index=int(indices[i]), fold=int(folds[i]), label=row["label"], **{k: v[i] for k, v in scores.items()},
                fixed_logit_delta=out_score-full_score, safe_to_unsafe=int(full_score < 0 <= out_score),
                unsafe_to_safe=int(full_score >= 0 > out_score)))
        k = protocol["intervention_fold"]
        fit = joblib.load(full_folder / f"fold_{k}.joblib")
        for j, i in enumerate(panel_indices):
            row = manifest["rows"][i]
            if row["track"] != track:
                continue
            assert row["group_id"] in fit["split"]["validation_groups"]
            before = score(fit, full_panel[j:j+1])[0]
            after = score(fit, out_panel[j:j+1])[0]
            panel_records.append(dict(model=name, track=track, item_id=row["item_id"], label=row["label"],
                validation_fold=k, original_logit=before, removed_logit=after, delta=after-before,
                flipped=int((before>=0) != (after>=0)),
                live_minus_archived_logit=before-score(fit,full[i:i+1])[0],
                live_minus_offline_removed_logit=after-score(fit,out[i:i+1])[0]))
    assert len(sample_records) == 457 and len(fold_records) == 50 and len(panel_records) == 12
    write_csv(folder / "fold_metrics.csv", fold_records)
    write_csv(folder / "predictions.csv", sample_records)
    write_csv(folder / "summary.csv", summaries)
    write_csv(folder / "intervention_probe_scores.csv", panel_records)
    write_json(folder / "audit.json", dict(model=name, feature_checks=numerics, fixed_probe_checks=fixed_checks))
    print(name, "OOF and validation-panel readouts complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    analyze(parser.parse_args().model)
