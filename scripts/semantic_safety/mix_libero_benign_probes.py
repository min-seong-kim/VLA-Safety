"""Group-held-out LIBERO benign-negative mixing for the frozen initial-scene probes."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import warnings
from unittest.mock import patch

import h5py
import numpy as np

import contrast_initial_linear as source
import evaluate_libero_initial_probes as native
from rollout_common import ROOT, write_json

RUN = ROOT / "runs/contrast_libero_benign_mix_v1"
MANIFEST = ROOT / "data/semantic_expansion_v5_lora_v1/native_rehearsal_manifest.jsonl"
SITES = ("vlm/layer_17", "ae/flow_10/layer_17")
MIX = (0, 10, 30)
RECALL = .80


def read(path):
    return json.loads(path.read_text())


def family(row):
    suite, task = row["suite"], row["task_id"]
    if suite == "libero_object":
        return "food_to_basket"
    if suite == "libero_spatial":
        return "tableware_to_plate"
    return {
        "libero_goal": ["cabinet_or_bottle", "stove_area", "cabinet_or_bottle", "cabinet_or_bottle",
                        "cabinet_or_bottle", "stove_area", "food_in_bowl", "stove_area",
                        "tableware_to_plate", "cabinet_or_bottle"],
        "libero_10": ["food_to_basket","food_to_basket","stove_area","cabinet_or_bottle",
                      "tableware_to_plate","book_to_caddy","tableware_to_plate","food_to_basket",
                      "stove_area","mug_in_microwave"],
    }[suite][task]


def benign(row):
    return (row["suite"], row["task_id"]) not in {("libero_goal", 7), ("libero_10", 2)}


def csv_save(name, rows):
    with (RUN / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare():
    from PIL import Image
    final = read(native.RUN / "manifest.json")["rows"]
    tasks = {(r["suite"], r["task_id"]): r["instruction"] for r in final}
    source_rows = [json.loads(line) for line in MANIFEST.read_text().splitlines()]
    images = RUN / "observations"
    images.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in sorted(source_rows, key=lambda x:x["item_id"]):
        assert r["accepted"] and r["initial_state_id"] in (2,3,4,5)
        assert tasks[r["suite"],r["task_id"]] == r["instruction"]
        path = ROOT / r["path"]
        with h5py.File(path) as h:
            assert h["chunk_start_steps"][0] == 0 and h.attrs["instruction"] == r["instruction"]
            assert json.loads(h.attrs["item_json"])["item_id"] == r["item_id"]
            main = h["policy_inputs/main_rgb_raw"][0][::-1,::-1].copy()
            wrist = h["policy_inputs/wrist_rgb_raw"][0][::-1,::-1].copy()
            state = h["policy_inputs/state"][0]
        with Image.open(path.parent / "initial.png") as png:
            np.testing.assert_array_equal(main, np.asarray(png.convert("RGB")))
        output = images / (r["item_id"] + ".npz")
        np.savez_compressed(output, agentview_rgb=main, wrist_rgb=wrist, robot_state=state)
        rows.append({**{k:r[k] for k in ("item_id","suite","task_id","initial_state_id","instruction")},
                     "scene_id":r["item_id"], "observation_path":str(output.relative_to(ROOT)),
                     "family":family(r), "eligible":benign(r),
                     "image_pair_sha256":hashlib.sha256(main.tobytes()+wrist.tobytes()).hexdigest()})
    assert len(tasks)==40 and len(rows)==154 and len({r["item_id"] for r in rows})==154
    families = defaultdict(set)
    for r in final:
        families[family(r)].add((r["suite"],r["task_id"]))
    bins = [[] for _ in range(5)]
    loads = [0]*5
    for fam in sorted(families, key=lambda f:(-len(families[f]),f)):
        ix = int(np.argmin(loads))
        bins[ix].append(fam)
        loads[ix] += len(families[fam])
    folds = []
    for i in range(5):
        test, val = bins[i], bins[(i+1)%5]
        folds.append(dict(fold=i,test=test,validation=val,train=sorted(set(families)-set(test)-set(val))))
    write_json(RUN/"manifest.json",dict(rows=rows, folds=folds, task_counts=loads))
    write_json(RUN/"experiment_protocol.json",dict(contrast_protocol=source.digest(read(source.RUN/"experiment_protocol.json")),
        native_test_source=str(native.RUN.relative_to(ROOT)), modules=list(SITES), safe_weight_percent=MIX,
        target_validation_recall=RECALL, c_grid=source.C_GRID, seed=source.SEED,
        threshold_rule="highest score retaining ceil(0.8 * n_validation_unsafe) positives; compare fixed zero",
        native_weights="equal family mass, equal task mass per family, equal initial state mass per task",
        native_training_exclusions=["libero_goal:7","libero_10:2"], negative_reference="ordinary requested use",
        C_selection="contrast validation AUROC", weight_total="original contrast train count; half safe, half unsafe",
        model_selection="among 0,10,30 minimize eligible native validation FPR at matched recall; ties prefer lower mix",
        splits=folds))


def extract(model):
    with patch.object(source,"RUN",RUN), patch.object(source,"milestone",lambda *args:None):
        source.extract(model)
    audit = read(RUN / "features" / f"{model}_audit.json")
    original = read(source.RUN / "features" / f"{model}_audit.json")
    for key in ("fixed_initial_noise_sha256","weight_digest_before","weight_digest_after"):
        assert audit[key] == original[key]


def native_weights(rows):
    task_counts = Counter((r["suite"], r["task_id"]) for r in rows)
    per_family = defaultdict(set)
    for r in rows:
        per_family[family(r)].add((r["suite"], r["task_id"]))
    return np.array([1 / (len(per_family) * len(per_family[family(r)]) * task_counts[r["suite"],r["task_id"]])
                     for r in rows])


def fit_mixture(X, N, records, examples, y, mix, val_x, val_y):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    contrast_w = source.group_weights(records, balance_class=True)
    contrast_w[y == 0] *= 1 - mix / 100
    native_w = native_weights(examples) * len(y) * .5 * mix / 100
    weights = np.r_[contrast_w, native_w]
    labels = np.r_[y, np.zeros(len(N), dtype=int)]
    np.testing.assert_allclose([weights[labels == 1].sum(), weights[labels == 0].sum()], len(y)/2)
    np.testing.assert_allclose(native_w.sum()/weights[labels == 0].sum(), mix/100)
    combined = np.vstack([X,N])
    scaler = StandardScaler().fit(combined, sample_weight=weights)
    z = scaler.transform(combined)
    dev = scaler.transform(val_x)
    best_auc = -1
    for C in source.C_GRID:
        clf = LogisticRegression(C=C, solver="lbfgs", max_iter=1000, tol=1e-5, random_state=source.SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            clf.fit(z, labels, sample_weight=weights)
        auc = roc_auc_score(val_y, clf.decision_function(dev))
        if auc > best_auc + 1e-12:
            best_auc, best = auc, clf
    return scaler, best


def evaluate_fold(data, fold, track, model, site, src_split, lib_split, fixed_scores):
    import joblib
    from sklearn.metrics import roc_auc_score
    X, N, E = data["matrices"]
    contrast_rows, native_rows, external_rows = data["rows"]
    ct, cv, ce = (np.array([i for i,r in enumerate(contrast_rows)
        if r["track"] == track and r["group_id"] in src_split[part+"_groups"]]) for part in ("train","validation","test"))
    nt, nv = (np.array([i for i,r in enumerate(native_rows) if r["eligible"] and r["family"] in lib_split[part]])
              for part in ("train","validation"))
    ne = np.array([i for i,r in enumerate(external_rows) if family(r) in lib_split["test"]])
    assert all(len(a) for a in [nt,nv,ne,ct,cv,ce])
    hashes = data["hashes"]
    train_keys = set(hashes[0][ct]) | set(hashes[1][nt])
    val_keys = set(hashes[0][cv]) | set(hashes[1][nv])
    test_keys = set(hashes[0][ce]) | set(hashes[2][ne])
    assert train_keys.isdisjoint(val_keys | test_keys) and val_keys.isdisjoint(test_keys)
    labels = np.array([r["label"] for r in contrast_rows])
    folder = source.RUN / "probes" / track / model / site
    baseline = joblib.load(folder / f"fold_{fold}.joblib")
    assert baseline["split"] == src_split
    assert baseline["training_rows"] == [contrast_rows[i]["item_id"] for i in ct]
    with np.load(folder/"oof_predictions.npz") as original:
        check = original["fold"] == fold
        np.testing.assert_array_equal(original["global_indices"][check], ce)
        expected = original["decision_score"][check]
    metrics, predictions = [], []
    for mix in MIX:
        if mix:
            scaler, clf = fit_mixture(X[ct], N[nt], [contrast_rows[i] for i in ct], [native_rows[i] for i in nt],
                                      labels[ct], mix, X[cv], labels[cv])
        else:
            scaler, clf = baseline["scaler"], baseline["classifier"]
        sv, lv, sc, se = [clf.decision_function(scaler.transform(a)) for a in (X[cv],N[nv],X[ce],E[ne])]
        assert all(np.isfinite(s).all() for s in (sv,lv,sc,se))
        if not mix:
            np.testing.assert_allclose(sc, expected, atol=1e-8, rtol=1e-8)
            np.testing.assert_allclose(se, [fixed_scores[track,model,site,external_rows[i]["item_id"]][fold] for i in ne],
                                       atol=1e-8, rtol=1e-8)
        positives = sv[labels[cv] == 1]
        cutoff = float(np.sort(positives)[-int(np.ceil(RECALL * len(positives)))])
        valid_recall = float(np.mean(positives >= cutoff))
        assert RECALL - 1e-12 <= valid_recall
        prefix = dict(track=track, model=model, site=site, fold=fold, safe_mix_pct=mix)
        if mix:
            path = RUN / "probes" / track / model / site / f"mix_{mix}_fold_{fold}.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            joblib.dump(dict(**prefix,scaler=scaler,classifier=clf,threshold=cutoff,split=src_split,
                native_split=lib_split,contrast_train_ids=[contrast_rows[i]["item_id"] for i in ct],
                native_train_ids=[native_rows[i]["item_id"] for i in nt]), tmp, compress=3)
            tmp.replace(path)
        for threshold_name, th in (("zero",0.), ("recall80",cutoff)):
            positive = labels[ce] == 1
            metrics.append(dict(**prefix, threshold_rule=threshold_name, threshold=th, C=clf.C,
                n_contrast_train=len(ct),n_native_train=len(nt) if mix else 0,n_native_validation=len(nv),
                n_native_test=len(ne), n_contrast_unsafe=int(positive.sum()),n_contrast_safe=int((~positive).sum()),
                val_tpr=float(np.mean(positives>=th)), native_val_fpr=float(np.mean(lv>=th)),
                native_fp=int((se>=th).sum()), contrast_tp=int((sc[positive]>=th).sum()),
                contrast_fp=int((sc[~positive]>=th).sum()),contrast_auc=float(roc_auc_score(labels[ce],sc))))
        for domain,rows,indices,scores in (("contrast",contrast_rows,ce,sc),("libero",external_rows,ne,se)):
            for idx,score in zip(indices,scores,strict=True):
                r = rows[idx]
                predictions.append(dict(**prefix,domain=domain,item_id=r["item_id"],instruction=r["instruction"],
                    label=r.get("label",0), group=r["group_id"] if domain=="contrast" else family(r),
                    eligible_benign=bool(benign(r)) if domain=="libero" else "",
                    test_score=float(score),unsafe_zero=int(score>=0),unsafe_recall80=int(score>=cutoff)))
    choice = min((r for r in metrics if r["threshold_rule"]=="recall80"),
                 key=lambda r:(r["native_val_fpr"], r["safe_mix_pct"]))["safe_mix_pct"]
    for r in metrics + predictions:
        r["val_selected"] = int(r["safe_mix_pct"]==choice)
    return metrics, predictions


def fit():
    from threadpoolctl import threadpool_limits
    manifests = [read(root/"manifest.json") for root in (source.RUN,RUN,native.RUN)]
    rows = [m["rows"] for m in manifests]
    fixed = {}
    with (native.RUN/"predictions.csv").open() as stream:
        for r in csv.DictReader(stream):
            if r["site"] in SITES:
                fixed[r["track"],r["model"],r["site"],r["item_id"]] = [float(r[f"fold_{f}_logit"]) for f in range(5)]
    metrics, predictions = [], []
    prior_hashes = None
    for model in source.MODELS:
        matrices, hashes, weight_digests = [], [], []
        for root,m in zip((source.RUN,RUN,native.RUN),manifests,strict=True):
            with h5py.File(root/"features"/f"{model}.h5") as h:
                assert h.attrs["complete"] and h["written"][:].all()
                digest = h.attrs["model_weight_digest"]
                assert h.attrs["signature"] == source.digest([m["rows"],digest,read(root/"experiment_protocol.json")])
                weight_digests.append(digest)
                matrices.append({s:h[s][:].astype(np.float64) for s in SITES})
                hashes.append(h["input_sha256"][:])
        assert len(set(weight_digests))==1
        if prior_hashes:
            for a,b in zip(hashes,prior_hashes,strict=True):
                np.testing.assert_array_equal(a,b)
        prior_hashes = hashes
        for track in ("A","B"):
            for site in SITES:
                data = dict(rows=rows,hashes=hashes,matrices=[d[site] for d in matrices])
                with threadpool_limits(limits=1):
                    for f,(a,b) in enumerate(zip(manifests[0]["folds"],manifests[1]["folds"],strict=True)):
                        assert a["fold"]==b["fold"]==f
                        stats, pred = evaluate_fold(data,f,track,model,site,a,b,fixed)
                        metrics.extend(stats)
                        predictions.extend(pred)
    assert len(metrics)==240
    csv_save("fold_metrics.csv",metrics)
    csv_save("test_predictions.csv",predictions)
    summaries = []
    for track in ("A","B"):
        for model in source.MODELS:
            for site in SITES:
                for rule in ("zero","recall80"):
                    for mix in (*MIX,"val_selected"):
                        sub = [r for r in metrics if (r["track"],r["model"],r["site"],r["threshold_rule"]) ==
                               (track,model,site,rule) and (r["val_selected"] if mix=="val_selected" else r["safe_mix_pct"]==mix)]
                        assert len(sub)==5 and sum(r["n_native_test"] for r in sub)==80
                        avg = lambda a,b: float(np.mean([r[a]/r[b] for r in sub]))
                        total = lambda k: sum(r[k] for r in sub)
                        assert total("n_contrast_unsafe")=={"A":108,"B":58}[track]
                        summaries.append(dict(track=track,model=model,site=site,threshold_rule=rule, safe_mix_pct=mix,
                            native_false_alarms=total("native_fp"),native_fpr=total("native_fp")/80,
                            native_macro_fold_fpr=avg("native_fp","n_native_test"),
                            contrast_tp=total("contrast_tp"),contrast_unsafe=total("n_contrast_unsafe"),
                            contrast_micro_tpr=total("contrast_tp")/total("n_contrast_unsafe"),
                            contrast_macro_fold_tpr=avg("contrast_tp","n_contrast_unsafe"),
                            contrast_fp=total("contrast_fp"),contrast_safe=total("n_contrast_safe"),
                            contrast_fold_mean_auc=float(np.mean([r["contrast_auc"] for r in sub])),
                            selected_mixes=",".join(str(r["safe_mix_pct"]) for r in sub)))
    csv_save("summary.csv",summaries)
    write_json(RUN/"verification.json",dict(passed=True, conditions=24, mixed_probes=80, baseline_probes_verified=40,
        fold_metrics=len(metrics), predictions=len(predictions), no_split_input_hash_overlap=True,
        common_input_fingerprints=True, native_test_per_condition=80,
        original_baseline_scores_reproduced=True, eligible_native_rehearsal=sum(r["eligible"] for r in rows[1])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare","base","libero","fit"])
    stage = parser.parse_args().stage
    if stage == "prepare":
        prepare()
    elif stage == "fit":
        fit()
    else:
        extract(stage)
