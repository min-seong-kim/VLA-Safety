"""Compare family, connected-component, and individual-contrast probe splits."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import h5py
import numpy as np

from experiment_notifications import acknowledge_report, notify_progress, write_json
from representation_common import feature_sites

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/semantic_safety_split_comparison_v1"
OLD = ROOT / "runs/semantic_safety_representation_linear_v1"
NEW = ROOT / "runs/semantic_safety_contrast_initial_linear_v1"
BENIGN = ROOT / "runs/semantic_benign_adaptation_v1"
MODELS = ("base", "libero", "ae", "vlm", "all")
STRATEGIES = ("family", "connected", "contrast")
SEED = 20261116
SCRIPT = Path(__file__).resolve()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def immutable(path, value):
    value = json.loads(json.dumps(value, ensure_ascii=False))
    if path.exists():
        assert json.loads(path.read_text()) == value, str(path)
    else:
        write_json(path, value)


def milestone(key, title, result):
    write_json(RUN / "milestones" / (key + ".json"),
               dict(complete=True, title=title, result=result))
    notify_progress(RUN)


def new_family(set_id):
    exact = {
        "body_knife": "body_sharp_processing", "body_saw": "body_sharp_processing",
        "body_hammer": "body_impact_tools", "body_grater": "body_abrasion",
        "body_sandpaper": "body_abrasion", "body_wirebrush": "body_abrasion",
        "microwave_material": "microwave_compatibility", "candle_target": "flame_contact",
        "hot_water_target": "body_thermal_contact", "socket_object": "electrical_material_choice",
        "battery_terminals": "electrical_material_choice", "cable_power": "electrical_state",
        "mix_vinegar": "chemical_mixing", "mix_ammonia": "chemical_mixing",
    }
    if set_id in exact:
        return exact[set_id]
    prefixes = [
        ("chemical_contact_", "chemical_contact"), ("thermal_contact_", "body_thermal_contact"),
        ("open_pressure_", "stored_energy_state"), ("pressure_", "stored_energy_state"),
        ("battery_", "stored_energy_state"), ("contact_", "body_benign_contact_controls"),
        ("infant_", "body_recipient_access"), ("food_", "food_spoilage"),
        ("foreign_", "food_foreign"), ("chemical_", "chemical_food"),
        ("ecology_", "aquatic"), ("heat_", "direct_heating"),
        ("liquid_", "liquid_electronics"), ("preserve_", "preserve_surfaces_and_originals"),
        ("obstruct_", "obstruct_required_function"),
    ]
    if set_id.startswith("hygiene_"):
        return "hygiene_contact" if set_id in {"hygiene_toyturtle", "hygiene_teether"} else "hygiene_food"
    for prefix, family in prefixes:
        if set_id.startswith(prefix):
            return family
    raise ValueError("Unmapped contrast: " + set_id)


def observation_digest(main, wrist, state):
    h = hashlib.sha256()
    for array in (main, wrist, state):
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def connect(rows):
    units = sorted({s for row in rows for s in row["contrast_ids"]})
    parent = {s: s for s in units}

    def root(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    def union(a, b):
        a, b = root(a), root(b)
        parent[max(a, b)] = min(a, b)

    seen = {}
    for row in rows:
        for s in row["contrast_ids"][1:]:
            union(row["contrast_ids"][0], s)
        for token in ("condition:" + row["condition_id"], "observation:" + row["observation_sha256"]):
            if token in seen:
                union(row["contrast_ids"][0], seen[token])
            seen[token] = row["contrast_ids"][0]
    parts = defaultdict(list)
    for s in units:
        parts[root(s)].append(s)
    ids = {s: "g_" + digest(sorted(parts[root(s)]))[:12] for s in units}
    for row in rows:
        row["group_id"] = ids[row["contrast_ids"][0]]
    return {ids[v[0]]: sorted(v) for v in parts.values()}


def unit_ids(row, strategy):
    return row["family_ids"] if strategy == "family" else (
        row["contrast_ids"] if strategy == "contrast" else [row["group_id"]])


def tokens(row):
    return {"condition:" + row["condition_id"], "observation:" + row["observation_sha256"]}


def materialize(rows, bins, strategy):
    folds = []
    for f in range(5):
        sets = dict(test=set(bins[f]), validation=set(bins[(f + 1) % 5]))
        sets["train"] = set(s for b in bins for s in b) - sets["test"] - sets["validation"]
        candidates = {p: [i for i, row in enumerate(rows) if set(unit_ids(row, strategy)) & selected]
                      for p, selected in sets.items()}
        chosen, excluded = {}, {}
        occupied = set()
        # Preserve the test set; purge observation/condition overlaps upstream.
        for part in ("test", "validation", "train"):
            chosen[part] = [i for i in candidates[part] if not tokens(rows[i]) & occupied]
            excluded[part] = len(candidates[part]) - len(chosen[part])
            occupied.update(t for i in chosen[part] for t in tokens(rows[i]))
        counts = {part: {track: dict(Counter("unsafe" if rows[i]["label"] else "safe"
                                             for i in indices if rows[i]["track"] == track))
                         for track in sorted({r["track"] for r in rows})}
                  for part, indices in chosen.items()}
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            assert not {t for i in chosen[a] for t in tokens(rows[i])} & {
                t for i in chosen[b] for t in tokens(rows[i])}
        folds.append(dict(fold=f, units={p: sorted(v) for p, v in sets.items()}, indices=chosen,
                          excluded_shared_rows=excluded, counts=counts))
    return folds


def viable(folds):
    return all(set(counts) == {"safe", "unsafe"} and min(counts.values()) >= 2
               for fold in folds for tracks in fold["counts"].values() for counts in tracks.values())


def make_bins(rows, strategy):
    units = sorted({s for row in rows for s in unit_ids(row, strategy)})
    tracks = sorted({r["track"] for r in rows})
    columns = [(t, y) for t in tracks for y in [0, 1]]
    counts = np.array([[sum(s in unit_ids(r, strategy) and r["track"] == t and r["label"] == y
                            for r in rows) for t, y in columns] for s in units])
    target = counts.sum(0) / 5
    rng = np.random.default_rng(SEED)
    best = None
    for _ in range(1200):
        loads = np.zeros((5, len(columns)), dtype=int)
        bins = [[] for _ in range(5)]
        order = np.argsort(-(counts / target).sum(1) + rng.normal(0, .8, len(units)))
        for index in order:
            costs = []
            for b in range(5):
                candidate = loads.copy()
                candidate[b] += counts[index]
                costs.append(float(np.square((candidate - target) / target).sum())
                             + .025 * len(bins[b]) + rng.uniform(0, .05))
            b = int(np.argmin(costs))
            loads[b] += counts[index]
            bins[b].append(units[index])
        if (loads < 2).any() or any(not b for b in bins):
            continue
        score = float(np.square((loads - target) / target).sum())
        if best is not None and score >= best[0]:
            continue
        folds = materialize(rows, bins, strategy)
        if viable(folds):
            best = score, bins, folds
    assert best is not None, (strategy, len(units), "No viable purged split")
    return best[1], best[2]


def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    if not (RUN / "reporting/delivery_mode.json").exists():
        write_json(RUN / "reporting/delivery_mode.json", dict(mode="foreground"))
    source_rows = json.loads((OLD / "features/both/initial/rows.json").read_text())
    old_rows = []
    for source in source_rows:
        with h5py.File(ROOT / source["path"]) as data:
            main, wrist = (data["policy_inputs/" + k][0][::-1, ::-1].copy()
                           for k in ("main_rgb_raw", "wrist_rgb_raw"))
            state = data["policy_inputs/state"][0].astype(np.float32)
        obs_hash = observation_digest(main, wrist, state)
        path = RUN / "inputs/legacy" / (obs_hash + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            np.savez_compressed(path, agentview_rgb=main, wrist_rgb=wrist, robot_state=state)
        old_rows.append(dict(item_id=source["episode_id"], condition_id=f"legacy_c{source['condition_index']:03d}",
                             track="initial", label=source["label"], instruction=source["instruction"],
                             contrast_ids=[source["pair_id"]], family_ids=[source["family_id"]],
                             scene_id=obs_hash, observation_path=str(path.relative_to(ROOT)),
                             observation_sha256=obs_hash, source_path=source["path"]))
    current = json.loads((NEW / "manifest.json").read_text())
    new_rows, cached = [], {}
    sets = jsonl(ROOT / "data/semantic_safety_contrast_v1/contrast_sets.jsonl")
    mapping = {s["set_id"]: new_family(s["set_id"]) for s in sets}
    for source in current["rows"]:
        if source["scene_id"] not in cached:
            with np.load(ROOT / source["observation_path"]) as data:
                cached[source["scene_id"]] = observation_digest(
                    data["agentview_rgb"], data["wrist_rgb"], data["robot_state"].astype(np.float32))
        new_rows.append({**source, "contrast_ids": source["set_ids"],
                         "family_ids": sorted({mapping[s] for s in source["set_ids"]}),
                         "observation_sha256": cached[source["scene_id"]]})
    immutable(RUN / "family_mapping.json", dict(
        basis="Previous task/risk family taxonomy, not action verbs",
        extensions=["chemical_mixing", "preserve_surfaces_and_originals"], contrast_to_family=mapping))
    audits = {}
    for name, rows in (("legacy", old_rows), ("contrast", new_rows)):
        folder = RUN / "datasets" / name
        components = connect(rows)
        if name == "contrast":
            assert {k: sorted(v) for k, v in components.items()} == {
                k: sorted(v) for k, v in current["component_sets"].items()}, "Changed original connected groups"
            assert all(a["group_id"] == b["group_id"] for a, b in zip(rows, current["rows"], strict=True))
        folds = {}
        for strategy in STRATEGIES:
            if name == "contrast" and strategy == "connected":
                bins = [f["test_groups"] for f in current["folds"]]
                plan = materialize(rows, bins, strategy)
                for old_f, new_f in zip(current["folds"], plan, strict=True):
                    for p in ("train", "validation", "test"):
                        assert new_f["indices"][p] == [
                            i for i, r in enumerate(rows) if r["group_id"] in old_f[p + "_groups"]]
            elif name == "legacy" and strategy == "family":
                original = json.loads((OLD / "splits.json").read_text())["folds"]
                # Reverse the original test-bin order so the next bin is validation.
                bins = [original[i]["test_families"] for i in [0, 4, 3, 2, 1]]
                plan = materialize(rows, bins, strategy)
            elif name == "legacy" and strategy == "contrast":
                assert all(len(v) == 1 for v in components.values())
                bins = [[components[g][0] for g in block] for block in folds["connected"]["bins"]]
                plan = materialize(rows, bins, strategy)
                assert [f["indices"] for f in plan] == [
                    f["indices"] for f in folds["connected"]["folds"]]
            else:
                bins, plan = make_bins(rows, strategy)
            assert viable(plan), (name, strategy, "Insufficient classes after purge")
            folds[strategy] = dict(bins=bins, folds=plan)
        manifest = dict(rows=rows, component_sets=components, folds=folds)
        immutable(folder / "manifest.json", manifest)
        if name == "contrast":
            (folder / "features").mkdir(parents=True, exist_ok=True)
            for model in MODELS:
                target = folder / "features" / (model + ".h5")
                source = (NEW if model in {"base", "libero"} else BENIGN) / "features" / (model + ".h5")
                if not target.exists():
                    target.symlink_to(source)
        audits[name] = dict(
            rows=len(rows), families=len({s for r in rows for s in r["family_ids"]}),
            contrasts=len({s for r in rows for s in r["contrast_ids"]}), connected_groups=len(components),
            split_counts={s: [f["counts"] for f in v["folds"]] for s, v in folds.items()},
            purged={s: [f["excluded_shared_rows"] for f in v["folds"]] for s, v in folds.items()},
            test_coverage={s: dict(Counter(Counter(i for f in v["folds"] for i in f["indices"]["test"]).values()))
                           for s, v in folds.items()})
    protocol = dict(
        models=list(MODELS), strategies=list(STRATEGIES), datasets=["legacy", "contrast"],
        only_initial_scene=True, VLA_training=False, LDA=False, folds=5, C_grid=[.01, .1, 1.],
        solver="lbfgs", max_iter=1000, tol=1e-5, seed=SEED, threshold_logit=0.,
        pooling="same original Capture; token axis mean only", sites=78, batch_size=12,
        common_readout=dict(action_horizon=10, action_dim=32, discrete_state_input=False,
                            denoising_steps=10, noise_seed=SEED),
        weights="Fixed connected-component IDs across all split strategies; train-only inverse component count, then equal class mass; sum=n_train",
        scaler="train-only with same weights", validation="unweighted AUROC; lower C on ties; no dev refit",
        test="unweighted AUROC and balanced accuracy averaged across five folds",
        purge="test retained, validation purged against test, train purged against both; match semantic condition OR identical two-camera RGB and state; before track separation",
        repeated_tests="Shared conditions may be tested in multiple folds for nonconnected splits; no duplication within a fold; report test appearances and unique coverage separately",
        checkpoint_selection="Existing selected benign AE8k/VLM12k/Both8k, fixed across both datasets",
        family_definition="Old 35 families preserved; new contrasts mapped before fitting by primary task/risk theme to 23 families, including two extensions; this is not verb-only or a strict single-hazard holdout",
        equivalent_partitions="Legacy has 169 contrasts and 169 singleton connected components; use identical folds and reuse the same fitted probes for these two names",
        feature_reuse="Contrast features reused read-only; legacy re-extracted with the same five checkpoints, fixed noise, batch and input readout",
        dataset_changes="None: preserve original instructions, labels, images and old two layouts",
        exclusions=["new policy training", "new environment rollouts", "LDA", "new data conditions"])
    immutable(RUN / "experiment_protocol.json", protocol)
    for name in ("legacy", "contrast"):
        immutable(RUN / "datasets" / name / "experiment_protocol.json", protocol)
    immutable(RUN / "split_audit.json", audits)
    milestone("splits", "두 데이터셋의 세 분할 방식 및 공유 표본 제거 검증 완료", audits)
    print(json.dumps({k: {a: v[a] for a in ["rows", "families", "contrasts", "connected_groups"]}
                      for k, v in audits.items()}), flush=True)


def extract(model, limit):
    from unittest.mock import patch
    import contrast_initial_linear as original
    from openpi.policies import policy_config
    original.RUN = RUN / "datasets/legacy"
    factory = policy_config.create_trained_policy
    adapter = None
    if model in {"ae", "vlm", "all"}:
        selected = json.loads((BENIGN / "frozen_selection.json").read_text())["selected"][model]
        adapter = ROOT / selected["adapter"]
        assert sha(adapter) == selected["adapter_sha256"]

    def create(*args, **kwargs):
        policy = factory(*args, **kwargs)
        if adapter is not None:
            from v6_scoped_lora import load
            load(policy._model, adapter)
            policy._model.requires_grad_(False)
            policy._model.eval()
        return policy

    with patch("openpi.policies.policy_config.create_trained_policy", new=create), patch.object(
            original, "milestone", lambda *args: None):
        original.extract(model, limit)


def train_weights(rows):
    # The weighting unit stays fixed when the partition unit changes.
    counts = Counter(r["group_id"] for r in rows)
    weights = np.asarray([1. / counts[r["group_id"]] for r in rows])
    labels = np.asarray([r["label"] for r in rows])
    for y in (0, 1):
        weights[labels == y] *= .5 / weights[labels == y].sum()
    return weights * len(rows) / weights.sum()


def metric(labels, scores):
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
    predictions = scores >= 0
    return dict(n=len(labels), safe=int((labels == 0).sum()), unsafe=int((labels == 1).sum()),
                AUROC=float(roc_auc_score(labels, scores)),
                balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
                accuracy=float(np.mean(labels == predictions)),
                predicted_unsafe_fraction=float(np.mean(predictions)),
                confusion_matrix=confusion_matrix(labels, predictions, labels=[0, 1]).tolist())


def probe_job(job):
    import joblib
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    name, strategy, track, model, site = job
    folder = RUN / "datasets" / name
    manifest = json.loads((folder / "manifest.json").read_text())
    rows = manifest["rows"]
    plans = manifest["folds"][strategy]["folds"]
    output = RUN / "probes" / name / strategy / track / model / site
    output.mkdir(parents=True, exist_ok=True)
    with (output / "fit.lock").open("a") as lock, threadpool_limits(limits=1):
        fcntl.flock(lock, fcntl.LOCK_EX)
        signature = digest([sha(folder / "manifest.json"), sha(RUN / "experiment_protocol.json"),
                            name, strategy, track, model, site])
        cached = output / "metrics.json"
        if cached.exists():
            result = json.loads(cached.read_text())
            assert result["signature"] == signature
            return result
        if name == "legacy" and strategy == "contrast":
            source_job = (name, "connected", track, model, site)
            result = probe_job(source_job)
            source = RUN / "probes" / name / "connected" / track / model / site
            result = {**result, "signature": signature, "strategy": strategy,
                      "equivalent_partition_reuse": str(source.relative_to(ROOT))}
            for filename in ["heldout_predictions.npz"] + [f"fold_{f}.joblib" for f in range(5)]:
                target = output / filename
                if not target.exists():
                    target.symlink_to(source / filename)
            write_json(output / "metrics.json", result)
            return result
        with h5py.File(folder / "features" / (model + ".h5")) as source:
            assert source.attrs["complete"] and source["written"][:].all()
            matrix = source[site][:].astype(np.float64)
        assert np.isfinite(matrix).all()
        labels = np.asarray([r["label"] for r in rows])
        results, all_labels, all_scores, all_folds, all_indices = [], [], [], [], []
        for plan in plans:
            indices = {p: np.asarray([i for i in ii if rows[i]["track"] == track], dtype=int)
                       for p, ii in plan["indices"].items()}
            train, dev, test = (indices[p] for p in ("train", "validation", "test"))
            package_path = output / f"fold_{plan['fold']}.joblib"
            if package_path.exists():
                package = joblib.load(package_path)
                assert package["signature"] == signature
                assert package["indices"] == {p: v.tolist() for p, v in indices.items()}
                scaler, classifier = package["scaler"], package["classifier"]
                result = package["result"]
            else:
                weights = train_weights([rows[i] for i in train])
                assert np.allclose([weights[labels[train] == y].sum() for y in [0, 1]], len(train) / 2)
                scaler = StandardScaler().fit(matrix[train], sample_weight=weights)
                x_train, x_dev = scaler.transform(matrix[train]), scaler.transform(matrix[dev])
                best, candidates = None, []
                for C in [.01, .1, 1.]:
                    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=1000, tol=1e-5, random_state=SEED)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        clf.fit(x_train, labels[train], sample_weight=weights)
                    assert not any(issubclass(w.category, ConvergenceWarning) for w in caught), (job, C)
                    score = float(roc_auc_score(labels[dev], clf.decision_function(x_dev)))
                    candidates.append(dict(C=C, validation_AUROC=score, iterations=int(clf.n_iter_.max())))
                    if best is None or score > best[0] + 1e-12:
                        best = score, clf
                score, classifier = best
                predicted = classifier.decision_function(scaler.transform(matrix[test]))
                result = dict(fold=plan["fold"], C=classifier.C, validation_AUROC=score,
                              candidates=candidates, train_samples=len(train), validation_samples=len(dev),
                              test=metric(labels[test], predicted),
                              train=metric(labels[train], classifier.decision_function(x_train)))
                package = dict(signature=signature, scaler=scaler, classifier=classifier, result=result,
                               indices={p: v.tolist() for p, v in indices.items()})
                temporary = package_path.with_suffix(".joblib.tmp")
                joblib.dump(package, temporary, compress=3)
                temporary.replace(package_path)
            assert np.isfinite(scaler.scale_).all() and np.isfinite(classifier.coef_).all()
            predicted = classifier.decision_function(scaler.transform(matrix[test]))
            assert np.isfinite(predicted).all()
            results.append(result)
            all_labels.extend(labels[test].tolist())
            all_scores.extend(predicted.tolist())
            all_folds.extend([plan["fold"]] * len(test))
            all_indices.extend(test.tolist())
        result = dict(complete=True, signature=signature, dataset=name, strategy=strategy, track=track,
                      model=model, site=site, folds=results,
                      fold_mean_AUROC=float(np.mean([r["test"]["AUROC"] for r in results])),
                      fold_sd_AUROC=float(np.std([r["test"]["AUROC"] for r in results], ddof=1)),
                      fold_mean_balanced_accuracy=float(np.mean([r["test"]["balanced_accuracy"] for r in results])),
                      test_appearances=len(all_indices), unique_test_inputs=len(set(all_indices)),
                      test_appearance_metrics=metric(np.asarray(all_labels), np.asarray(all_scores)))
        assert set(all_indices) == {i for i, r in enumerate(rows) if r["track"] == track}
        if name == "contrast" and strategy == "connected":
            reference = (NEW if model in {"base", "libero"} else BENIGN) / "probes" / track / model / site
            previous = json.loads((reference / "metrics.json").read_text())
            with np.load(reference / "oof_predictions.npz") as data:
                previous_scores = dict(zip(data["global_indices"].tolist(), data["decision_score"].tolist(), strict=True))
            difference = max(abs(s - previous_scores[i]) for i, s in zip(all_indices, all_scores, strict=True))
            assert difference < 1e-8 and abs(result["fold_mean_AUROC"] - previous["fold_mean_AUROC"]) < 1e-12, (
                job, "Original protocol reproduction failed", difference)
            result["previous_protocol_reproduced"] = True
            result["reference_prediction_max_abs_difference"] = difference
        np.savez_compressed(output / "heldout_predictions.npz", label=all_labels, decision_score=all_scores,
                            fold=all_folds, global_indices=all_indices)
        write_json(output / "metrics.json", result)
        return result


def probes(workers, smoke):
    jobs = []
    for name in ("legacy", "contrast"):
        manifest = json.loads((RUN / "datasets" / name / "manifest.json").read_text())
        for strategy in STRATEGIES:
            for track in sorted({r["track"] for r in manifest["rows"]}):
                for model in MODELS:
                    for site, _ in feature_sites():
                        if smoke and not (name == "contrast" and strategy == "connected"
                                          and site in {"vlm/layer_17", "ae/flow_10/layer_17"}):
                            continue
                        jobs.append((name, strategy, track, model, site))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(probe_job, job) for job in jobs]
        for index, future in enumerate(as_completed(futures)):
            result = future.result()
            write_json(RUN / ("probe_smoke_progress.json" if smoke else "probe_progress.json"),
                       dict(completed=index + 1, total=len(jobs), last=[result[k] for k in (
                           "dataset", "strategy", "track", "model", "site")]))
    if smoke:
        milestone("probe_preflight", "최근 실험의 주요 probe 20조건 수치 재현 검증 완료",
                  dict(conditions=len(jobs), maximum_allowed_prediction_difference=1e-8))
    else:
        milestone("probes", "두 데이터셋·세 분할·다섯 모델의 전체 Linear probe 학습·평가 완료",
                  dict(conditions=len(jobs), selected_fold_classifiers=len(jobs) * 5))


def report():
    import csv
    import html
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = RUN / "report"
    out.mkdir(exist_ok=True)
    numerical = json.loads((RUN / "numerical_audit.json").read_text())
    assert numerical["passed"] and numerical["unique_selected_classifiers"] == 15600
    results = [json.loads(p.read_text()) for p in (RUN / "probes").rglob("metrics.json")]
    assert len(results) == 3510 and all(r["complete"] for r in results)
    lookup = {(r["dataset"], r["strategy"], r["track"], r["model"], r["site"]): r for r in results}
    reproduction = [r for r in results if r["dataset"] == "contrast" and r["strategy"] == "connected"]
    assert len(reproduction) == 780 and all(r["previous_protocol_reproduced"] for r in reproduction)
    for model in MODELS:
        for site, _ in feature_sites():
            a = lookup["legacy", "connected", "initial", model, site]
            b = lookup["legacy", "contrast", "initial", model, site]
            assert a["folds"] == b["folds"] and a["fold_mean_AUROC"] == b["fold_mean_AUROC"]
    for name in ("legacy", "contrast"):
        with h5py.File(RUN / "datasets" / name / "features/libero.h5") as ref:
            for model in MODELS:
                with h5py.File(RUN / "datasets" / name / "features" / (model + ".h5")) as other:
                    assert other.attrs["complete"]
                    assert np.array_equal(ref["input_sha256"][:], other["input_sha256"][:])
                    if model == "ae":
                        for site, _ in feature_sites():
                            if site.startswith("vlm/"):
                                assert np.array_equal(ref[site][:], other[site][:])
    for model in MODELS:
        old_audit = json.loads((RUN / "datasets/legacy/features" / (model + "_audit.json")).read_text())
        owner = NEW if model in {"base", "libero"} else BENIGN
        new_audit = json.loads((owner / "features" / (model + "_audit.json")).read_text())
        assert old_audit["weight_digest_before"] == new_audit["weight_digest_before"], model
        assert old_audit["fixed_initial_noise_sha256"] == new_audit["fixed_initial_noise_sha256"], model
    overlap = json.loads((RUN / "preprocessed_input_overlap_audit.json").read_text())
    assert all(v["actual_preprocessed_input_hash_overlap_across_partitions"] == 0 for v in overlap.values())
    columns = ["dataset", "track", "strategy", "model", "site", "fold_mean_AUROC", "fold_sd_AUROC",
               "fold_mean_balanced_accuracy", "test_appearances", "unique_test_inputs",
               "test_appearance_accuracy", "predicted_unsafe_fraction", "TN", "FP", "FN", "TP"]
    with (out / "all_sites.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for r in results:
            metrics = r["test_appearance_metrics"]
            tn, fp = metrics["confusion_matrix"][0]
            fn, tp = metrics["confusion_matrix"][1]
            values = {**r, "test_appearance_accuracy": metrics["accuracy"],
                      "predicted_unsafe_fraction": metrics["predicted_unsafe_fraction"],
                      "TN": tn, "FP": fp, "FN": fn, "TP": tp}
            writer.writerow({k: values[k] for k in columns})
    primary, table = [], []
    for name, track in (("legacy", "initial"), ("contrast", "A"), ("contrast", "B")):
        for strategy in STRATEGIES:
            for model in MODELS:
                sites = ["vlm/layer_17", "ae/flow_10/layer_17", "velocity_pre/flow_10", "velocity_post/flow_10"]
                rr = [lookup[name, strategy, track, model, s] for s in sites]
                primary.append(dict(dataset=name, track=track, strategy=strategy, model=model,
                                    results={s: r for s, r in zip(sites, rr, strict=True)}))
                values = [name, track, strategy, model] + [
                    f"{r['fold_mean_AUROC']:.3f} ± {r['fold_sd_AUROC']:.3f}" for r in rr]
                table.append("<tr>" + "".join("<td>" + html.escape(v) + "</td>" for v in values) + "</tr>")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for column, (name, track) in enumerate((("legacy", "initial"), ("contrast", "A"), ("contrast", "B"))):
        for row, site in enumerate(("vlm/layer_17", "ae/flow_10/layer_17")):
            ax = axes[row, column]
            for model in MODELS:
                rr = [lookup[name, strategy, track, model, site] for strategy in STRATEGIES]
                ax.errorbar(range(3), [r["fold_mean_AUROC"] for r in rr],
                            yerr=[r["fold_sd_AUROC"] for r in rr], marker="o", capsize=3, label=model)
            ax.set_xticks(range(3), STRATEGIES)
            ax.set_ylim(0, 1.03)
            ax.axhline(.5, color=".6", linestyle="--", linewidth=1)
            ax.set_title(name + " / " + track + " / " + ("VLM17" if row == 0 else "AE17 flow10"))
            ax.set_ylabel("Test AUROC (fold mean ± SD)")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "split_auroc.png", dpi=180)
    plt.close(fig)
    audit = json.loads((RUN / "split_audit.json").read_text())
    headline = []
    for name, track in (("legacy", "initial"), ("contrast", "A"), ("contrast", "B")):
        for site in ("vlm/layer_17", "ae/flow_10/layer_17"):
            values = [name, track, "VLM17" if site.startswith("vlm/") else "AE17 flow10"] + [
                f"{lookup[name, strategy, track, 'libero', site]['fold_mean_AUROC']:.3f}"
                for strategy in STRATEGIES]
            headline.append("<tr>" + "".join("<td>" + html.escape(v) + "</td>" for v in values) + "</tr>")
    def auc(name, strategy, track, site="vlm/layer_17"):
        return lookup[name, strategy, track, "libero", site]["fold_mean_AUROC"]
    interpretation = []
    for name, track in (("legacy", "initial"), ("contrast", "A"), ("contrast", "B")):
        a, b = auc(name, "family", track), auc(name, "connected", track)
        interpretation.append(f"<li>{name}/{track}의 공식 LIBERO VLM17은 family {a:.3f} → "
                              f"연결 group {b:.3f}, AUROC 차이 {b-a:+.3f}였다.</li>")
    same_family_gap = auc("contrast", "family", "A") - auc("legacy", "family", "initial")
    same_connected_gap = auc("contrast", "connected", "A") - auc("legacy", "connected", "initial")
    comparisons = [(name, track, model, site) for name, track in (
        ("legacy", "initial"), ("contrast", "A"), ("contrast", "B")) for model in MODELS
        for site in ("vlm/layer_17", "ae/flow_10/layer_17")]
    lower = sum(lookup[n, "family", t, m, s]["fold_mean_AUROC"] <
                lookup[n, "connected", t, m, s]["fold_mean_AUROC"] for n, t, m, s in comparisons)
    input_audit = {}
    for name in ("legacy", "contrast"):
        rr = json.loads((RUN / "datasets" / name / "manifest.json").read_text())["rows"]
        grouped = defaultdict(list)
        for r in rr:
            grouped[(r["track"], r["observation_sha256"], r["instruction"])].append(r)
        input_audit[name] = [[r["item_id"] for r in vv] for vv in grouped.values()
                            if len({r["label"] for r in vv}) > 1]
    write_json(RUN / "input_identifiability_audit.json", input_audit)
    texts = [
        "<!doctype html><html lang='ko'><meta charset='utf-8'><title>안전 위험 Linear Probe 분할 비교</title>",
        "<style>body{font:16px/1.7 system-ui;max-width:1500px;margin:30px auto;padding:0 24px;color:#243247}"
        "table{border-collapse:collapse;font-size:14px;width:100%;margin:20px 0}td,th{border:1px solid #ccd4db;padding:6px}"
        "th{background:#edf2f6}img{width:100%}pre{white-space:pre-wrap}</style>",
        "<h1>두 안전 위험 데이터셋에서 분할 단위에 따른 Linear Probe 성능 비교</h1>",
        "<p>공식 base·LIBERO FT·최신 benign AE-only/VLM-only/Both 다섯 모델을 고정하고, "
        "첫 장면에서 family·연결 group·개별 contrast 분할을 비교했다. VLA 추가 학습과 환경 rollout은 수행하지 않았다.</p>",
        "<h2>결과 해석</h2><p>같은 데이터와 모델에서도 family 분할의 주요 위치 점수가 연결 group 분할보다 낮았다. "
        f"다섯 모델·세 입력 조건·두 주요 위치의 {len(comparisons)}개 비교 중 {lower}개에서 이 방향이 일치했다. "
        "분할 기준이 성능 격차에 영향을 준다는 관찰이며 통계적 유의성 검정 결과는 아니다.</p><ul>",
        *interpretation, "</ul>",
        f"<p>그러나 같은 family 분할에서도 새 Track A와 기존 데이터의 공식 LIBERO VLM17 차이는 {same_family_gap:+.3f}, "
        f"같은 연결 group 분할에서도 {same_connected_gap:+.3f}였다. 분할 기준만으로 두 데이터셋 간 격차 전체가 설명되지는 않는다.</p>",
        "<p>개별 contrast 분할이 연결 group보다 항상 높지는 않았다. 공유 입력을 train/validation에서 제거했고, "
        "test 구성·반복 출현 수·학습 표본 수가 달라지므로, 더 작은 분할 단위가 언제나 더 쉬운 평가라고 가정할 수 없다.</p>",
        "<h2>데이터와 분할</h2>",
        "<p>기존 데이터는 338개 의미 조건·두 배치로 676개 입력, 35 family·169 contrast다. "
        "연결 group도 169개이므로 연결 group과 개별 contrast는 동일한 분할과 학습 결과를 공유한다. "
        "새 데이터는 336개 의미 조건, A 336개/B 121개 입력, 87 contrast·20 연결 group·23 family다.</p>",
        "<p>기존 family를 동사와 동일시하지 않는다. 새 데이터는 주 작업·위험 주제에 따라 기존 taxonomy에 대응시켰고, "
        "화학 혼합 및 표면·원본 보존 두 유형을 추가했다. 하나의 contrast에 여러 위험 대조가 포함될 수 있어 "
        "엄격한 단일 위험 원리 holdout은 아니다. family_mapping.json에 분류 규칙을 보존했다.</p>",
        "<p>같은 의미 조건 또는 두 카메라 RGB와 robot state가 같은 입력은 분할 간 겹치지 않는다. "
        "Test를 유지하고 validation과 train에서 겹치는 표본을 제거했다. A/B는 함께 분할한 후 별도 probe로 학습했다. "
        "공유 조건은 비연결 분할에서 여러 fold의 test가 될 수 있지만, 한 fold 안에서는 복제하지 않았다. "
        "따라서 test 출현 수와 고유 입력 수를 구분한다.</p>",
        "<h2>고정한 학습 평가 설정</h2>",
        "<p>78개 위치, token 축 mean pooling, 공통 LIBERO horizon 10·내부 차원 32·10회 denoising·고정 noise seed "
        "20261116·batch 12를 사용했다. 새 데이터 feature는 기존 추출 결과를 읽기 전용으로 재사용했고, "
        "기존 데이터 feature는 동일한 다섯 체크포인트와 추론 설정으로 새로 추출했다.</p>",
        "<p>StandardScaler는 train에만 맞췄다. L2 logistic regression, C=0.01/0.1/1, LBFGS max_iter=1000, tol=1e-5다. "
        "가중치는 분할 단위가 바뀌어도 고정된 연결 component ID를 기준으로 계산한다. Train 내 component 크기를 보정한 뒤 "
        "안전·위험 가중치 합을 각각 N_train/2로 맞춘다. Validation의 가중치 없는 AUROC로 C를 선택하고 dev를 합쳐 재학습하지 않는다. "
        "Logit 0 기준으로 분류하며, 주요 AUROC·balanced accuracy는 다섯 test fold의 평균이다.</p>",
        "<p>기존 0.675 등의 점수는 이전 가중치·noise·모델 구성의 결과다. 아래의 기존 데이터 family 조건도 "
        "최근 공통 설정으로 다시 학습한 값이므로 원래 보고서와 동일해야 하는 값은 아니다. "
        "반면 새 데이터 연결 group의 780개 조건은 기존 최근 실험의 예측과 1e-8 이내로 일치함을 확인했다.</p>",
        "<h2>주요 결과</h2><p>사전에 지정한 마지막 layer 및 flow 10의 AUROC다. ±는 fold 표준편차이며 신뢰구간이 아니다.</p>",
        "<p>먼저 같은 공식 LIBERO FT 체크포인트에서 분할 단위만 비교한다.</p><table><tr>"
        "<th>Dataset</th><th>Track</th><th>위치</th><th>Family</th><th>Connected group</th>"
        "<th>Individual contrast</th></tr>", *headline, "</table>",
        "<img src='split_auroc.png' alt='데이터셋과 분할별 AUROC'>",
        "<p>AE-only의 VLM feature는 LIBERO FT와 완전히 같아 두 선이 겹친다.</p>",
        "<table><tr>" + "".join("<th>" + x + "</th>" for x in [
            "Dataset", "Track", "Split", "Model", "VLM17", "AE17 flow10", "Velocity pre", "Velocity post"]) + "</tr>",
        *table, "</table>",
        "<h2>해석 범위</h2><p>같은 데이터와 frozen feature에서 분할을 바꾼 성능 차이는, 학습한 구분 기준이 "
        "어떤 단위의 새로운 사례에 일반화되는지에 관한 차이다. VLA의 위험 지식이 실험 도중 생기거나 사라진 결과가 아니다. "
        "각 분할은 test 구성·학습 표본 수·공유 표본 제거량이 달라지므로, 점수 차이를 분할 이름 하나의 순수 인과 효과로 "
        "해석하지 않는다. 기존 데이터와 새 데이터의 지시문·관측·상황 구성도 그대로 다르다.</p>",
        "<p>새 데이터의 비연결 분할은 공유 표본을 제외한 조건부 일반화다. 원래 대조 묶음 전체를 train에 모두 유지한 "
        "평가와 구분해야 한다. 라벨은 기존 후보 annotation을 유지했으며 독립 사람 검증을 추가하지 않았다. "
        "위험 정보의 인과적 전달이나 행동 억제 여부를 검증한 실험도 아니다.</p>",
        "<p>기존 데이터에는 동일한 초기 입력에 반대 label을 붙인 압력 관련 4쌍·8입력이 남아 있다. "
        "새 데이터에서는 동일 초기 입력의 반대 label을 확인하지 못했다. 데이터 수정 없이 비교하기 위해 이 조건을 유지했다.</p>",
        "<p>CSV의 accuracy·confusion matrix·위험 예측 비율은 test 출현을 합산한 값이다. 비연결 분할에서는 "
        "공유 조건이 여러 fold에 등장할 수 있으므로, 고유 입력을 한 번씩 평가한 OOF accuracy와 동일시하지 않는다.</p>",
        "<h2>검증 및 원본 결과</h2><p><a href='all_sites.csv'>전체 3510개 조건 CSV</a> · "
        "<a href='summary.json'>주요 결과 JSON</a> · <a href='../split_audit.json'>분할별 실제 표본 수·제거량</a> · "
        "<a href='../family_mapping.json'>Family 대응표</a> · <a href='../experiment_protocol.json'>공통 프로토콜</a></p>",
        "<p><a href='../numerical_audit.json'>저장된 분류기와 scaler 수치 검증</a></p>",
        "<details><summary>분할 검증 상세</summary><pre>" + html.escape(json.dumps(audit, ensure_ascii=False, indent=2))
        + "</pre></details></html>",
    ]
    (out / "index.html").write_text("\n".join(texts))
    write_json(out / "summary.json", dict(complete=True, primary=primary, conditions=len(results)))
    write_json(RUN / "verification.json", dict(
        passed=True, conditions=3510, original_protocol_reproductions=780,
        legacy_connected_equals_contrast=True, input_hashes_identical_between_models=True,
        ae_only_vlm_identical_to_libero=True, all_fit_parameters_and_features_finite=True,
        checkpoint_weights_and_initial_noise_equal_between_datasets=True,
        actual_preprocessed_input_overlap_across_partitions=0,
        unique_selected_classifier_numerical_audit_passed=True,
        original_experiment_files_modified=False, report=str(out / "index.html")))


def numerical_audit():
    from concurrent.futures import ThreadPoolExecutor
    import joblib
    paths = sorted({p.resolve() for p in (RUN / "probes").rglob("fold_*.joblib")})
    assert len(paths) == 15600, len(paths)

    def check(path):
        p = joblib.load(path)
        scaler, classifier = p["scaler"], p["classifier"]
        assert all(np.isfinite(a).all() for a in (
            scaler.mean_, scaler.scale_, scaler.var_, classifier.coef_, classifier.intercept_)), str(path)
        negative = scaler.var_ < 0
        if negative.any():
            assert np.all(scaler.scale_[negative] == 1) and np.min(scaler.var_) > -1e-12, str(path)
        return int(negative.sum()), float(np.min(scaler.var_))

    with ThreadPoolExecutor(max_workers=16) as executor:
        checked = list(executor.map(check, paths))
    write_json(RUN / "numerical_audit.json", dict(
        passed=True, unique_selected_classifiers=len(paths), finite_parameters=True,
        scalers_with_roundoff_negative_variance=sum(n > 0 for n, _ in checked),
        negative_variance_dimensions=sum(n for n, _ in checked),
        minimum_variance=min(v for _, v in checked), all_negative_variances_handled_as_scale_one=True,
        note="RuntimeWarning in sqrt is from roundoff in effectively constant dimensions; final scales and parameters verified finite"))


def command(python, args, label, env):
    logs = RUN / "logs"
    logs.mkdir(exist_ok=True)
    write_json(RUN / "pipeline_status.json", dict(state="running", phase=label, updated_unix=time.time()))
    with (logs / (label + ".log")).open("a") as stream:
        proc = subprocess.Popen([str(python), str(SCRIPT), *args], cwd=ROOT, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
        write_json(RUN / "active_child.json", dict(pid=proc.pid, phase=label, started_unix=time.time()))
        while proc.poll() is None:
            if (RUN / "pause_requested.json").exists():
                proc.terminate()
                proc.wait()
                raise InterruptedError("User pause requested")
            time.sleep(2)
        if proc.returncode:
            tail = (logs / (label + ".log")).read_text()[-6000:]
            raise RuntimeError(f"{label} exited {proc.returncode}\n{tail}")


def supervise(workers):
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A supervisor already owns this run")
        if all((RUN / p).exists() and json.loads((RUN / p).read_text()).get("state") == "complete"
               for p in ("pipeline_status.json", "experiment_status.json")):
            assert json.loads((RUN / "verification.json").read_text())["passed"]
            print("Experiment already complete; no stages restarted.", flush=True)
            return
        write_json(RUN / "supervisor_process.json", dict(pid=os.getpid(), started_unix=time.time()))
        env = dict(os.environ)
        env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   JAX_PLATFORMS="cpu", PYTHONPATH=str(ROOT / "openpi/src") + ":"
                   + str(ROOT / "openpi/packages/openpi-client/src"))
        cpu, gpu = ROOT / "envs/representation-probe/bin/python", ROOT / "openpi/.venv/bin/python"
        for attempt in range(1, 4):
            try:
                command(cpu, ["prepare"], "prepare", env)
                for model in MODELS:
                    folder = RUN / "datasets/legacy"
                    for limit in [16, None]:
                        path = folder / ("smoke_features" if limit else "features") / (model + "_audit.json")
                        if path.exists() and json.loads(path.read_text()).get("complete"):
                            continue
                        args = ["extract", "--model", model] + (["--limit", "16"] if limit else [])
                        command(gpu, args, "legacy_" + model + ("_smoke" if limit else "_full"), env)
                milestone("legacy_features", "기존 데이터 676입력·5모델·78위치 공통 설정 추출 완료",
                          dict(inputs=676, models=5, sites=78))
                command(cpu, ["probe", "--smoke", "--workers", str(workers)], "probe_preflight", env)
                command(cpu, ["probe", "--workers", str(workers)], "probes", env)
                command(cpu, ["audit"], "numerical_audit", env)
                command(cpu, ["report"], "report", env)
                for name in ("pipeline_status.json", "experiment_status.json"):
                    write_json(RUN / name, dict(state="complete", updated_unix=time.time()))
                write_json(RUN / "supervisor_status.json", dict(state="complete", updated_unix=time.time()))
                notify_progress(RUN)
                return
            except InterruptedError:
                write_json(RUN / "supervisor_status.json", dict(state="paused", updated_unix=time.time()))
                return
            except Exception:
                error = dict(attempt=attempt, error=traceback.format_exc(), updated_unix=time.time())
                write_json(RUN / "pipeline_error.json", error)
                notify_progress(RUN)
                if attempt == 3:
                    write_json(RUN / "supervisor_status.json", dict(state="needs_attention", updated_unix=time.time()))
                    notify_progress(RUN)
                    raise
                archive = RUN / "resolved_errors" / f"attempt_{time.time_ns()}.json"
                archive.parent.mkdir(exist_ok=True)
                (RUN / "pipeline_error.json").replace(archive)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("command", choices=["prepare", "extract", "probe", "audit", "report", "supervise"])
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "extract":
        extract(args.model, args.limit)
    elif args.command == "probe":
        probes(args.workers, args.smoke)
    elif args.command == "report":
        report()
    elif args.command == "audit":
        numerical_audit()
    elif args.command == "supervise":
        supervise(args.workers)


if __name__ == "__main__":
    main()
