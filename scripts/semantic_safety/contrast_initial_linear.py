"""Frozen base/LIBERO initial-scene extraction and leakage-controlled linear probes."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np

from representation_common import feature_sites
from rollout_common import ROOT, write_json

DATA = ROOT / "data/semantic_safety_contrast_v1"
RUN = ROOT / "runs/semantic_safety_contrast_initial_linear_v1"
MODELS = ("base", "libero")
SEED = 20261116
C_GRID = (.01, .1, 1.)


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def milestone(key, title, result):
    from experiment_notifications import notify_progress
    write_json(RUN / "milestones" / (key + ".json"), dict(complete=True, title=title, result=result))
    notify_progress(RUN)


def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    write_json(RUN / "reporting/delivery_mode.json", {"delivery_mode": "foreground"})
    conditions = {r["condition_id"]: r for r in read_jsonl(DATA / "conditions.jsonl")}
    items = read_jsonl(DATA / "items.jsonl")
    sets = read_jsonl(DATA / "contrast_sets.jsonl")
    parent = {s["set_id"]: s["set_id"] for s in sets}

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(a, b):
        a, b = root(a), root(b)
        parent[max(a, b)] = min(a, b)

    physical = {}
    reports = {}
    for row in conditions.values():
        for key in row["set_ids"][1:]: union(row["set_ids"][0], key)
        sid = row["scene_id"]
        if sid not in reports:
            reports[sid] = json.loads((DATA / "observations" / sid / "validation.json").read_text())
        report = reports[sid]
        assert report["technical_pass"] and report["rendered"], sid
        key = (report["images"]["agentview"]["sha256"], report["images"]["wrist"]["sha256"], report["robot_state_sha256"])
        if key in physical: union(row["set_ids"][0], physical[key])
        physical[key] = row["set_ids"][0]
    components = defaultdict(list)
    for group in parent: components[root(group)].append(group)
    group_ids = {k: "g_" + digest(sorted(v))[:12] for k, v in components.items()}
    rows = []
    for item in items:
        original = conditions[item["condition_id"]]
        row = {**item, "group_id": group_ids[root(original["set_ids"][0])],
               "category": original["category"], "actor_id": original["actor_id"],
               "target_id": original["target_id"], "instruction_ko": original["instruction_ko"] if item["track"] == "A" else original["visual_instruction_ko"]}
        report = reports[row["scene_id"]]
        row["observation_path"] = str((DATA / row["observation_path"]).relative_to(ROOT))
        row["image_hash"] = digest([report["images"][c]["sha256"] for c in ["agentview", "wrist"]])
        rows.append(row)
    groups = sorted({r["group_id"] for r in rows})
    counts = np.array([[sum(r["group_id"] == group and r["track"] == track and r["label"] == label for r in rows)
                        for track, label in [("A", 0), ("A", 1), ("B", 0), ("B", 1)]] for group in groups])
    # Optimize only metadata counts, before representations or model predictions exist.
    rng = np.random.default_rng(SEED)
    target = counts.sum(0) / 5
    best = None
    for _ in range(6000):
        loads = np.zeros((5, 4), dtype=int)
        bins = [[] for _ in range(5)]
        order = np.argsort(-(counts / target).sum(1) + rng.normal(0, .8, len(groups)))
        for index in order:
            costs = []
            for b in range(5):
                candidate = loads.copy(); candidate[b] += counts[index]
                costs.append(float(np.square((candidate - target) / target).sum()) + .025 * len(bins[b]) + rng.uniform(0, .05))
            b = int(np.argmin(costs)); loads[b] += counts[index]; bins[b].append(groups[index])
        if (loads < 2).any() or any(not b for b in bins): continue
        score = float(np.square((loads - target) / target).sum())
        if best is None or score < best[0]: best = score, bins, loads
    assert best is not None, "Could not construct five viable bins"
    _, bins, loads = best
    folds = []
    for fold in range(5):
        test = set(bins[fold]); dev = set(bins[(fold + 1) % 5])
        train = set(groups) - test - dev
        parts = dict(train_groups=sorted(train), validation_groups=sorted(dev), test_groups=sorted(test))
        count_table = {}
        for name in ["train", "validation", "test"]:
            chosen = [r for r in rows if r["group_id"] in parts[name + "_groups"]]
            count_table[name] = {track: dict(Counter("unsafe" if r["label"] else "safe" for r in chosen if r["track"] == track)) for track in ["A", "B"]}
            assert all(len(v) == 2 for v in count_table[name].values())
        folds.append(dict(fold=fold, **parts, counts=count_table))
    for row in rows:
        assert len({group_ids[root(s)] for s in conditions[row["condition_id"]]["set_ids"]}) == 1
    assert all(sum(r["group_id"] in f["test_groups"] for f in folds) == 1 for r in rows)
    manifest = dict(rows=rows, folds=folds, grouping="connected shared controls plus identical two-camera observations",
                    groups=len(groups), component_sets={group_ids[k]: v for k, v in components.items()},
                    random_seed=SEED, split_uses_features_or_predictions=False)
    if (RUN / "manifest.json").exists(): assert json.loads((RUN / "manifest.json").read_text()) == manifest
    else: write_json(RUN / "manifest.json", manifest)
    protocol = dict(models=list(MODELS), dataset="semantic_safety_contrast_v1", semantic_conditions=len(conditions),
                    inputs=len(rows), tracks_separately_fitted=True, groups=len(groups), folds=5, C_grid=list(C_GRID),
                    VLA_training=False, LDA=False, only_initial_scene=True, environment_steps=0,
                    native_instruction=True, safety_question_added=False, cameras=["agentview", "wrist"],
                    input_orientation="upright RGB from new dataset; no further rotation or reflection",
                    shared_readout=dict(action_horizon=10, action_dim=32, discrete_state_input=False, denoising_steps=10,
                                        noise_seed=SEED, same_initial_noise_for_all_inputs_and_models=True),
                    base_default_readout_note="Base default is 50 action tokens and discrete state. Both checkpoints use the LIBERO readout here for controlled comparison, without weight updates.",
                    pooling="VLM valid-token mean including images/instruction; AE and velocity action-token mean only",
                    sites=list(dict(feature_sites())), primary_sites=["vlm/layer_17", "ae/flow_10/layer_17", "velocity_pre/flow_10", "velocity_post/flow_10"],
                    training_weights="equal connected-group mass then train-only inverse weighted class mass; sum=n_train",
                    scaler="training only, with same weights", selection="validation AUROC, smaller C on ties, no validation refit",
                    threshold_logit=0., max_iter=1000, tol=1e-5,
                    labels="provisional request-level annotations, not independently human-validated gold",
                    scope="Held-out connected contrast groups; not a separately engineered leave-one-hazard-type-out benchmark",
                    track_comparison_note="A has 336 inputs and B 121; aggregate scores cannot isolate modality effects.")
    write_json(RUN / "experiment_protocol.json", protocol)
    milestone("inputs_and_splits", "첫 장면 입력 및 누출 방지 분할 확정", {"inputs": len(rows), "groups": len(groups), "bin_counts": loads.tolist()})
    print(json.dumps({"inputs": len(rows), "groups": len(groups), "bins_A_safe_unsafe_B_safe_unsafe": loads.tolist()}), flush=True)


def extract(model_name, limit=None):
    import dataclasses
    import torch
    from openpi.training import config, checkpoints
    from openpi.policies import policy_config
    from openpi_client import image_tools
    from transformers.modeling_utils import no_init_weights
    from unittest.mock import patch
    from safetensors import safe_open
    import safetensors.torch
    from motor_lora import frozen_parameter_digest, tensor_batch
    from extract_representation_linear import Capture

    manifest = json.loads((RUN / "manifest.json").read_text())
    rows = manifest["rows"][:limit] if limit else manifest["rows"]
    cfg = config.get_config("pi05_libero")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    path = ROOT / "checkpoints" / ("pi05_base_pytorch" if model_name == "base" else "pi05_libero_pytorch")
    stats = checkpoints.load_norm_stats(ROOT / "checkpoints/pi05_libero_pytorch/assets", "physical-intelligence/libero")
    torch.set_num_threads(8)
    print(json.dumps({"stage":"loading_checkpoint","model":model_name}),flush=True)
    # Every stored parameter is loaded strictly; avoid initializing billions of
    # throwaway random values before replacing them with checkpoint weights.
    original_load = safetensors.torch.load_model
    restored_ties = []
    def load_declared_ties(instance, filename, *args, **kwargs):
        # no_init_weights also skips HF's tie_weights. Restore only aliases
        # explicitly declared by the saved checkpoint before strict loading.
        with safe_open(filename, framework="pt", device="cpu") as source:
            aliases = source.metadata() or {}
        for alias, retained in aliases.items():
            if not alias.endswith(".weight") or not retained.endswith(".weight"): continue
            alias_parent, leaf = alias.rsplit(".", 1)
            retained_parent, retained_leaf = retained.rsplit(".", 1)
            parameter = getattr(instance.get_submodule(retained_parent), retained_leaf)
            setattr(instance.get_submodule(alias_parent), leaf, parameter)
            restored_ties.append([alias, retained])
        return original_load(instance, filename, *args, **kwargs)
    with no_init_weights(), patch("safetensors.torch.load_model", new=load_declared_ties):
        policy = policy_config.create_trained_policy(cfg, path, norm_stats=stats, pytorch_device="cuda")
    print(json.dumps({"stage":"checkpoint_loaded","model":model_name}),flush=True)
    model = policy._model
    model.requires_grad_(False); model.eval(); torch.set_num_threads(8)
    assert not any(p.requires_grad for p in model.parameters())
    model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
    before = frozen_parameter_digest(model)
    smoke_audit = RUN / "smoke_features" / (model_name + "_audit.json")
    if not limit and smoke_audit.exists():
        assert before == json.loads(smoke_audit.read_text())["weight_digest_before"]
    folder = RUN / ("smoke_features" if limit else "features")
    folder.mkdir(parents=True, exist_ok=True)
    capture = Capture(model)
    text_head_calls = []
    for head in [model.paligemma_with_expert.paligemma.lm_head, model.paligemma_with_expert.gemma_expert.lm_head]:
        head.register_forward_hook(lambda *args: text_head_calls.append(True))
    fixed_noise = np.random.default_rng(SEED).standard_normal((10, 32)).astype(np.float32)
    loaded = {}
    def inputs(selected):
        values, fingerprints = [], []
        for row in selected:
            if row["scene_id"] not in loaded:
                with np.load(ROOT / row["observation_path"]) as source:
                    loaded[row["scene_id"]] = tuple(source[key].copy() for key in ["agentview_rgb", "wrist_rgb", "robot_state"])
            main, wrist, state = loaded[row["scene_id"]]
            value = policy._input_transform({"observation/image": image_tools.resize_with_pad(main, 224, 224),
                       "observation/wrist_image": image_tools.resize_with_pad(wrist, 224, 224),
                       "observation/state": state, "prompt": row["instruction"]})
            assert np.array_equal(value["tokenized_prompt_mask"], np.arange(200) < value["tokenized_prompt_mask"].sum())
            assert value["tokenized_prompt_mask"].sum() < 200, "Potential instruction truncation"
            h = hashlib.sha256()
            for key in sorted(value["image"]): h.update(np.ascontiguousarray(value["image"][key]).tobytes())
            for key in ["state", "tokenized_prompt", "tokenized_prompt_mask"]: h.update(np.ascontiguousarray(value[key]).tobytes())
            values.append(value); fingerprints.append(h.hexdigest())
        obs, _ = tensor_batch(values)
        noise = torch.from_numpy(np.repeat(fixed_noise[None], len(selected), axis=0)).cuda()
        return obs, noise, fingerprints

    started = time.monotonic()
    batch, noise, hashes = inputs(rows[:2])
    capture.enabled = False
    with torch.inference_mode(): expected = model.sample_actions("cuda", batch, noise=noise.clone(), num_steps=10)
    capture.enabled = True; capture.reset()
    with torch.inference_mode(): actual = model.sample_actions("cuda", batch, noise=noise.clone(), num_steps=10)
    assert torch.equal(actual, expected), "Hooks altered actions"
    capture.arrays()
    signature = digest([rows, before, json.loads((RUN / "experiment_protocol.json").read_text())])
    out_path = folder / (model_name + ".h5")
    with h5py.File(out_path, "a") as output:
        if "written" not in output:
            output.attrs.update(signature=signature, complete=False, model_weight_digest=before)
            output.create_dataset("written", shape=(len(rows),), dtype="bool")
            output.create_dataset("input_sha256", shape=(len(rows),), dtype="S64")
            for site, width in feature_sites(): output.create_dataset(site, shape=(len(rows), width), dtype="float32", compression="lzf", chunks=(min(16, len(rows)), width))
        assert output.attrs["signature"] == signature
        for start in range(0, len(rows), 12):
            indices = np.arange(start, min(start + 12, len(rows)))
            if output["written"][indices].all(): continue
            batch, noise, hashes = inputs([rows[i] for i in indices])
            capture.reset()
            with torch.inference_mode(): model.sample_actions("cuda", batch, noise=noise, num_steps=10)
            for site, values in capture.arrays().items(): output[site][indices] = values
            output["input_sha256"][indices] = np.asarray(hashes, dtype="S64")
            output.flush(); output["written"][indices] = True; output.flush()
            write_json(folder / (model_name + "_progress.json"), {"completed": int(output["written"][:].sum()), "total": len(rows)})
        assert output["written"][:].all()
        output.attrs["complete"] = True
    after = frozen_parameter_digest(model)
    assert before == after, "Frozen weights changed"
    assert not text_head_calls, "Unexpected text-generation head use"
    audit = dict(complete=True, model=model_name, rows=len(rows), sites=78, all_weights_frozen=True,
                 weight_digest_before=before, weight_digest_after=after, hook_action_max_abs_difference=0.,
                 fixed_initial_noise_sha256=hashlib.sha256(fixed_noise.tobytes()).hexdigest(),
                 unused_text_generation_head_calls=len(text_head_calls),
                 checkpoint_declared_ties_restored=restored_ties,
                 seconds=time.monotonic()-started)
    other = folder / ("base.h5" if model_name == "libero" else "libero.h5")
    if other.exists():
        with h5py.File(other) as a, h5py.File(out_path) as b: assert np.array_equal(a["input_sha256"][:], b["input_sha256"][:])
        audit["inputs_bit_exact_between_checkpoints"] = True
    write_json(folder / (model_name + "_audit.json"), audit)
    if not limit: milestone("features_" + model_name, model_name + " 첫 장면 VLM·AE 추출 및 freeze 검증 완료", audit)
    print(json.dumps(audit), flush=True)


def group_weights(rows, balance_class=False):
    counts = Counter(r["group_id"] for r in rows)
    weight = np.asarray([1. / counts[r["group_id"]] for r in rows])
    if balance_class:
        labels = np.asarray([r["label"] for r in rows])
        for label in [0, 1]: weight[labels == label] *= .5 / weight[labels == label].sum()
    return weight * len(rows) / weight.sum()


def metrics(rows, scores):
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
    y = np.asarray([r["label"] for r in rows]); pred = scores >= 0
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    weights = group_weights(rows)
    return dict(n=len(y), safe=int((y == 0).sum()), unsafe=int((y == 1).sum()),
                accuracy=float(np.mean(pred == y)), balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                AUROC=float(roc_auc_score(y, scores)), group_weighted_AUROC=float(roc_auc_score(y, scores, sample_weight=weights)),
                group_weighted_balanced_accuracy=float(balanced_accuracy_score(y, pred, sample_weight=weights)),
                predicted_unsafe=int(pred.sum()), predicted_unsafe_fraction=float(pred.mean()),
                safe_recall=float(tn / (tn + fp)), unsafe_recall=float(tp / (tp + fn)),
                confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]])


def probe_job(job):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    import warnings
    model, track, site = job
    manifest = json.loads((RUN / "manifest.json").read_text())
    all_rows = manifest["rows"]
    indices = np.asarray([i for i, r in enumerate(all_rows) if r["track"] == track])
    rows = [all_rows[i] for i in indices]
    with h5py.File(RUN / "features" / (model + ".h5")) as f:
        assert f.attrs["complete"]
        matrix = f[site][:][indices].astype(np.float64)
        feature_sig = str(f.attrs["signature"])
    signature = digest([feature_sig, track, site, manifest["folds"], C_GRID, "group_then_class_weights_v1"])
    output = RUN / "probes" / track / model / site
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists():
        cached = json.loads((output / "metrics.json").read_text()); assert cached["signature"] == signature
        return cached
    labels = np.asarray([r["label"] for r in rows]); groups = np.asarray([r["group_id"] for r in rows])
    oof = np.full(len(rows), np.nan); fold_ids = np.full(len(rows), -1, dtype=int); results = []
    with threadpool_limits(limits=1):
        for split in manifest["folds"]:
            idx = {part: np.flatnonzero(np.isin(groups, split[part + "_groups"])) for part in ["train", "validation", "test"]}
            for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]: assert not set(idx[a]) & set(idx[b])
            train = idx["train"]; dev = idx["validation"]; test = idx["test"]
            weights = group_weights([rows[i] for i in train], balance_class=True)
            scaler = StandardScaler().fit(matrix[train], sample_weight=weights)
            transformed = {k: scaler.transform(matrix[v]) for k, v in idx.items()}
            best = None; candidates = []
            for C in C_GRID:
                classifier = LogisticRegression(C=C, solver="lbfgs", max_iter=1000, tol=1e-5, random_state=SEED)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    classifier.fit(transformed["train"], labels[train], sample_weight=weights)
                assert not any(issubclass(w.category, ConvergenceWarning) for w in caught), (job, split["fold"], C)
                score = float(roc_auc_score(labels[dev], classifier.decision_function(transformed["validation"])))
                candidates.append(dict(C=C, validation_AUROC=score, iterations=int(classifier.n_iter_.max())))
                if best is None or score > best[0] + 1e-12: best = score, classifier
            value, classifier = best
            predicted = classifier.decision_function(transformed["test"])
            assert np.isnan(oof[test]).all(); oof[test] = predicted; fold_ids[test] = split["fold"]
            result = dict(fold=split["fold"], C=classifier.C, validation_AUROC=value, candidates=candidates,
                          train_indices=indices[train].tolist(), validation_indices=indices[dev].tolist(), test_indices=indices[test].tolist(),
                          train_weighted_safe_mass=float(weights[labels[train] == 0].sum()),
                          train_weighted_unsafe_mass=float(weights[labels[train] == 1].sum()),
                          test=metrics([rows[i] for i in test], predicted))
            temporary = output / (f"fold_{split['fold']}.joblib.tmp")
            joblib.dump(dict(scaler=scaler, classifier=classifier, split=split, signature=signature,
                             training_rows=[rows[i]["item_id"] for i in train]), temporary, compress=3)
            temporary.replace(output / f"fold_{split['fold']}.joblib")
            results.append(result)
    assert np.isfinite(oof).all() and (fold_ids >= 0).all()
    np.savez_compressed(output / "oof_predictions.npz", decision_score=oof, fold=fold_ids, label=labels, global_indices=indices)
    summary = dict(complete=True, signature=signature, model=model, track=track, site=site, folds=results,
                   fold_mean_AUROC=float(np.mean([r["test"]["AUROC"] for r in results])),
                   fold_sd_AUROC=float(np.std([r["test"]["AUROC"] for r in results], ddof=1)),
                   fold_mean_balanced_accuracy=float(np.mean([r["test"]["balanced_accuracy"] for r in results])),
                   oof=metrics(rows, oof), caution="OOF logits come from five separately fitted probes; primary AUROC is fold mean.")
    write_json(output / "metrics.json", summary)
    return summary


def probes(workers):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    jobs = [(model, track, site) for model in MODELS for track in ["A", "B"] for site, _ in feature_sites()]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(probe_job, job) for job in jobs]
        for i, future in enumerate(as_completed(futures)):
            future.result()
            write_json(RUN / "probe_progress.json", dict(completed=i + 1, total=len(jobs)))
    milestone("probes", "두 모델·두 트랙의 전체 위치 linear probe 학습·평가 완료", {"conditions": len(jobs), "selected_fits": len(jobs) * 5})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["prepare", "extract", "probe"])
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    if args.command == "prepare": prepare()
    elif args.command == "extract":
        assert args.model
        extract(args.model, args.limit)
    else: probes(args.workers)


if __name__ == "__main__": main()
