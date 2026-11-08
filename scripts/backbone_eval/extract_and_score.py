"""Run ONE model over the evaluation manifests: validation checks, feature extraction, single-token likelihood.

For every dataset and split it stores runs/backbone_eval_v1/features/<model>/<dataset>/<split>.npz with
    item_id, label, n_choices, h [n, D] float32, cand_ids [K], cand_logp [n, K] (nan = choice not present),
    pred_ll, top1_id, top1_logp, cand_mass, margin, n_prompt_tokens
plus test_imgshuf.npz (same test items, images deranged across image groups) for the diagnostic control.
Validation results go to runs/backbone_eval_v1/validation/<model>.json and the model card to models_manifest.json.

Usage (inside the right Python environment):
    python extract_and_score.py --model pi0 [--datasets ...] [--splits train dev test] [--smoke 16] [--batch 16]
"""

from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import json
import pathlib
import sys
import time
import traceback

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from backbone_eval.adapters import MODEL_SPECS, load_reader  # noqa: E402
from backbone_eval.adapters.base import summarize_checks  # noqa: E402
from common import (  # noqa: E402
    DATA_EVAL,
    DATASETS,
    LETTERS,
    RUN_DIR,
    SEED,
    ImageStore,
    candidate_labels,
    dump_json,
    item_rng,
    read_items,
    unique_item_ids,
)
from evaluation_checks import extraction_failures  # noqa: E402

AVAILABLE = [d for d in DATASETS if (DATA_EVAL / d / "items.parquet").exists()]


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def score_batch(out, cand_ids: list[int | None], n_choices: np.ndarray, task_type: str):
    """From full-vocab logp -> per-candidate logp (nan for absent choices), prediction, diagnostics."""
    b, K = out.logp.shape[0], len(cand_ids)
    cand_logp = np.full((b, K), np.nan, np.float32)
    for k, cid in enumerate(cand_ids):
        if cid is not None:
            cand_logp[:, k] = out.logp[:, cid]
    if task_type == "mc":
        for i in range(b):
            cand_logp[i, n_choices[i] :] = np.nan
    pred = np.array([int(np.nanargmax(r)) if np.isfinite(r).any() else -1 for r in cand_logp])
    top1 = out.logp.argmax(-1)
    top1_logp = out.logp[np.arange(b), top1]
    mass = np.nansum(np.exp(cand_logp), axis=1)
    srt = np.sort(np.where(np.isnan(cand_logp), -np.inf, cand_logp), axis=1)
    margin = srt[:, -1] - srt[:, -2] if K >= 2 else np.full(b, np.nan)
    return cand_logp, pred, top1, top1_logp, mass, margin


def run_split(reader, ds, items, store, cand_ids, task_type, batch, out_path, image_override=None, log=print):
    """items: list of manifest rows (already length-filtered). image_override: dict item_id -> image_key."""
    items = sorted(items, key=lambda r: len(reader.prompt_ids(r["body"])))  # length-sorted batches = less padding
    recs = collections.defaultdict(list)
    t0 = time.time()
    for bi, chunk in enumerate(batched(items, batch)):
        keys = [image_override[r["item_id"]] if image_override else r["image_key"] for r in chunk]
        pils = [store.pil(k) for k in keys]
        out = reader.forward(pils, [r["body"] for r in chunk])
        if not np.isfinite(out.h).all():
            raise ValueError("Non-finite hidden state")
        n_ch = np.array([r["n_choices"] if task_type == "mc" else 2 for r in chunk])
        cand_logp, pred, top1, top1_logp, mass, margin = score_batch(out, cand_ids, n_ch, task_type)
        if all(cid is not None for cid in cand_ids):
            for i, count in enumerate(n_ch):
                if not np.isfinite(cand_logp[i, :count]).all():
                    raise ValueError("Non-finite candidate log probability")
        recs["item_id"] += [r["item_id"] for r in chunk]
        recs["source_item_id"] += [r.get("source_item_id", r["item_id"]) for r in chunk]
        recs["label"] += [r["label"] for r in chunk]
        recs["n_choices"] += list(n_ch)
        recs["h"].append(out.h)
        recs["cand_logp"].append(cand_logp)
        recs["pred_ll"] += list(pred)
        recs["top1_id"] += list(top1)
        recs["top1_logp"] += list(top1_logp)
        recs["cand_mass"] += list(mass)
        recs["margin"] += list(margin)
        recs["n_prompt_tokens"] += list(out.n_prompt_tokens)
        recs["image_key"] += keys
        if bi % 20 == 0:
            log(f"    batch {bi} ({len(recs['item_id'])}/{len(items)}) {time.time() - t0:.0f}s")
    np.savez(out_path, item_id=np.array(recs["item_id"]), label=np.array(recs["label"], np.int32),
             n_choices=np.array(recs["n_choices"], np.int32), h=np.concatenate(recs["h"]).astype(np.float32),
             cand_ids=np.array([-1 if c is None else c for c in cand_ids], np.int64), cand_logp=np.concatenate(recs["cand_logp"]),
             pred_ll=np.array(recs["pred_ll"], np.int32), top1_id=np.array(recs["top1_id"], np.int64),
             top1_logp=np.array(recs["top1_logp"], np.float32), cand_mass=np.array(recs["cand_mass"], np.float32),
             margin=np.array(recs["margin"], np.float32), n_prompt_tokens=np.array(recs["n_prompt_tokens"], np.int32),
             image_key=np.array(recs["image_key"]), source_item_id=np.array(recs["source_item_id"]))
    acc = float(np.mean(np.array(recs["pred_ll"]) == np.array(recs["label"]))) if recs["pred_ll"] else float("nan")
    log(f"    saved {out_path.name}: n={len(recs['item_id'])} ll_acc={acc:.3f} mean_cand_mass={np.mean(recs['cand_mass']):.3f} "
        f"{time.time() - t0:.0f}s")
    return acc


def derangement(items, rng):
    """Map each item to the image of an item from a DIFFERENT image group (seeded, rejection sampled)."""
    keys = [r["image_key"] for r in items]
    groups = [r["group_id"] for r in items]
    for _ in range(1000):
        perm = rng.permutation(len(items))
        if all(groups[perm[i]] != groups[i] for i in range(len(items))):
            return {items[i]["item_id"]: keys[perm[i]] for i in range(len(items))}
    # fall back: greedy
    out = {}
    for i, r in enumerate(items):
        cands = [j for j in range(len(items)) if groups[j] != r["group_id"]]
        out[r["item_id"]] = keys[cands[rng.integers(len(cands))]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_SPECS))
    ap.add_argument("--datasets", nargs="+", default=AVAILABLE)
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--smoke", type=int, default=0, help="limit items per split (validation only; writes to <out>/smoke)")
    ap.add_argument("--imgshuf", type=int, default=100)
    ap.add_argument("--out", default=str(RUN_DIR))
    ap.add_argument("--no_validate", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="re-extract splits whose .npz already exists (default: skip them)")
    ap.add_argument("--unique-item-ids", action="store_true", help="disambiguate IDs in memory; preserve frozen source rows")
    ap.add_argument("--strict-validation", action="store_true", help="exit nonzero on failed checks or missing splits")
    args = ap.parse_args()
    if args.strict_validation and args.no_validate:
        ap.error("--strict-validation requires validation")

    out_root = pathlib.Path(args.out) / ("smoke" if args.smoke else "")
    feat_root = out_root / "features" / args.model
    val_dir = out_root / "validation"
    feat_root.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg):
        print(f"[{args.model}] {msg}", flush=True)
        log_lines.append(msg)

    t0 = time.time()
    reader = load_reader(args.model, batch_size=args.batch)
    log(f"loaded in {time.time() - t0:.0f}s: {json.dumps(reader.describe(), default=str)[:400]}...")
    prev_path = val_dir / f"{args.model}.json"
    prev = json.loads(prev_path.read_text()) if prev_path.exists() else {}
    # merge with an earlier (partial) run of the same model: datasets re-run below overwrite their entries
    report = {"model": args.model, "describe": reader.describe(), "datasets": dict(prev.get("datasets", {})), "status": "running",
              "previous_runs": prev.get("previous_runs", []) + ([{"status": prev.get("status"), "seconds": prev.get("seconds")}] if prev else [])}
    fp_before = reader.param_fingerprint()
    report["param_fingerprint_before"] = fp_before

    if not args.no_validate:
        report["head"] = reader.check_head()
        report["attention_direction"] = reader.check_attention_direction()
        log(f"head={report['head']}")
        log(f"attention={report['attention_direction']}")

    for ds in args.datasets:
        if ds not in AVAILABLE:
            report["datasets"][ds] = {"status": "blocked", "reason": "no manifest (gated data)"}
            continue
        task_type = DATASETS[ds][1]
        items = read_items(ds)
        if args.unique_item_ids:
            items = unique_item_ids(items)
        store = ImageStore(ds)
        kmax = max(r["n_choices"] for r in items) if task_type == "mc" else 2
        labels = candidate_labels(task_type, kmax)
        checks = reader.candidate_checks(labels, [r["body"] for r in items[:5]])
        cand_ids = [c.token_id for c in checks]
        dsrep = {"task_type": task_type, "candidates": summarize_checks(checks), "splits": {},
                 "unique_item_ids": args.unique_item_ids,
                 "items_sha256": hashlib.sha256((DATA_EVAL / ds / "items.parquet").read_bytes()).hexdigest()}
        if not all(c.ok for c in checks):
            dsrep["likelihood_status"] = "unsupported (candidate not a single token)"
            log(f"{ds}: candidate check FAILED: {dsrep['candidates']}")
        else:
            dsrep["likelihood_status"] = "ok"
        # length filter (shared across splits)
        lens = {r["item_id"]: len(reader.prompt_ids(r["body"])) for r in items}
        keep = [r for r in items if lens[r["item_id"]] <= reader.max_text_tokens]
        dsrep["n_excluded_too_long"] = len(items) - len(keep)
        dsrep["prompt_tokens"] = {"min": int(min(lens.values())), "max": int(max(lens.values())),
                                  "mean": float(np.mean(list(lens.values())))}
        dsrep["n_newline_tokens_example"] = None
        if not args.no_validate and keep:
            # padding/batch invariance on 4 items of mixed length
            probe = sorted(keep, key=lambda r: lens[r["item_id"]])
            probe = [probe[0], probe[len(probe) // 3], probe[2 * len(probe) // 3], probe[-1]]
            single = [reader.forward([store.pil(r["image_key"])], [r["body"]]) for r in probe]
            batchd = reader.forward([store.pil(r["image_key"]) for r in probe], [r["body"] for r in probe])
            h1 = np.concatenate([s.h for s in single])
            rel = float(np.abs(h1 - batchd.h).max() / (np.abs(h1).max() + 1e-6))
            lp1 = np.concatenate([s.logp[:, [c for c in cand_ids if c is not None]] for s in single])
            lpb = batchd.logp[:, [c for c in cand_ids if c is not None]]
            decision_single, decision_batch = lp1.copy(), lpb.copy()
            if task_type == "mc":
                for i, item in enumerate(probe):
                    decision_single[i, item["n_choices"]:] = -np.inf
                    decision_batch[i, item["n_choices"]:] = -np.inf
            same_pred = bool(np.all(np.nanargmax(decision_single, axis=1) == np.nanargmax(decision_batch, axis=1)))
            # bf16 + different SDPA kernel paths (unpadded bs=1 vs padded batch) give ~1e-2 relative noise in h; we require
            # identical candidate argmax and bounded noise, and record the raw numbers (plan §8: BF16 error is reported).
            dsrep["padding_invariance"] = {"items": [r["item_id"] for r in probe], "h_max_abs_diff_rel": rel,
                                           "cand_logp_max_abs_diff": float(np.abs(lp1 - lpb).max()),
                                           "cand_logp_mean_abs_diff": float(np.abs(lp1 - lpb).mean()),
                                           "same_candidate_argmax": same_pred, "ok": rel < 5e-2 and same_pred,
                                           "aux_single": [s.aux for s in single][:1], "aux_batch": batchd.aux}
            dsrep["padding_invariance"]["execution_scope"] = reader.describe().get("batch_execution", "native fixed-shape reader")
            # leakage: the prompt depends on body only -> identical ids when label field is changed
            r = probe[0]
            ids_a = reader.prompt_ids(r["body"])
            r2 = dict(r); r2["label"] = (r["label"] + 1) % max(2, r["n_choices"]); r2["label_str"] = "x"
            dsrep["label_independence"] = {"same_prompt_ids_after_label_change": reader.prompt_ids(r2["body"]) == ids_a,
                                           "label_str_in_body": r["label_str"] in r["body"].split() if task_type == "binary" else None}
            log(f"{ds}: tokens {dsrep['prompt_tokens']} excluded_long={dsrep['n_excluded_too_long']} padding={dsrep['padding_invariance']['ok']}")
        for split in args.splits:
            sub = [r for r in keep if r["split"] == split]
            if args.smoke:
                sub = sub[: args.smoke]
            if not sub:
                dsrep["splits"][split] = {"n": 0}
                continue
            (feat_root / ds).mkdir(parents=True, exist_ok=True)
            if (feat_root / ds / f"{split}.npz").exists() and not args.overwrite and not args.smoke:
                dsrep["splits"][split] = {**report["datasets"].get(ds, {}).get("splits", {}).get(split, {}), "n": len(sub), "status": "kept_existing"}
                log(f"{ds}/{split}: kept existing npz")
                continue
            try:
                acc = run_split(reader, ds, sub, store, cand_ids, task_type, args.batch, feat_root / ds / f"{split}.npz", log=log)
                dsrep["splits"][split] = {"n": len(sub), "ll_acc": acc, "status": "ok"}
            except Exception as e:  # noqa: BLE001
                dsrep["splits"][split] = {"n": len(sub), "status": "failed", "error": repr(e)}
                log(f"{ds}/{split} FAILED: {traceback.format_exc()}")
        # image-shuffle control on (a subset of) test
        test = [r for r in keep if r["split"] == "test"]
        if test and args.imgshuf and "test" in args.splits:
            rng = item_rng(ds, "imgshuf", "shared")  # same subset for every model
            sel_idx = rng.permutation(len(test))[: args.imgshuf]
            sel = [test[i] for i in sorted(sel_idx)]
            if args.smoke:
                sel = sel[: args.smoke]
            if (feat_root / ds / "test_imgshuf.npz").exists() and not args.overwrite and not args.smoke:
                dsrep["splits"]["test_imgshuf"] = {**report["datasets"].get(ds, {}).get("splits", {}).get("test_imgshuf", {}), "status": "kept_existing"}
                report["datasets"][ds] = dsrep
                dump_json(val_dir / f"{args.model}.json", report)
                continue
            try:
                mapping = derangement(sel, item_rng(ds, "imgshuf_perm"))
                acc = run_split(reader, ds, sel, store, cand_ids, task_type, args.batch, feat_root / ds / "test_imgshuf.npz",
                                image_override=mapping, log=log)
                dsrep["splits"]["test_imgshuf"] = {"n": len(sel), "ll_acc": acc, "status": "ok"}
            except Exception as e:  # noqa: BLE001
                dsrep["splits"]["test_imgshuf"] = {"status": "failed", "error": repr(e)}
                log(f"{ds}/imgshuf FAILED: {traceback.format_exc()}")
        report["datasets"][ds] = dsrep
        dump_json(val_dir / f"{args.model}.json", report)

    report["param_fingerprint_after"] = reader.param_fingerprint()
    report["params_unchanged"] = report["param_fingerprint_after"] == fp_before
    failures = extraction_failures(report, args.datasets, args.splits, bool(args.imgshuf)) if args.strict_validation else []
    report["validation_failures"] = failures
    report["status"] = "failed" if failures else "done"
    report["seconds"] = round(time.time() - t0)
    dump_json(val_dir / f"{args.model}.json", report)
    mm_path = out_root / "models_manifest.json"
    with (out_root / "models_manifest.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        mm = json.loads(mm_path.read_text()) if mm_path.exists() else {}
        mm[args.model] = reader.describe()
        dump_json(mm_path, mm)
    log(f"DONE in {report['seconds']}s params_unchanged={report['params_unchanged']}")
    if failures:
        raise RuntimeError("Validation failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
