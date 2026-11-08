"""Compare mathematically equivalent attention/padding paths on frozen GPU checkpoints.

Only four validation items per dataset are used; no probe is fitted and no main features are overwritten.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import pathlib
import sys

if not os.environ.get("SLURM_JOB_ID"):
    raise RuntimeError("Run numerical diagnostics inside a Slurm GPU allocation")

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from adapters import load_reader
from common import DATASETS, ImageStore, candidate_labels, dump_json, read_items, unique_item_ids


class PadTokenizer:
    """Add one masked pad token so even singleton inference uses an explicit padding mask."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, key):
        return getattr(self.tokenizer, key)

    def __call__(self, *args, **kwargs):
        encoded = self.tokenizer(*args, **kwargs)
        if kwargs.get("return_tensors") == "pt":
            ids = encoded["input_ids"]
            encoded["input_ids"] = torch.cat([ids, torch.full_like(ids[:, :1], self.pad_token_id)], dim=1)
            mask = encoded["attention_mask"]
            encoded["attention_mask"] = torch.cat([mask, torch.zeros_like(mask[:, :1])], dim=1)
        return encoded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    reader = load_reader(args.model, batch_size=8)
    tokenizer, model_forward = reader.tok, reader.model.forward
    report = {"model": args.model, "describe": reader.describe(), "modes": {}}
    for mode in ["native", "masked_pad", "eager_masked_pad"]:
        reader.tok = tokenizer if mode == "native" else PadTokenizer(tokenizer)
        reader.model.forward = functools.partial(model_forward, output_attentions=True) if mode == "eager_masked_pad" else model_forward
        results = {}
        for dataset, (_, task_type) in DATASETS.items():
            if dataset == "llavaguard_rating":
                continue
            items = unique_item_ids(read_items(dataset))
            items = [r for r in items if len(reader.prompt_ids(r["body"])) <= reader.max_text_tokens]
            items.sort(key=lambda r: len(reader.prompt_ids(r["body"])))
            selected = [items[0], items[len(items)//3], items[2*len(items)//3], items[-1]]
            labels = candidate_labels(task_type, max(r["n_choices"] for r in items) if task_type == "mc" else 2)
            candidate_ids = [c.token_id for c in reader.candidate_checks(labels, [r["body"] for r in selected])]
            store = ImageStore(dataset)
            singles = [reader.forward([store.pil(r["image_key"])], [r["body"]]) for r in selected]
            batch = reader.forward([store.pil(r["image_key"]) for r in selected], [r["body"] for r in selected])
            h = np.concatenate([s.h for s in singles])
            relative = float(np.abs(h-batch.h).max()/(np.abs(h).max()+1e-6))
            left = np.concatenate([s.logp[:, candidate_ids] for s in singles])
            right = batch.logp[:, candidate_ids]
            decision_left, decision_right = left.copy(), right.copy()
            if task_type == "mc":
                for i, item in enumerate(selected):
                    decision_left[i, item["n_choices"]:] = -np.inf
                    decision_right[i, item["n_choices"]:] = -np.inf
            pred_left, pred_right = decision_left.argmax(1), decision_right.argmax(1)
            same = bool(np.array_equal(pred_left, pred_right))
            results[dataset] = {"h_relative_difference": relative, "max_candidate_logp_difference": float(np.abs(left-right).max()),
                                "same_argmax": same, "ok": relative < 0.05 and same,
                                "item_ids": [r["item_id"] for r in selected],
                                "singleton_predictions": pred_left.tolist(), "batch_predictions": pred_right.tolist(),
                                "singleton_scores": left.tolist(), "batch_scores": right.tolist()}
            print(args.model, mode, dataset, f"h_rel={relative:.4f}", "same_argmax=", same, flush=True)
        report["modes"][mode] = {"all_ok": all(r["ok"] for r in results.values()), "datasets": results}
        dump_json(args.out / f"{args.model}.json", report)
    print(json.dumps({m:r["all_ok"] for m,r in report["modes"].items()}), flush=True)


if __name__ == "__main__":
    main()
