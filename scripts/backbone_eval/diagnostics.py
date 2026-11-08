"""Diagnostics beyond the main table (runs in openpi/.venv: needs sentencepiece + tokenizers, no torch).

Writes runs/backbone_eval_v1/diagnostics.md with, per model x dataset (test split):
  * candidate mass = sum of p(candidate tokens) under the model's own head, fraction of items whose full-vocab top-1
    is one of the candidates, and the most frequent top-1 tokens (decoded) -> shows what the language head actually emits
  * per-category accuracy (likelihood and probe) for MME / POPE / SQA / VQA / SIUO categories, with macro means
"""

from __future__ import annotations

import collections
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATASETS, ROOT, RUN_DIR, read_items, unique_item_ids  # noqa: E402

MODELS = ["pi0", "pi05", "paligemma", "paligemma_lb", "openvla", "prismatic", "openvla_oft"]


class Decoder:
    def __init__(self):
        import sentencepiece
        from tokenizers import Tokenizer

        self.sp = sentencepiece.SentencePieceProcessor(model_file=str(ROOT / "checkpoints/tokenizer/paligemma_tokenizer.model"))
        self.llama = Tokenizer.from_file(str(ROOT / "checkpoints/hf/openvla-7b/tokenizer.json"))

    def piece(self, model: str, tid: int) -> str:
        if model.startswith(("pi0", "paligemma")):
            return self.sp.id_to_piece(int(tid))
        return self.llama.id_to_token(int(tid)) or f"<{tid}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=pathlib.Path, default=RUN_DIR)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--unique-item-ids", action="store_true")
    args = ap.parse_args()
    run = args.run_dir
    dec = Decoder()
    out = ["# Diagnostics (test split)\n",
           "`cand_mass` = p(no)+p(yes) or sum over option letters under the model's own LM head (full-vocab softmax). "
           "`top1∈cands` = share of items whose most likely next token is one of the candidates. Llama action tokens are ids "
           "31744-31999 (OpenVLA bins); PaliGemma ids are listed by piece.\n"]
    out.append("## Language-head behaviour\n")
    out.append("| model | dataset | n | cand_mass mean | cand_mass median | top1∈cands | top1-decision acc | top-5 most frequent top-1 tokens (id:piece:count) |")
    out.append("|---|---|---|---|---|---|---|---|")
    out.insert(2, "`top1-decision acc` = accuracy when the full-vocab top-1 token itself is read as the answer, case-insensitively "
                  "('Yes'/'yes'/'▁yes' -> yes; a letter piece -> that option); any other token counts as wrong. Auxiliary only: the "
                  "main likelihood metric uses the fixed lowercase candidates from the plan.\n")
    percat = []
    for m in args.models:
        for ds, (dname, ttype) in DATASETS.items():
            f = run / "features" / m / ds / "test.npz"
            if not f.exists():
                continue
            z = np.load(f, allow_pickle=False)
            cand = [int(c) for c in z["cand_ids"] if c >= 0]
            top1 = z["top1_id"]
            in_c = float(np.mean(np.isin(top1, cand)))
            cnt = collections.Counter(top1.tolist()).most_common(5)
            toks = ", ".join(f"{t}:{dec.piece(m, t)!r}:{c}" for t, c in cnt)
            # read the top-1 token itself as the answer (case-insensitive)
            dec_pred = []
            for t in top1.tolist():
                p = dec.piece(m, t).replace("▁", "").strip().lower()
                if ttype == "binary":
                    dec_pred.append({"no": 0, "yes": 1}.get(p, -1))
                else:
                    dec_pred.append("abcde".index(p) if len(p) == 1 and p in "abcde" else -1)
            top1_acc = float(np.mean(np.array(dec_pred) == z["label"]))
            out.append(f"| {m} | {dname} | {len(top1)} | {float(np.mean(z['cand_mass'])):.3f} | {float(np.median(z['cand_mass'])):.3f} | {in_c:.2f} | {top1_acc:.3f} | {toks} |")
            # per-category
            pred_path = run / "predictions" / m / f"{ds}.jsonl"
            if pred_path.exists():
                preds = {json.loads(l)["item_id"]: json.loads(l) for l in pred_path.open()}
                rows = read_items(ds)
                if args.unique_item_ids:
                    rows = unique_item_ids(rows)
                cats = {r["item_id"]: r["category"] for r in rows if r["split"] == "test"}
                acc = collections.defaultdict(lambda: [[], []])
                for iid, p in preds.items():
                    c = cats.get(iid, "")
                    acc[c][0].append(p["pred_ll"] == p["label"])
                    acc[c][1].append(p["pred_probe"] == p["label"])
                if len(acc) > 1:
                    rows = sorted(acc.items())
                    percat.append((m, dname, rows))
    out.append("\n## Per-category accuracy (likelihood / probe), macro mean over categories\n")
    for m, dname, rows in percat:
        macro_ll = np.mean([np.mean(v[0]) for _, v in rows])
        macro_pr = np.mean([np.mean(v[1]) for _, v in rows])
        cells = ", ".join(f"{c or '(none)'}: {np.mean(v[0]):.2f}/{np.mean(v[1]):.2f} (n={len(v[0])})" for c, v in rows)
        out.append(f"- **{m} · {dname}** macro ll={macro_ll:.3f} probe={macro_pr:.3f} — {cells}")
    (run / "diagnostics.md").write_text("\n".join(out) + "\n")
    print("\n".join(out[:4] + out[5:5 + 20]))
    print(f"-> {run / 'diagnostics.md'}")


if __name__ == "__main__":
    main()
