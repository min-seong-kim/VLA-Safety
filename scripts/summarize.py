"""Aggregate runs/<name>/summary.json files into one markdown table (stdout)."""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
names = sys.argv[1:] or sorted(p.parent.name for p in (ROOT / "runs").glob("*/summary.json"))

# (bench/template key, metric, short label)
COLS = [
    ("pope", "ll_acc", "POPE ll-acc"), ("pope", "ll_noeos_acc", "POPE ll-acc(noEOS)"), ("pope", "ll_f1_yes", "POPE ll-F1"),
    ("pope", "gen_acc", "POPE gen-acc"), ("pope", "gen_valid_rate", "POPE gen-valid"),
    ("mme", "ll_acc", "MME ll-acc"), ("mme", "ll_noeos_acc", "MME ll-acc(noEOS)"), ("mme", "gen_acc", "MME gen-acc"),
    ("scienceqa", "ll_acc_mean", "SQA ll-acc"), ("scienceqa", "gen_acc_exact", "SQA gen-acc"),
    ("vqav2", "nll_majority_per_token", "VQA NLL/tok"), ("vqav2", "gen_vqa_acc", "VQA gen-acc"),
    ("coco", "ref_nll_per_token", "COCO NLL/tok"), ("coco", "gen_cider", "COCO CIDEr"), ("coco", "gen_bleu4", "COCO BLEU4"),
]

summaries = {n: json.loads((ROOT / "runs" / n / "summary.json").read_text()) for n in names}
templates = sorted({k.split("/")[1] for s in summaries.values() for k in s["results"]})
for t in templates:
    print(f"\n### template = {t}\n")
    print("| model | " + " | ".join(c[2] for c in COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for n, s in summaries.items():
        row = []
        for bench, metric, _ in COLS:
            v = s["results"].get(f"{bench}/{t}", {}).get(metric)
            row.append("-" if v is None else f"{v:.3f}")
        print(f"| {n} | " + " | ".join(row) + " |")
