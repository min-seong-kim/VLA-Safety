"""Download VLM benchmarks from Hugging Face and store them as 224x224 (resize-with-pad) PNGs + JSONL.

Output layout:  data/<bench>/items.jsonl, data/<bench>/images/<idx>.png

The resize matches openpi.shared.image_tools.resize_with_pad (keep aspect ratio, pad with black,
centered) so the images go through exactly the preprocessing the VLA saw at training time.
"""

import argparse
import json
import pathlib

import datasets
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = 224


def resize_with_pad(img: Image.Image, size: int = RES) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    ratio = max(w / size, h / size)
    nh, nw = int(h / ratio), int(w / ratio)
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def write(bench: str, rows, limit: int | None):
    out = ROOT / "data" / bench
    (out / "images").mkdir(parents=True, exist_ok=True)
    n = 0
    with (out / "items.jsonl").open("w") as f:
        for i, item in enumerate(rows):
            if limit is not None and i >= limit:
                break
            img = item.pop("image")
            rel = f"images/{i:06d}.png"
            resize_with_pad(img).save(out / rel)
            item["image"] = rel
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
    print(f"{bench}: wrote {n} items to {out}", flush=True)


def pope(limit):
    ds = datasets.load_dataset("lmms-lab/POPE", split="test", streaming=True)
    for r in ds:
        yield {"id": str(r["question_id"]), "task": "yesno", "question": r["question"],
               "answers": [r["answer"].strip().lower()], "category": r.get("category", ""), "image": r["image"]}


def mme(limit):
    ds = datasets.load_dataset("lmms-lab/MME", split="test", streaming=True)
    for r in ds:
        yield {"id": str(r["question_id"]), "task": "yesno", "question": r["question"],
               "answers": [r["answer"].strip().lower()], "category": r["category"], "image": r["image"]}


def coco(limit):
    ds = datasets.load_dataset("lmms-lab/COCO-Caption2017", split="val", streaming=True)
    for r in ds:
        yield {"id": str(r["question_id"]), "task": "caption", "question": "",
               "answers": [a.strip() for a in r["answer"]], "category": "", "image": r["image"]}


def vqav2(limit):
    ds = datasets.load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
    for r in ds:
        yield {"id": str(r["question_id"]), "task": "vqa", "question": r["question"],
               "answers": [a["answer"] for a in r["answers"]], "majority": r["multiple_choice_answer"],
               "category": r["answer_type"], "image": r["image"]}


def scienceqa(limit):
    ds = datasets.load_dataset("lmms-lab/ScienceQA-IMG", split="test", streaming=True)
    for i, r in enumerate(ds):
        q = r["question"]
        if r.get("hint"):
            q = f"{r['hint']} {q}"
        yield {"id": str(i), "task": "mc", "question": q, "choices": r["choices"], "answer_idx": int(r["answer"]),
               "answers": [r["choices"][int(r["answer"])]], "category": r["subject"], "image": r["image"]}


BENCHES = {"pope": pope, "mme": mme, "coco": coco, "vqav2": vqav2, "scienceqa": scienceqa}
DEFAULT_LIMITS = {"pope": None, "mme": None, "coco": 2000, "vqav2": 3000, "scienceqa": None}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", nargs="+", default=list(BENCHES))
    ap.add_argument("--limit", type=int, default=None, help="override per-bench default limits")
    args = ap.parse_args()
    for b in args.benches:
        lim = args.limit if args.limit is not None else DEFAULT_LIMITS[b]
        write(b, BENCHES[b](lim), lim)
