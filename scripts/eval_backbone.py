"""Evaluate the VLM backbone of an openpi checkpoint on VLM benchmarks prepared by prepare_data.py.

Two families of metrics are reported for every benchmark so that a model whose free-form text
generation is broken by action training can still be measured:

  * likelihood-based (no generation needed): yes/no and multiple-choice decided by comparing the
    log-probability of the candidate answers; caption NLL per token on reference captions.
  * generation-based: greedy decoding, scored with the usual metric (accuracy / VQA acc / CIDEr).
"""

import argparse
import collections
import json
import pathlib
import re
import string
import time

import numpy as np
from PIL import Image

from vlm_backbone import BackboneLM

ROOT = pathlib.Path(__file__).resolve().parents[1]

TEMPLATES = {
    # PaliGemma's native pre-training prompt format.
    "paligemma": {"vqa": "answer en {q}", "yesno": "answer en {q}", "mc": "answer en {q}", "caption": "caption en"},
    # Plain natural-language prompt (closer to what the VLA saw: a bare instruction).
    "plain": {"vqa": "{q}", "yesno": "{q}", "mc": "{q}", "caption": "Describe the image."},
    # Same as paligemma but drops MME's trailing instruction, which PaliGemma-pt never saw.
    "paligemma_strip": {"vqa": "answer en {q}", "yesno": "answer en {q}", "mc": "answer en {q}", "caption": "caption en",
                        "strip": "Please answer yes or no."},
}


def load_items(bench: str, limit: int | None):
    d = ROOT / "data" / bench
    items = [json.loads(l) for l in (d / "items.jsonl").open()]
    if limit:
        items = items[:limit]
    return d, items


def load_images(d: pathlib.Path, items):
    return np.stack([np.asarray(Image.open(d / it["image"]).convert("RGB"), dtype=np.uint8) for it in items])


def batches(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


_ARTICLES = {"a", "an", "the"}


def norm_answer(s: str) -> str:
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(w for w in s.split() if w not in _ARTICLES)


def tok_caption(s: str) -> str:
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def parse_yesno(s: str) -> str:
    m = re.match(r"\W*(yes|no)\b", s.lower())
    return m.group(1) if m else "invalid"


# ------------------------------------------------------------------------------------------ tasks
def run_yesno(lm, d, items, tmpl, bs, gen):
    recs = []
    for chunk in batches(items, bs):
        imgs = load_images(d, chunk)
        prompts = [tmpl["yesno"].format(q=it["question"].replace(tmpl.get("strip", "\0"), "").strip()) for it in chunk]
        # likelihood of "yes" vs "no"
        # with EOS: p("yes"<eos>) vs p("no"<eos>); without EOS: p("yes") vs p("no") as lm-eval-harness does
        s_yes, _ = lm.score(imgs, prompts, ["yes"] * len(chunk))
        s_no, _ = lm.score(imgs, prompts, ["no"] * len(chunk))
        s_yes0, _ = lm.score(imgs, prompts, ["yes"] * len(chunk), add_eos=False)
        s_no0, _ = lm.score(imgs, prompts, ["no"] * len(chunk), add_eos=False)
        outs = lm.generate(imgs, prompts, max_new_tokens=4) if gen else [""] * len(chunk)
        for it, a, b, a0, b0, o in zip(chunk, s_yes, s_no, s_yes0, s_no0, outs, strict=True):
            recs.append({"id": it["id"], "gold": it["answers"][0], "logp_yes": float(a), "logp_no": float(b),
                         "logp_yes_noeos": float(a0), "logp_no_noeos": float(b0),
                         "pred_ll": "yes" if a > b else "no", "pred_ll_noeos": "yes" if a0 > b0 else "no",
                         "gen": o, "pred_gen": parse_yesno(o), "category": it.get("category", "")})
    gold = np.array([r["gold"] for r in recs])
    pll = np.array([r["pred_ll"] for r in recs])
    pll0 = np.array([r["pred_ll_noeos"] for r in recs])
    pgen = np.array([r["pred_gen"] for r in recs])
    tp = np.sum((pll == "yes") & (gold == "yes"))
    prec = tp / max(np.sum(pll == "yes"), 1)
    rec = tp / max(np.sum(gold == "yes"), 1)
    m = {
        "n": len(recs),
        "ll_acc": float(np.mean(pll == gold)),
        "ll_f1_yes": float(2 * prec * rec / max(prec + rec, 1e-9)),
        "ll_yes_rate": float(np.mean(pll == "yes")),
        "ll_noeos_acc": float(np.mean(pll0 == gold)),
        "ll_noeos_yes_rate": float(np.mean(pll0 == "yes")),
        "gold_yes_rate": float(np.mean(gold == "yes")),
        "gen_acc": float(np.mean(pgen == gold)) if gen else None,
        "gen_valid_rate": float(np.mean(pgen != "invalid")) if gen else None,
    }
    cats = collections.defaultdict(list)
    for r in recs:
        cats[r["category"]].append(r["pred_ll"] == r["gold"])
    m["ll_acc_by_category"] = {k: float(np.mean(v)) for k, v in cats.items()}
    return m, recs


def run_mc(lm, d, items, tmpl, bs, gen):
    recs = []
    for chunk in batches(items, bs):
        imgs = load_images(d, chunk)
        prompts = [tmpl["mc"].format(q=it["question"]) for it in chunk]
        # flatten (item, choice) pairs
        flat_img, flat_p, flat_a, owner = [], [], [], []
        for j, it in enumerate(chunk):
            for c in it["choices"]:
                flat_img.append(imgs[j]); flat_p.append(prompts[j]); flat_a.append(c); owner.append(j)
        sums, lens = [], []
        for k in range(0, len(flat_p), bs):
            s, n = lm.score(np.stack(flat_img[k : k + bs]), flat_p[k : k + bs], flat_a[k : k + bs])
            sums.extend(s); lens.extend(n)
        outs = lm.generate(imgs, prompts, max_new_tokens=16) if gen else [""] * len(chunk)
        per = collections.defaultdict(list)
        for o_idx, s, n in zip(owner, sums, lens, strict=True):
            per[o_idx].append((float(s), int(n)))
        for j, it in enumerate(chunk):
            scores = per[j]
            pred_sum = int(np.argmax([s for s, _ in scores]))
            pred_mean = int(np.argmax([s / max(n, 1) for s, n in scores]))
            recs.append({"id": it["id"], "gold": it["answer_idx"], "pred_sum": pred_sum, "pred_mean": pred_mean,
                         "scores": scores, "gen": outs[j], "n_choices": len(it["choices"]),
                         "category": it.get("category", "")})
    gold = np.array([r["gold"] for r in recs])
    m = {
        "n": len(recs),
        "ll_acc_sum": float(np.mean(np.array([r["pred_sum"] for r in recs]) == gold)),
        "ll_acc_mean": float(np.mean(np.array([r["pred_mean"] for r in recs]) == gold)),
        "chance": float(np.mean([1 / r["n_choices"] for r in recs])),
    }
    if gen:
        hits = [norm_answer(r["gen"]) == norm_answer(items[i]["choices"][r["gold"]]) for i, r in enumerate(recs)]
        m["gen_acc_exact"] = float(np.mean(hits))
    return m, recs


def run_vqa(lm, d, items, tmpl, bs, gen):
    recs = []
    for chunk in batches(items, bs):
        imgs = load_images(d, chunk)
        prompts = [tmpl["vqa"].format(q=it["question"]) for it in chunk]
        s, n = lm.score(imgs, prompts, [it["majority"] for it in chunk])
        outs = lm.generate(imgs, prompts, max_new_tokens=8) if gen else [""] * len(chunk)
        for it, si, ni, o in zip(chunk, s, n, outs, strict=True):
            golds = [norm_answer(a) for a in it["answers"]]
            acc = min(1.0, sum(g == norm_answer(o) for g in golds) / 3.0) if gen else None
            recs.append({"id": it["id"], "majority": it["majority"], "nll_majority": float(-si / max(ni, 1)),
                         "gen": o, "vqa_acc": acc, "category": it.get("category", "")})
    m = {"n": len(recs), "nll_majority_per_token": float(np.mean([r["nll_majority"] for r in recs]))}
    if gen:
        m["gen_vqa_acc"] = float(np.mean([r["vqa_acc"] for r in recs]))
        cats = collections.defaultdict(list)
        for r in recs:
            cats[r["category"]].append(r["vqa_acc"])
        m["gen_vqa_acc_by_type"] = {k: float(np.mean(v)) for k, v in cats.items()}
    return m, recs


def run_caption(lm, d, items, tmpl, bs, gen):
    recs = []
    for chunk in batches(items, bs):
        imgs = load_images(d, chunk)
        prompts = [tmpl["caption"]] * len(chunk)
        # NLL of every reference caption (up to 5 per image), flattened
        flat_img, flat_p, flat_a, owner = [], [], [], []
        for j, it in enumerate(chunk):
            for ref in it["answers"][:5]:
                flat_img.append(imgs[j]); flat_p.append(prompts[j]); flat_a.append(ref); owner.append(j)
        nll = collections.defaultdict(list)
        for k in range(0, len(flat_p), bs):
            s, n = lm.score(np.stack(flat_img[k : k + bs]), flat_p[k : k + bs], flat_a[k : k + bs])
            for o_idx, si, ni in zip(owner[k : k + bs], s, n, strict=True):
                nll[o_idx].append(float(-si / max(ni, 1)))
        outs = lm.generate(imgs, prompts, max_new_tokens=32) if gen else [""] * len(chunk)
        for j, it in enumerate(chunk):
            recs.append({"id": it["id"], "nll_per_token": float(np.mean(nll[j])), "gen": outs[j], "refs": it["answers"][:5]})
    m = {"n": len(recs), "ref_nll_per_token": float(np.mean([r["nll_per_token"] for r in recs]))}
    if gen:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider

        gts = {r["id"]: [tok_caption(x) for x in r["refs"]] for r in recs}
        res = {r["id"]: [tok_caption(r["gen"]) or "empty"] for r in recs}
        m["gen_cider"] = float(Cider().compute_score(gts, res)[0])
        m["gen_bleu4"] = float(Bleu(4).compute_score(gts, res, verbose=0)[0][3])
        m["gen_mean_len_words"] = float(np.mean([len(r["gen"].split()) for r in recs]))
        m["gen_empty_rate"] = float(np.mean([len(r["gen"].strip()) == 0 for r in recs]))
    return m, recs


RUNNERS = {"yesno": run_yesno, "mc": run_mc, "vqa": run_vqa, "caption": run_caption}
BENCH_TASK = {"pope": "yesno", "mme": "yesno", "scienceqa": "mc", "vqav2": "vqa", "coco": "caption"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run name, e.g. pi0_base")
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--pi05", action="store_true")
    ap.add_argument("--benches", nargs="+", default=list(BENCH_TASK))
    ap.add_argument("--templates", nargs="+", default=["paligemma"], choices=list(TEMPLATES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--no_gen", action="store_true")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--tokenizer", default=str(ROOT / "checkpoints/tokenizer/paligemma_tokenizer.model"))
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir or ROOT / "runs" / args.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lm = BackboneLM(args.ckpt_dir, pi05=args.pi05, tokenizer_path=args.tokenizer)
    print(f"[{args.name}] model loaded in {time.time() - t0:.0f}s", flush=True)

    # Merge into an existing summary so partial re-runs (e.g. only some benches) keep earlier results.
    summary = {"name": args.name, "ckpt_dir": args.ckpt_dir, "pi05": args.pi05, "limit": args.limit, "results": {}}
    if (out_dir / "summary.json").exists() and not args.limit:
        summary["results"] = json.loads((out_dir / "summary.json").read_text()).get("results", {})
    for tname in args.templates:
        for bench in args.benches:
            d, items = load_items(bench, args.limit)
            t1 = time.time()
            metrics, recs = RUNNERS[BENCH_TASK[bench]](lm, d, items, TEMPLATES[tname], args.batch, not args.no_gen)
            metrics["seconds"] = round(time.time() - t1, 1)
            summary["results"][f"{bench}/{tname}"] = metrics
            with (out_dir / f"{bench}.{tname}.jsonl").open("w") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[{args.name}] {bench}/{tname}: {json.dumps(metrics)}", flush=True)
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{args.name}] DONE in {time.time() - t0:.0f}s", flush=True)
