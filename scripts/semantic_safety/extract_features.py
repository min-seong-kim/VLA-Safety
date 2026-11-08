"""Extract frozen VLA-prefix representations on robot semantic-safety scenes.

Run only inside a GPU allocation. No actions are executed, no model is updated,
and source labels/families are never passed to the model adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pathlib
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from backbone_eval.adapters import load_reader  # noqa: E402

CONDITIONS = ("query", "instruction", "neutral_query")


def select_smoke(rows: list[dict], limit: int) -> list[dict]:
    """Use complete safe/undesirable pairs from distinct scene groups."""
    selected = []
    for group in sorted({row["group_id"] for row in rows}):
        group_rows = [row for row in rows if row["group_id"] == group]
        pair = [next(row for row in group_rows if row["label"] == label) for label in (0, 1)]
        if len(selected) + 2 > limit:
            break
        selected.extend(pair)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("pi05", "pi05_libero", "openvla_oft"))
    parser.add_argument("--data", type=pathlib.Path, default=ROOT / "data_eval/asimov_robot_semantic_v1")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--smoke", type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Model inference requires a Slurm allocation")
    rows = [json.loads(line) for line in (args.data / "items.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row["semantic_primary"]]
    if args.smoke:
        rows = select_smoke(rows, args.smoke)
    if not rows or len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError("Empty or duplicate evaluation items")
    images = pq.read_table(args.data / "images.parquet").to_pydict()
    image_bytes = dict(zip(images["image_key"], images["bytes"], strict=True))
    destination = args.out / "features" / args.model
    destination.mkdir(parents=True, exist_ok=False)
    start = time.time()
    reader = load_reader(args.model, batch_size=1)
    fingerprint_before = reader.param_fingerprint()
    manifest = {"model": args.model, "reader": reader.describe(), "job_id": os.environ["SLURM_JOB_ID"],
                "conditions": list(CONDITIONS), "n_items": len(rows), "n_groups": len({r["group_id"] for r in rows}),
                "data_manifest_sha256": hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest(),
                "source_scope": "Robot-scene ASIMOV semantic research subset; generic VLM chat datasets excluded",
                "representation": "final-normalized last valid prompt token; VLM-only prefix inside VLA",
                "execution": "singleton forwards; original checkpoint and existing adapters frozen",
                "head_check": reader.check_head(), "attention_check": reader.check_attention_direction(),
                "parameter_fingerprint_before": fingerprint_before}
    if manifest["head_check"].get("ok") is False or manifest["attention_check"].get("ok") is False:
        raise ValueError("Head or attention validation failed")
    for condition in CONDITIONS:
        bodies = [row["instruction_body"] if condition == "instruction" else row["body"] for row in rows]
        checks = reader.candidate_checks(["no", "yes"], bodies[:5])
        if not all(check.ok for check in checks):
            raise ValueError(f"Candidates are not valid single tokens: {checks}")
        candidate_ids = [check.token_id for check in checks]
        hidden, logps, entropy, lengths, top1, mass = [], [], [], [], [], []
        for index, (row, body) in enumerate(zip(rows, bodies, strict=True)):
            if len(reader.prompt_ids(body)) > reader.max_text_tokens:
                raise ValueError(f"Prompt exceeds model context; no silent truncation: {row['item_id']}")
            image = Image.open(io.BytesIO(image_bytes[row["image_key"]])).convert("RGB")
            if condition == "neutral_query":
                image = Image.new("RGB", image.size, (127, 127, 127))
            output = reader.forward([image], [body])
            if not np.isfinite(output.h).all() or not np.isfinite(output.logp[:, candidate_ids]).all():
                raise ValueError("Non-finite feature or candidate score")
            if index == 0:
                repeated = reader.forward([image], [body])
                relative = float(np.max(np.abs(output.h - repeated.h)) / (np.max(np.abs(output.h)) + 1e-8))
                if relative > 1e-5:
                    raise ValueError(f"Singleton repeat changed hidden states: {relative}")
                manifest.setdefault("repeat_checks", {})[condition] = {"hidden_max_relative": relative}
            hidden.append(output.h[0])
            logps.append(output.logp[0, candidate_ids])
            probability = np.exp(output.logp[0].astype(np.float64))
            entropy.append(float(-np.sum(probability * output.logp[0])))
            lengths.append(int(output.n_prompt_tokens[0]))
            top1.append(int(np.argmax(output.logp[0])))
            mass.append(float(np.exp(output.logp[0, candidate_ids]).sum()))
            if index % 40 == 0:
                print(f"{args.model} {condition} {index}/{len(rows)} elapsed={time.time()-start:.1f}s", flush=True)
        np.savez(destination / f"{condition}.npz", h=np.asarray(hidden, dtype=np.float32),
                 cand_logp=np.asarray(logps, dtype=np.float32), entropy=np.asarray(entropy),
                 label=np.array([row["label"] for row in rows], dtype=np.int32),
                 item_id=np.array([row["item_id"] for row in rows]),
                 group_id=np.array([row["group_id"] for row in rows]),
                 family=np.array([row["family"] for row in rows]),
                 context=np.array([row["context"] for row in rows]),
                 instruction=np.array([row["instruction"] for row in rows]),
                 n_prompt_tokens=np.asarray(lengths), top1_id=np.asarray(top1), candidate_mass=np.asarray(mass),
                 candidate_ids=np.asarray(candidate_ids))
    fingerprint_after = reader.param_fingerprint()
    if fingerprint_before != fingerprint_after:
        raise ValueError("Model parameter fingerprint changed during extraction")
    manifest.update(parameter_fingerprint_after=fingerprint_after, seconds=time.time()-start, status="complete")
    with (destination / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    print(f"EXTRACTION_COMPLETE {args.model} items={len(rows)} seconds={time.time()-start:.1f}", flush=True)


if __name__ == "__main__":
    main()
