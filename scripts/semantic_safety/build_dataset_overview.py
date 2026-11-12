#!/usr/bin/env python3
"""Build a standalone, offline atlas of the generated semantic-safety datasets."""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import io
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
MODEL_NAMES = {
    "pi0_libero": "π0 · LIBERO",
    "pi05_libero": "π0.5 · LIBERO",
    "openvla_libero_spatial": "OpenVLA · LIBERO-Spatial",
    "openvla_oft_libero_spatial": "OpenVLA-OFT · LIBERO-Spatial",
}
NATIVE_LOCATIONS = [
    "접시와 라메킨 사이",
    "라메킨 옆",
    "테이블 중앙",
    "쿠키 상자 위",
    "나무 수납장의 맨 위 서랍 안",
    "라메킨 위",
    "쿠키 상자 옆",
    "스토브 위",
    "접시 옆",
    "나무 수납장 위",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def embed_image(path: Path, *, thumbnail: bool = False) -> str:
    if thumbnail:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((384, 384))
            buffer = io.BytesIO()
            image.save(buffer, format="WEBP", quality=82)
        raw, mime = buffer.getvalue(), "image/webp"
    else:
        raw, mime = path.read_bytes(), "image/png"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def native_smoke_payload() -> dict:
    root = ROOT / "runs/native_rollout_smoke_v1"
    tasks = {}
    totals = {}
    for model in MODEL_NAMES:
        successes = 0
        successes_by_160 = 0
        episodes = 0
        for path in sorted((root / model / "episodes").glob("*/summary.json")):
            result = read_json(path)
            meta = read_json(path.with_name("episode.json"))
            task_id = meta["task_id"]
            if task_id not in tasks:
                tasks[task_id] = {
                    "task_id": task_id,
                    "bowl_location_ko": NATIVE_LOCATIONS[task_id],
                    "instruction": meta["instruction"],
                    "source_bddl": meta["source_bddl"],
                    "image": embed_image(root / "pi0_libero" / "episodes" /
                                         f"native_task{task_id:02d}_init00" / "initial.png"),
                    "results": {key: [] for key in MODEL_NAMES},
                }
            tasks[task_id]["results"][model].append({
                "initial_state_id": meta["initial_state_id"],
                "success": result["task_success"],
                "steps": result["steps"],
                "episode_id": result["episode_id"],
            })
            episodes += 1
            successes += int(result["task_success"])
            successes_by_160 += int(result["task_success"] and result["steps"] <= 160)
            assert meta["horizon"] == 220
        totals[model] = {"episodes": episodes, "successes_by_220": successes,
                         "successes_by_160": successes_by_160}
    assert len(tasks) == 10
    assert all(row["episodes"] == 16 for row in totals.values())
    return {"suite": "libero_spatial", "tasks": [tasks[key] for key in sorted(tasks)],
            "totals": totals, "original_horizon": 220, "comparison_cutoff": 160,
            "cutoff_source": "Success timestamps in existing recordings; no new policy runs"}


def build_payload() -> dict:
    v1_root = ROOT / "data/libero_semantic_explicit_v1"
    v2_root = ROOT / "data/semantic_safety_catalog_v2"
    v3_root = ROOT / "data/semantic_rollout_v3"
    run_root = ROOT / "runs/semantic_rollouts_v3"
    v1_summary = read_json(v1_root / "summary.json")
    v2_summary = read_json(v2_root / "summary.json")
    v1_rows = read_jsonl(v1_root / "items.jsonl")
    config = read_json(v1_root / "generation.json")["configuration"]
    titles = {row["id"]: row["title_ko"] for row in config["scenarios"]}
    groups = defaultdict(list)
    for row in v1_rows:
        groups[row["family_id"]].append(row)
    representatives = []
    for family, rows in sorted(groups.items()):
        group_id = rows[0]["group_id"]
        selected = [row for row in rows if row["group_id"] == group_id]
        views = {row["view_id"]: row["image"] for row in selected}
        representatives.append({
            "family_id": family,
            "title_ko": titles[family],
            "group_id": group_id,
            "split": selected[0]["split"],
            "views": {view: embed_image(v1_root / path, thumbnail=True) for view, path in views.items()},
            "safe": sorted({row["instruction"] for row in selected if not row["task_is_unsafe"]}),
            "unsafe": sorted({row["instruction"] for row in selected if row["task_is_unsafe"]}),
        })
    assert len(v1_rows) == v1_summary["records"]
    assert dict(Counter(row["safety_label"] for row in v1_rows)) == v1_summary["labels"]

    scenes = read_jsonl(v3_root / "scenes.jsonl")
    catalog = read_jsonl(v2_root / "scenario_pairs.jsonl")
    quality = read_json(run_root / "quality_review.json")
    audit = read_json(run_root / "audit_summary.json")
    excluded = quality["passive_stability"]["flagged_families"]
    all_summaries = []
    for scene in scenes:
        family = scene["family_id"]
        scene["images"] = {
            view: embed_image(v3_root / "observations" / family / f"{view}.png")
            for view in ("agentview", "wrist")
        }
        scene["excluded"] = family in excluded
        scene["rollouts"] = []
        for model in MODEL_NAMES:
            for label in ("safe", "unsafe"):
                episode_id = f"{family}__{label}__seed0"
                summary = read_json(run_root / model / "episodes" / episode_id / "summary.json")
                assert summary["status"] == "complete"
                scene["rollouts"].append({
                    "model": model,
                    "label": label,
                    "steps": summary["steps"],
                    "task_success": summary["task_success"],
                    "risk_proxy_ever": summary["risk_proxy_ever"],
                    "episode_id": episode_id,
                })
                all_summaries.append(summary)
    assert len(all_summaries) == audit["audited_complete_episodes"]
    assert sum(row["steps"] for row in all_summaries) == audit["timestep_records"]
    assert all(row["same_initial_image_and_state"] for row in audit["pairing"].values())
    assert len(catalog) == v2_summary["contrast_pairs"]

    matrices = []
    for path in sorted((run_root / "probe_matrices").glob("*.json")):
        meta = read_json(path)
        filtered = next(row for row in quality["filtered_matrices"] if Path(row["file"]).stem == path.stem)
        matrices.append({
            key: meta[key] for key in ("model", "feature", "region", "layer", "forward_call", "fold")
        } | {
            "shape": meta["matrix_shape"],
            "filtered_shape": filtered["shape"],
            "path": str(path.with_suffix(".npz").relative_to(ROOT)),
            "filtered_path": filtered["file"],
            "partitions": dict(Counter(row["partition"] for row in meta["rows"])),
            "filtered_partitions": filtered["partitions"],
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schema": "semantic_safety_dataset_atlas_v1",
        "v1": {"summary": v1_summary, "representatives": representatives},
        "v2": {"summary": v2_summary, "pairs": catalog},
        "scenes": scenes,
        "models": MODEL_NAMES,
        "splits": read_json(v3_root / "probe_splits.json")["folds"],
        "matrices": matrices,
        "quality": quality,
        "behavior": read_json(run_root / "safe_task_execution_breakdown.json"),
        "native_smoke": native_smoke_payload(),
        "audit": {key: audit[key] for key in ("status", "audited_complete_episodes", "timestep_records")},
        "source_paths": [
            "data/libero_semantic_explicit_v1/summary.json",
            "data/libero_semantic_explicit_v1/items.jsonl",
            "data/semantic_safety_catalog_v2/summary.json",
            "data/semantic_safety_catalog_v2/scenario_pairs.jsonl",
            "data/semantic_rollout_v3/scenes.jsonl",
            "data/semantic_rollout_v3/probe_splits.json",
            "runs/semantic_rollouts_v3/audit_summary.json",
            "runs/semantic_rollouts_v3/quality_review.json",
            "runs/semantic_rollouts_v3/safe_task_execution_breakdown.json",
            "runs/native_rollout_smoke_v1/<model>/episodes/<episode>/episode.json",
            "runs/native_rollout_smoke_v1/<model>/episodes/<episode>/summary.json",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/semantic_safety_overview.html")
    args = parser.parse_args()
    payload = build_payload()
    template = Path(__file__).with_name("dataset_overview.html").read_text(encoding="utf-8")
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    assert template.count("__DATASET_PAYLOAD__") == 1
    document = template.replace("__DATASET_PAYLOAD__", serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "v3_scenes": len(payload["scenes"]),
        "v3_embedded_original_pngs": len(payload["scenes"]) * 2,
        "v1_representative_views": sum(len(row["views"]) for row in payload["v1"]["representatives"]),
        "v2_blueprints": len(payload["v2"]["pairs"]),
        "episodes": payload["audit"]["audited_complete_episodes"],
        "timestep_records": payload["audit"]["timestep_records"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
