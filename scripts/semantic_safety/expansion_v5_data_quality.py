"""Audit recorded policy inputs and provide a review of actual training trajectories."""

from collections import defaultdict
import html
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from rollout_common import ROOT, write_json


def audit_episode(row, data, spec, replay):
    steps = row["steps"]
    folder = (ROOT / row["path"]).parent
    summary = json.loads((folder / "summary.json").read_text())
    metric = replay["final_metric"]
    assert replay["independent_environment"] and replay["steps"] == steps
    assert replay["max_sim_state_abs_error"] < 1e-6
    assert metric["proxy_success"] and metric["stable_steps"] >= 6
    assert metric["ever_grasped"] and metric["ever_lifted"] and metric["verified_transport"]
    assert summary["passed"] and summary["physics_stable"] and summary["steps"] == steps
    assert summary["all_actions_physical"] and not summary["teleport_during_recording"]
    assert bool(data.attrs["teacher_success"])
    assert spec["condition_index"] == row["condition_index"] and spec["item_id"] == row["item_id"]
    assert spec["pair_group"] == row["pair_group"]
    assert spec["instruction"] == row["instruction"]
    assert steps >= 10 and data["actions"].shape == (steps, 7)
    actions = data["actions"][:]
    assert np.isfinite(actions).all() and np.max(np.abs(actions[:, :6])) <= 1.000001
    assert np.isin(actions[:, 6], [-1., 1.]).all()
    assert data["observations/state"].shape == (steps + 1, 8)
    assert data["observations/sim_state"].shape[0] == steps + 1
    assert data["teacher_stage"].shape == (steps,)
    for key in ["observations/state", "observations/sim_state"]:
        assert np.isfinite(data[key][:]).all()
    camera_spread = {}
    for camera in ["main_rgb_raw", "wrist_rgb_raw"]:
        frames = data["observations/" + camera]
        assert frames.shape == (steps + 1, 224, 224, 3) and frames.dtype == np.uint8
        spread = [float(np.std(frames[index])) for index in [0, steps // 2, steps]]
        assert max(spread) > 1., (row["item_id"], camera, "blank sampled observations")
        camera_spread[camera] = spread
    return dict(item_id=row["item_id"], condition_index=row["condition_index"], steps=steps,
        passed=True, replay_error=replay["max_sim_state_abs_error"],
        stable_destination_steps=metric["stable_steps"], sampled_camera_standard_deviations=camera_spread)


def write_training_review(data_root, rows, definitions):
    """Show real training frames, including each condition's last accepted layout."""
    by_condition = defaultdict(list)
    for row in rows:
        by_condition[row["condition_index"]].append(row)
    folder = data_root / "quality_review"
    folder.mkdir(exist_ok=True)
    chosen = []
    cards = []
    for index in range(0, len(definitions), 2):
        cells = []
        for definition in definitions[index:index + 2]:
            condition = definition["condition_index"]
            row = sorted(by_condition[condition], key=lambda value: value["item_id"])[-1]
            with h5py.File(ROOT / row["path"]) as data:
                frames = data["observations/main_rgb_raw"]
                indices = [0, row["steps"] // 2, row["steps"]]
                sheet = Image.new("RGB", (672, 246), "white")
                draw = ImageDraw.Draw(sheet)
                for column, step in enumerate(indices):
                    sheet.paste(Image.fromarray(np.ascontiguousarray(frames[step][::-1, ::-1])), (224 * column, 0))
                    draw.text((224 * column + 6, 226), f"control step {step}", fill="black")
            filename = f"c{condition:03d}.jpg"
            sheet.save(folder / filename, quality=90)
            relative = (ROOT / row["path"]).relative_to(data_root)
            chosen.append(dict(condition_index=condition, item_id=row["item_id"], path=row["path"],
                image=str((folder / filename).relative_to(ROOT)), sampled_control_steps=indices))
            cells.append(f'<div><b>{html.escape(definition["side"])} · {html.escape(row["item_id"])}</b>'
                f'<p>{html.escape(row["instruction"])}</p><a href="quality_review/{filename}">'
                f'<img loading="lazy" src="quality_review/{filename}" alt="initial middle final training frames"></a>'
                f'<p><a href="{relative}">실제 학습 trajectory HDF5</a></p></div>')
        cards.append(f'<article><h2>상황 {index // 2 + 1}</h2><section>{"".join(cells)}</section></article>')
    (data_root / "training_quality_review.html").write_text(
        '<!doctype html><html lang="ko"><meta charset="utf-8"><title>v5 학습 시연 품질 검토</title>'
        '<style>body{font-family:sans-serif;margin:24px;background:#f5f5f5}article{background:white;padding:16px;margin:16px 0}'
        'section{display:grid;grid-template-columns:1fr 1fr;gap:20px}img{width:100%}'
        '@media(max-width:850px){section{grid-template-columns:1fr}}</style>'
        '<h1>실제 학습 시연 검토</h1><p>각 조건에서 마지막으로 채택한 학습 배치의 시작·중간·종료 관찰이다. '
        '안전·위험 조건의 채택 배치 번호는 다를 수 있다. 이 표는 표본 검토용이며 전체 영상을 사람이 검수했다는 뜻은 아니다.</p>'
        + ''.join(cards) + '</html>')
    write_json(folder / "selection.json", dict(method="last accepted training layout in each condition",
        conditions=len(chosen), all_episodes_manually_reviewed=False, examples=chosen))
