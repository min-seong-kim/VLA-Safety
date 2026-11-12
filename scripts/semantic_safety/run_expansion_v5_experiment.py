"""Gate full data collection and three-scope training on physical replay validation."""

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from rollout_common import ROOT, write_json

DATA = ROOT / "data/semantic_expansion_v5_lora_v1"
RUN = ROOT / "runs/semantic_expansion_v5_lora_v1"
RENDER = str(ROOT / "envs/libero-render/bin/python")
PYTHON = str(ROOT / "openpi/.venv/bin/python")


def fingerprint(paths):
    return {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def data_fingerprint(tag):
    scripts=ROOT/"scripts/semantic_safety"
    return fingerprint([DATA/"combined_train_manifest.jsonl",DATA/"training_data_audit.json",
        DATA/"conditions.jsonl",DATA/"design.json",DATA/"assets/manifest.json",
        DATA/tag/"train_manifest.jsonl",
        *[DATA/"holdouts"/(name+".jsonl") for name in ["selection","confirmation","test"]],
        *[scripts/name for name in ["expansion_v5_motor.py","expansion_v5_teacher.py",
            "collect_expansion_v5_demonstrations.py","prepare_expansion_v5_lora_data.py",
            "expansion_v5_data_quality.py"]]])


def command(arguments, label):
    write_json(RUN / "experiment_status.json", dict(state="running", phase=label,
        updated_unix=time.time(), command=arguments))
    with (RUN / (label + ".log")).open("a") as log:
        subprocess.run(arguments, cwd=ROOT, env=os.environ.copy(), stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def review(tag):
    definitions = [json.loads(line) for line in (DATA / "conditions.jsonl").read_text().splitlines()]
    summary = json.loads((DATA / tag / "collect_summary.json").read_text())
    cards = []
    for index in range(0, len(definitions), 2):
        cells = []
        for item in definitions[index:index+2]:
            condition = item["condition_index"]
            location = f"{tag}/calibration/c{condition:03d}"
            count = summary["coverage"][str(condition)]
            cells.append(f'<div><h3>{html.escape(item["side"])} · 학습 시연 {count}개</h3>'
                f'<p>{html.escape(item["instruction"])}</p>'
                f'<a href="{location}/views.mp4"><img loading="lazy" src="{location}/initial.png" '
                f'alt="initial observation"><img loading="lazy" src="{location}/final.png" '
                f'alt="final observation"></a><p>{html.escape(item["skill"])}</p></div>')
        cards.append(f'<article><h2>상황 {index//2+1}</h2><section>{"".join(cells)}</section></article>')
    (DATA / "demonstration_review.html").write_text(
        '<!doctype html><html lang="ko"><meta charset="utf-8"><title>v5 시연 검토</title>'
        '<style>body{font-family:sans-serif;margin:24px;background:#f4f4f4}article{background:white;padding:16px;margin:18px 0}'
        'section{display:grid;grid-template-columns:1fr 1fr;gap:24px}img{width:45%;max-width:224px}'
        '@media(max-width:700px){section{grid-template-columns:1fr}}</style>'
        '<h1>확장 v5 시연 검토</h1><p>동일 기본 배치의 안전·위험 조건이다. 이미지를 누르면 실제 시뮬레이션 영상을 연다. '
        '신규 학습 시연은 이 기본 배치와 다른 위치·방향에서 수집했다. 모든 채택 시연을 독립 재생 검증했다. '
        '성공은 명시한 운반·접촉·전달·기울이기 기준이며, 실제 위해 발생을 뜻하지 않는다.</p>'
        + ''.join(cards) + '</html>')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--tag", default="collection_r10")
    parser.add_argument("--warm-start-tags", default="collection_r9,collection_r8,collection_r7,collection_r6")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--episodes-per-condition", type=int, default=32)
    parser.add_argument("--recovery-workers", type=int, default=24)
    args = parser.parse_args()
    if (RUN / "pause_requested.json").exists():
        print("PAUSED_BY_USER: use the explicit resume entry point", flush=True)
        return
    assert args.episodes_per_condition >= 32
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(ROOT.parent / ".persist/nvidia-egl/10_nvidia.json")
    os.environ["LD_LIBRARY_PATH"] = str(ROOT.parent / ".persist/nvidia-egl")
    assert json.loads((DATA / "native_rehearsal_audit.json").read_text())["passed"]
    assert json.loads((RUN / "training_preflight_complete.json").read_text())["passed"]
    ready=RUN/"experiment_data_ready.json"
    if ready.exists():
        previous=json.loads(ready.read_text())
        assert previous["tag"]==args.tag and previous["episodes_per_condition"]==args.episodes_per_condition
        assert previous["fingerprint"]==data_fingerprint(args.tag),"Frozen inputs changed after the full data audit"
        command([PYTHON,"scripts/semantic_safety/run_expansion_v5_training.py"],"three_scope_training_and_evaluation")
        write_json(RUN/"experiment_status.json",dict(state="complete",updated_unix=time.time()))
        return
    base = [RENDER, "scripts/semantic_safety/collect_expansion_v5_demonstrations.py"]
    common = ["--tag", args.tag, "--workers", str(args.workers)]
    command(base + ["calibrate", "--profile-limit", "130", "--warm-start-tag", args.warm_start_tags]
            + common, "final_calibration_" + args.tag)
    calibration = json.loads((DATA / args.tag / "calibrate_summary.json").read_text())
    assert calibration["conditions"] == calibration["success"] == 338, calibration["failed"]
    summary=DATA/args.tag/"collect_summary.json"
    collection=json.loads(summary.read_text()) if summary.exists() else {}
    full_attempt=(collection.get("conditions")==338 and collection.get("signature")==calibration["signature"]
                  and collection.get("requested_indices")==list(range(338)))
    if not full_attempt:
        pilot = "0,4,12,16,32,38,46,48,50,54,56,57,63,65,79,81,84,98,100,104,128,135,158,286,288,293,319,324,328,330"
        command(base + ["collect", "--ids", pilot, "--count", "2", "--attempt-limit", "40"]
                + common, "final_data_pilot_" + args.tag)
        assert json.loads(summary.read_text())["complete"]
        command(base + ["collect", "--count", str(args.episodes_per_condition), "--attempt-limit", "240"]
                + common, "full_demonstrations_" + args.tag)
        collection=json.loads(summary.read_text())
    if not collection["complete"] or min(collection["coverage"].values())<args.episodes_per_condition:
        command([RENDER,"scripts/semantic_safety/recover_expansion_v5_collection.py","--tag",args.tag,
            "--count",str(args.episodes_per_condition),"--workers",str(args.recovery_workers)],"recover_missing_demonstrations")
        collection=json.loads(summary.read_text())
    assert collection["complete"], collection["coverage"]
    command([PYTHON, "scripts/semantic_safety/prepare_expansion_v5_lora_data.py", "freeze", "--tag", args.tag],
            "freeze_final_demonstrations")
    for name in ["conditions.jsonl", "design.json"]:
        shutil.copy2(DATA / name, DATA / args.tag / name)
    shutil.copytree(DATA / "assets", DATA / args.tag / "assets_snapshot", dirs_exist_ok=True)
    review(args.tag)
    write_json(ready,dict(tag=args.tag,episodes_per_condition=args.episodes_per_condition,
        fingerprint=data_fingerprint(args.tag),full_data_audit_passed=True,updated_unix=time.time()))
    command([PYTHON, "scripts/semantic_safety/run_expansion_v5_training.py"], "three_scope_training_and_evaluation")
    write_json(RUN / "experiment_status.json", dict(state="complete", updated_unix=time.time()))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        write_json(RUN / "experiment_error.json", dict(error=traceback.format_exc(), updated_unix=time.time()))
        raise
