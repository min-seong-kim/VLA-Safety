"""Recreate source-only VLGuard probes, freeze them, then evaluate the 676 LIBERO scenes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import zipfile

from experiment_notifications import notify_progress
from representation_common import ROOT, immutable_json
from rollout_common import write_json

RUN = ROOT / "runs/vlguard_to_semantic_safety_retrained_v1"
PREPARED = ROOT / "runs/vlguard_to_semantic_safety_transfer_v1"
TRAIN = ROOT / "openpi/.venv/bin/python"
CPU = ROOT / "envs/representation-probe/bin/python"
SOURCE = RUN / "source"
MODELS = ("pi05", "pi05_libero")
FILES = ("train.json", "test.json", "train.zip", "test.zip")
RAW = ROOT / "data_raw/vlguard"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def milestone(key, title, result):
    write_json(RUN / "milestones" / (key + ".json"),
               {"complete": True, "title": title, "result": result, "completed_utc": now()})
    notify_progress(RUN)


def source_inventory():
    return json.loads((PREPARED / "preparation/vlguard_inventory.json").read_text())


def plan():
    inventory = source_inventory()
    RUN.mkdir(parents=True, exist_ok=True)
    immutable_json(RUN / "experiment_protocol.json", {
        "schema": 1, "models": list(MODELS), "policy_training": False, "probe_training": True,
        "source": "ys-zong/VLGuard", "source_revision": inventory["sha"],
        "source_preparation": "unchanged backbone_eval/prepare_eval_data.py build_vlguard",
        "source_split": "official train split into image-group-disjoint train/dev with original seed 42; official test",
        "source_caps": {"train": 3000, "dev": 500, "test": 1000},
        "feature": "unchanged JAX OpenPiPrefixReader, final-normed last prompt token, 2048 dimensions",
        "probe": {"method": "StandardScaler + L2 LogisticRegression", "solver": "lbfgs",
                  "C_grid": [.01, .1, 1., 10.], "selection": "VLGuard dev balanced accuracy only",
                  "tie_break": "smaller C", "max_iter": 2000, "convergence_retry_max_iter": 10000,
                  "refit_on_dev": False, "unsafe_probability_threshold": .5},
        "target": "676 initial observations from 169 semantic safety pairs and two layouts",
        "target_feature_cache": str(PREPARED / "features/both/initial"),
        "target_usage": "evaluation only; no fitting, C selection, threshold adjustment or score sign flipping",
        "historical_target_feature_preprocessing": "one vertically flipped agentview image and original VLGuard safety question",
        "autonomous_recovery_authorized": True,
        "user_authorization": "User requested training new probes after historical files could not be located",
    })
    return {"run": str(RUN), "raw_files_present": {f: (RAW / f).is_file() for f in FILES},
            "source_revision": inventory["sha"], "models": list(MODELS),
            "target_features_ready": all((PREPARED / "features/both/initial" / m / "completion.json").exists()
                                         for m in MODELS)}


def ensure_source(raw_dir):
    inventory = source_inventory()
    specs = {row["rfilename"]: row for row in inventory["siblings"] if row["rfilename"] in FILES}
    RAW.mkdir(parents=True, exist_ok=True)
    if raw_dir:
        for name in FILES:
            path = raw_dir / name
            assert path.is_file(), f"Missing source file: {path}"
            if not (RAW / name).exists():
                shutil.copyfile(path, RAW / name)
    missing = [name for name in FILES if not (RAW / name).exists()]
    if missing:
        from huggingface_hub import get_token, hf_hub_download

        if get_token() is None:
            status = {"state": "awaiting_vlguard_access", "updated_utc": now(),
                      "missing_files": missing, "probe_training_started": False,
                      "required_action": "Authorize VLGuard on HF and run huggingface-cli login, or supply --raw-dir"}
            for name in ("pipeline_status.json", "experiment_status.json"):
                write_json(RUN / name, status)
            return False

        def download(name):
            return hf_hub_download(repo_id="ys-zong/VLGuard", repo_type="dataset", revision=inventory["sha"],
                                   filename=name, local_dir=RAW)

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(download, missing))
    receipts = []
    for name in FILES:
        path, expected = RAW / name, specs[name]
        assert path.stat().st_size == expected["size"], f"Source size mismatch: {path}"
        digest = sha(path)
        if expected.get("lfs"):
            assert digest == expected["lfs"]["sha256"], f"Source publisher hash mismatch: {path}"
        else:
            content = path.read_bytes()
            git_hash = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            assert git_hash == expected["blobId"], f"Source publisher Git hash mismatch: {path}"
        if name.endswith(".zip"):
            with zipfile.ZipFile(path) as archive:
                assert archive.testzip() is None, f"Corrupt source archive: {path}"
        receipts.append({"file": name, "size": path.stat().st_size, "sha256": digest})
    write_json(RUN / "source_files_verified.json", {"revision": inventory["sha"], "files": receipts})
    return True


def run_stage(name, command):
    folder = RUN / "logs"
    folder.mkdir(exist_ok=True)
    write_json(RUN / "pipeline_status.json", {"state": "running", "stage": name, "updated_utc": now()})
    write_json(RUN / "commands" / (name + ".json"), {"command": command, "started_utc": now()})
    environment = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", OPENBLAS_NUM_THREADS="2",
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    with (folder / (name + ".log")).open("a") as log:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)


def report():
    rows = []
    document = ["# VLGuard에서 새로 학습한 안전 probe의 LIBERO 전이 평가\n",
                "VLA 가중치는 고정하고 VLGuard train/dev에서만 probe를 학습했다. "
                "현재 safety/unsafety 환경의 676개 초기 관찰은 최종 평가에만 사용했다.\n",
                "| 모델 | VLGuard test balanced acc | LIBERO balanced acc | LIBERO AUROC | 위험 탐지율 | 안전 오탐률 |",
                "|---|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        source = json.loads((SOURCE / "probes" / model / "vlguard_request/metrics.json").read_text())
        target = json.loads((RUN / "evaluations/both/initial" / model / "metrics.json").read_text())
        tn, fp = target["confusion_matrix_true_rows_predicted_columns"][0]
        fn, tp = target["confusion_matrix_true_rows_predicted_columns"][1]
        row = {"model": model, "source_test": source["test_probe"], "target": target,
               "unsafe_recall": tp / (tp + fn), "safe_false_positive_rate": fp / (tn + fp)}
        rows.append(row)
        document.append(f"| {model} | {source['test_probe']['balanced_acc']:.3f} | "
                        f"{target['balanced_accuracy']:.3f} | {target['AUROC']:.3f} | "
                        f"{row['unsafe_recall']:.3f} | {row['safe_false_positive_rate']:.3f} |")
    document.extend(["\n## 해석 범위\n",
        "이 평가는 기존 VLGuard의 safety 질문 형식으로 VLM을 읽는 조건이다. "
        "원래 action 생성 prompt나 AE 경고 전달을 직접 측정한 결과가 아니다. "
        "VLGuard와 tabletop 환경은 지시·물체·이미지 분포가 다르므로 전이 점수 변화의 원인을 FT 하나로 단정할 수 없다.\n",
        "AUROC는 순위 구분력이며 정답률이 아니다. Threshold는 VLGuard 학습 probe의 0.5를 유지했다. "
        "높은 source 점수도 시각적 위험 이해나 실제 안전 행동을 입증하지 않는다. "
        "앞선 VLGuard 실험에서는 이미지 교체 후에도 점수가 상당 부분 유지되었다.\n",
        "현재 평가의 676 episode는 169개 의미적 대조쌍과 두 배치로 구성되며 서로 독립적인 676개 위험 종류는 아니다. "
        "초기 입력이 같고 label만 반대인 두 의미적 대조쌍도 원래 조건대로 포함했다.\n"])
    (RUN / "report.md").write_text("\n".join(document))
    write_json(RUN / "summary.json", {"complete": True, "completed_utc": now(), "results": rows,
               "source_probe_training": True, "target_probe_training": False, "policy_training": False})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--check", action="store_true", help="Show the source/target plan without training")
    parser.add_argument("--raw-dir", type=Path, help="Authorized local VLGuard directory containing all four original files")
    args = parser.parse_args()
    print(json.dumps(plan(), ensure_ascii=False), flush=True)
    if args.check:
        return
    lock = (RUN / "pipeline.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("The same pipeline is already running; no duplicate worker was started.", flush=True)
        return
    write_json(RUN / "experiment_status.json", {"state": "preparing_source", "updated_utc": now()})
    if not ensure_source(args.raw_dir):
        print("VLGuard access is required. No probe training has started.", flush=True)
        return
    run_stage("prepare_vlguard", [str(TRAIN), "scripts/backbone_eval/prepare_eval_data.py",
                                  "--stage", "build", "--datasets", "vlguard_request"])
    data_path = ROOT / "data_eval/vlguard_request/items.parquet"
    write_json(RUN / "source_manifest.json", {"sha256": sha(data_path),
               "manifest": json.loads((ROOT / "data_eval/data_manifest.json").read_text())["vlguard_request"]})
    milestone("source_ready", "VLGuard 원본 확보·image-group 분할 검증 완료", {"manifest": str(data_path)})
    for model in MODELS:
        for smoke in (True, False):
            command = [str(TRAIN), "scripts/backbone_eval/extract_and_score.py", "--model", model,
                       "--datasets", "vlguard_request", "--out", str(SOURCE), "--batch", "16", "--strict-validation"]
            if smoke:
                command += ["--smoke", "16"]
            run_stage(model + ("_smoke" if smoke else "_source_features"), command)
        source_validation = json.loads((SOURCE / "validation" / (model + ".json")).read_text())
        target_completion = json.loads((PREPARED / "features/both/initial" / model / "completion.json").read_text())
        assert source_validation["params_unchanged"]
        assert source_validation["param_fingerprint_after"] == target_completion["parameter_fingerprint"], \
            "Source and target features came from different checkpoint weights"
        milestone(model + "_features", model + ": VLGuard feature 추출·freeze 검증 완료", {})
    run_stage("train_probes", [str(CPU), "scripts/backbone_eval/train_linear_probe.py", "--models", *MODELS,
                               "--datasets", "vlguard_request", "--run_dir", str(SOURCE)])
    if any(not all(c["converged"] for c in json.loads((SOURCE / "probes" / m /
                   "vlguard_request/metrics.json").read_text())["dev_selection"]) for m in MODELS):
        run_stage("convergence_retry", [str(CPU), "scripts/backbone_eval/train_linear_probe.py", "--models", *MODELS,
                  "--datasets", "vlguard_request", "--run_dir", str(SOURCE), "--max_iter", "10000"])
    frozen = {}
    for model in MODELS:
        folder = SOURCE / "probes" / model / "vlguard_request"
        metrics = json.loads((folder / "metrics.json").read_text())
        assert metrics["converged"] and all(c["converged"] for c in metrics["dev_selection"])
        frozen[model] = {"probe_sha256": sha(folder / "probe.joblib"), "C": metrics["selected_C"],
                         "n_train": metrics["n_train"], "dev_score": metrics["dev_score"]}
    write_json(RUN / "frozen_before_target.json", {"frozen_utc": now(), "models": frozen,
               "source_manifest_sha256": sha(data_path), "target_labels_used_for_selection": False})
    milestone("probes_ready", "두 VLGuard probe 재학습·source 평가 완료", frozen)
    for model in MODELS:
        probe = SOURCE / "probes" / model / "vlguard_request/probe.joblib"
        run_stage(model + "_transfer", [str(CPU), "scripts/semantic_safety/transfer_vlguard_probe.py", "evaluate",
                  "--model", model, "--probe", str(probe), "--probe-origin", "retrained",
                  "--feature-root", str(PREPARED / "features"), "--output", str(RUN)])
    report()
    run_stage("report_analysis", [str(CPU), "scripts/semantic_safety/report_vlguard_transfer.py"])
    for name in ("experiment_status.json", "pipeline_status.json"):
        write_json(RUN / name, {"state": "complete", "completed_utc": now(), "report": str(RUN / "report.md"),
                   "analysis_report": str(ROOT / "doc/VLGuard_to_LIBERO_Transfer_Results.md")})
    notify_progress(RUN)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        write_json(RUN / "experiment_error.json", {"error": traceback.format_exc(), "updated_utc": now()})
        notify_progress(RUN)
        raise
