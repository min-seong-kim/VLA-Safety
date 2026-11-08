"""Verify all new evaluation artifacts and compare each checkpoint with its existing family baseline."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import pathlib

from common import DATASETS, ROOT, RUN_DIR, dump_json, read_items, unique_item_ids

PAIRS = {
    "pi0_libero": "pi0",
    "pi05_libero": "pi05",
    "openvla_libero_spatial": "openvla",
    "openvla_oft_goal": "openvla_oft",
}
DATA = [d for d in DATASETS if d != "llavaguard_rating"]


def predictions(run, model, dataset):
    path = run / "predictions" / model / f"{dataset}.jsonl"
    return [json.loads(line) for line in path.open()]


def accuracy(records, key):
    return sum(r[key] == r["label"] for r in records) / len(records)


def probe_converged(record):
    selected = next(row for row in record["dev_selection"] if row["C"] == record["selected_C"])
    return record["converged"] and selected["n_iter"] < record.get("max_iter", 2000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=pathlib.Path, required=True)
    ap.add_argument("--baseline_dir", type=pathlib.Path, default=RUN_DIR)
    args = ap.parse_args()
    config = json.loads((args.run_dir / "run_config.json").read_text())
    for relative, expected in config["file_sha256"].items():
        if relative.startswith(("data_eval/", "runs/backbone_eval_v1/")):
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen data or baseline changed: {relative}")
    rows = []
    for new, base in PAIRS.items():
        if new.startswith("openvla"):
            for model in [new, base]:
                check = json.loads((args.run_dir / "validation" / f"{model}.json").read_text())
                if check["status"] != "done" or check.get("validation_failures"):
                    raise ValueError(f"Incomplete singleton control: {model}")
                if check["describe"].get("physical_batch_limit") != 1:
                    raise ValueError(f"HF comparison requires matched singleton execution: {model}")
        validation = json.loads((args.run_dir / "validation" / f"{new}.json").read_text())
        if validation["status"] != "done" or validation.get("validation_failures"):
            raise ValueError(f"Incomplete extraction: {new}")
        for dataset in DATA:
            baseline_run = (args.run_dir if (args.run_dir / "probes" / base / dataset / "metrics.json").exists()
                            else args.baseline_dir)
            expected = config["file_sha256"][f"data_eval/{dataset}/items.parquet"]
            if validation["datasets"][dataset]["items_sha256"] != expected:
                raise ValueError(f"Extraction used a different manifest: {new}/{dataset}")
            metrics = json.loads((args.run_dir / "probes" / new / dataset / "metrics.json").read_text())
            if metrics["status"] != "ok" or not probe_converged(metrics) or metrics["classes_missing_in_train"]:
                raise ValueError(f"Unusable probe: {new}/{dataset}")
            baseline_metrics = json.loads((baseline_run / "probes" / base / dataset / "metrics.json").read_text())
            if (baseline_metrics["status"] != "ok" or not probe_converged(baseline_metrics)
                    or baseline_metrics["classes_missing_in_train"]):
                raise ValueError(f"Unusable baseline probe: {base}/{dataset}")
            old_records = predictions(baseline_run, base, dataset)
            new_records = predictions(args.run_dir, new, dataset)
            if len({r["item_id"] for r in new_records}) != len(new_records):
                raise ValueError(f"Duplicate new IDs: {new}/{dataset}")
            source = {r["item_id"]: r for r in unique_item_ids(read_items(dataset)) if r["split"] == "test"}
            for record in new_records:
                if record["label"] != source[record["item_id"]]["label"]:
                    raise ValueError("Prediction/manifest label mismatch")
            old_counts = collections.Counter(
                (source.get(r["item_id"], {"source_item_id": r["item_id"]})["source_item_id"], r["label"])
                for r in old_records
            )
            new_counts = collections.Counter((source[r["item_id"]]["source_item_id"], r["label"]) for r in new_records)
            if old_counts != new_counts:
                raise ValueError(f"Baseline/new test populations differ: {base}/{new}/{dataset}")
            row = {"baseline": base, "checkpoint": new, "dataset": dataset, "n_test": len(new_records),
                   "baseline_run": baseline_run.name}
            for key, metric in [("pred_ll", "likelihood"), ("pred_probe", "probe")]:
                old_acc, new_acc = accuracy(old_records, key), accuracy(new_records, key)
                if abs(new_acc - metrics[f"test_{metric}"]["acc"]) > 1e-10:
                    raise ValueError(f"Saved accuracy mismatch: {new}/{dataset}/{metric}")
                if abs(old_acc - baseline_metrics[f"test_{metric}"]["acc"]) > 1e-10:
                    raise ValueError(f"Matched baseline accuracy mismatch: {base}/{dataset}/{metric}")
                row.update({f"baseline_{metric}": old_acc, f"new_{metric}": new_acc,
                            f"delta_{metric}_pp": 100 * (new_acc - old_acc)})
            row["comparison"] = "same source ID/label multiplicities; no POPE itemwise pairing" if dataset == "pope_yn" else "same test IDs and labels"
            rows.append(row)
    with (args.run_dir / "checkpoint_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dump_json(args.run_dir / "completion.json", {
        "status": "complete", "n_model_dataset_pairs": len(rows), "pairs": PAIRS,
        "baseline_dir": str(args.baseline_dir), "run_dir": str(args.run_dir),
        "checks": ["extraction gates", "probe convergence", "unique new IDs", "test labels/populations",
                   "prediction accuracies agree with probe metrics"],
    })
    md = ["# 추가 VLA 체크포인트 평가: 기존 체크포인트와의 비교", "",
          "공개된 체크포인트를 평가했다. VLA 자체는 추가 학습하지 않고 데이터셋별 선형 probe만 학습했다.", "",
          f"결과: `{args.run_dir.name}`. Pi 기준은 `{args.baseline_dir.name}`, OpenVLA 계열 기준은 이번 단독 실행 재평가를 사용했다.", "",
          "## 평가 조건", "",
          "- 기존 이미지·질문·선택지·split·seed와 첫 토큰 likelihood/마지막 표현 probe 방식을 유지했다.",
          "- OpenVLA 계열은 bf16 batch 수치 민감도 때문에 문항별 단독 실행을 사용했다. 두 HF 기준 모델도 같은 방식으로 재평가했다.",
          "- 단독 실행의 합성 호출 검증은 물리적인 padded batch 경로의 안정성을 증명하는 검사가 아니다. 원래 배치 경로의 진단은 numerics/에 보존했다.",
          "- 반복 상한에 도달한 OFT 및 Pi0 기준 SQA probe는 같은 캐시 특징으로 최대 10,000회까지 재최적화했다. 선택된 모델의 실제 반복 횟수도 검증했다.",
          "- 새 POPE 결과의 ID만 원본 manifest 행 번호로 구분했다. 원본 데이터와 v1 결과는 변경하지 않았다.",
          "- 각 모델별 16문항 smoke 검증 이후 본 추출을 수행했다. 아래 표는 전체 test 결과다.",
          "- π0/π0.5는 base→LIBERO, OpenVLA는 base→LIBERO-Spatial을 비교한다.",
          "- OFT는 LIBERO-Spatial→LIBERO-Goal 비교다. 두 독립 suite 체크포인트이며 순차 추가 학습의 전후 비교가 아니다.",
          "- LlavaGuard는 원 실험과 마찬가지로 미실행이다.", "",
          "## 정확도 비교", "",
          "정확도는 %, 변화량은 %p다. 같은 계열의 기준/추가 모델은 같은 test population을 사용했다.", "",
          "| 기준 → 추가 | 데이터 | N | 기존 LL | 추가 LL | 변화 | 기존 probe | 추가 probe | 변화 |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['baseline']} → {r['checkpoint']} | {DATASETS[r['dataset']][0]} | {r['n_test']} | "
                  f"{100*r['baseline_likelihood']:.1f} | {100*r['new_likelihood']:.1f} | {r['delta_likelihood_pp']:+.1f} | "
                  f"{100*r['baseline_probe']:.1f} | {100*r['new_probe']:.1f} | {r['delta_probe_pp']:+.1f} |")
    md += ["", "## 해석 범위", "",
           "POPE 전체 accuracy는 보존된 평가 행을 기준으로 비교했다. 기존 POPE 범주/셔플 결과에는 ID 중복 문제가 있어 전후 비교에 사용하지 않는다. 새 결과의 해당 집계는 고유 ID를 사용한다.", "",
           "SIUO는 같은 계열 안에서는 test ID가 같지만 Pi 계열은 26개, Llama 계열은 32개다. 계열 간 직접 순위 비교에는 공통 subset이 필요하다.", "",
           "VLGuard는 요청 분류이며 이미지 셔플과 balanced accuracy를 함께 봐야 한다. 실제 로봇 행동이나 거절 응답의 안전성을 측정하지 않았다.", "",
           "전처리와 판독 방식을 고정했어도 학습 데이터·목표·suite 차이를 하나의 인과 효과로 분리한 실험은 아니다.", "",
           "상세 지표와 검증은 같은 디렉터리의 `summary.md`, `validation_report.md`, `diagnostics.md`, `completion.json`을 참고한다."]
    (args.run_dir / "checkpoint_comparison.md").write_text("\n".join(md) + "\n")
    print(f"Verified {len(rows)} complete model/dataset combinations; wrote checkpoint_comparison.csv/.md")


if __name__ == "__main__":
    main()
