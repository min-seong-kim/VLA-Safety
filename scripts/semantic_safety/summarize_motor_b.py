"""Summarize completed matched A/B experiments and build an offline video report."""

import csv
import hashlib
import html
import json
from pathlib import Path
import zipfile

import cv2
import h5py
import matplotlib
import numpy as np

from rollout_common import ROOT, write_json


RUN = ROOT / "runs/motor_lora_b_v1"
DATA = ROOT / "data/motor_skill_b_v1/demonstrations"
matplotlib.use("Agg")


def read(path):
    return json.loads(path.read_text())


def episodes(folder):
    return {path.parent.name: read(path) for path in sorted(folder.glob("*/summary.json"))}


def comparison(kind):
    name = {"motor": "test", "native": "native", "transfer": "transfer"}[kind]
    a, b = episodes(RUN / f"{name}_A"), episodes(RUN / f"{name}_B")
    assert a.keys() == b.keys(), (kind, len(a), len(b))
    assert all(a[key]["initial_observation_sha256"] == b[key]["initial_observation_sha256"] for key in a)
    assert all(a[key]["instruction"] == b[key]["instruction"] and
               a[key]["horizon"] == b[key]["horizon"] and
               a[key]["execute_chunk_steps"] == b[key]["execute_chunk_steps"] for key in a)
    for key in a:
        with h5py.File(RUN / f"{name}_A" / key / "trajectory.h5") as first, \
                h5py.File(RUN / f"{name}_B" / key / "trajectory.h5") as second:
            assert np.array_equal(first["initial_sim_state"][:], second["initial_sim_state"][:]), key
            count = min(len(first["flow_noise_seeds"]), len(second["flow_noise_seeds"]))
            assert np.array_equal(first["flow_noise_seeds"][:count], second["flow_noise_seeds"][:count]), key
    supported = [key for key in a if a[key]["task_success_supported"]]
    result = {"episodes": len(a), "success_supported_episodes": len(supported), "paired_initial_images_exact": True,
              "paired_initial_simulator_states_exact": True, "paired_decision_noise_seeds_exact": True,
              "A_successes": sum(a[key]["task_success"] for key in supported),
              "B_successes": sum(b[key]["task_success"] for key in supported),
              "A_grasps": sum(row["ever_grasped"] for row in a.values()) if kind != "native" else None,
              "B_grasps": sum(row["ever_grasped"] for row in b.values()) if kind != "native" else None,
              "A_lifts": sum(row["ever_lifted"] for row in a.values()) if kind != "native" else None,
              "B_lifts": sum(row["ever_lifted"] for row in b.values()) if kind != "native" else None,
              "A_only_success": sum(a[key]["task_success"] and not b[key]["task_success"] for key in supported),
              "B_only_success": sum(b[key]["task_success"] and not a[key]["task_success"] for key in supported),
              "horizon": next(iter(a.values()))["horizon"],
              "execute_chunk_steps": next(iter(a.values()))["execute_chunk_steps"]}
    if kind == "motor":
        assets = sorted({row["asset"] for row in a.values()})
        effects = [np.mean([float(b[key]["task_success"]) - float(a[key]["task_success"])
                            for key in a if a[key]["asset"] == asset]) for asset in assets]
        rng = np.random.default_rng(991)
        samples = rng.choice(effects, size=(20000, len(effects)), replace=True).mean(1)
        result["paired_difference_percentage_points"] = float(np.mean(effects) * 100)
        result["asset_cluster_bootstrap_difference_95pct_pp"] = (np.quantile(samples, [.025, .975]) * 100).tolist()
        result["uncertainty_scope"] = "resample 27 asset clusters; one training seed; not multi-seed uncertainty"
    gallery = []
    for key in a:
        for side in ("A", "B"):
            folder = RUN / f"{name}_{side}" / key
            with h5py.File(folder / "trajectory.h5") as trajectory:
                pair = np.concatenate([trajectory["observations/main_rgb_raw"][0][::-1, ::-1],
                                       trajectory["observations/wrist_rgb_raw"][0][::-1, ::-1]], axis=1)
                cv2.imwrite(str(folder / "poster.png"), np.ascontiguousarray(pair[:, :, ::-1]))
        gallery.append({"kind": kind, "id": key, "asset": a[key].get("asset", key),
                        "instruction": a[key]["instruction"], "A": a[key]["task_success"], "B": b[key]["task_success"],
                        "video_A": f"{name}_A/{key}/views.mp4", "video_B": f"{name}_B/{key}/views.mp4",
                        "poster_A": f"{name}_A/{key}/poster.png", "poster_B": f"{name}_B/{key}/poster.png"})
    return result, gallery, a, b


def main():
    assert (RUN / "workflow_completed.json").exists()
    seen_training = read(RUN / "seen_train_B/aggregate.json")
    provenance = read(RUN / "provenance.json")
    scripts = ROOT / "scripts/semantic_safety"
    paths = list(scripts.glob("*motor*.py")) + [scripts / name for name in (
        "collect_motor_demonstrations.py", "rollout_common.py", "rollout_safety_env.py", "motor_b_report_template.html")]
    provenance["script_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    write_json(RUN / "provenance.json", provenance)
    coverage, design = read(DATA / "coverage.json"), read(DATA / "frozen_design.json")
    selection, freeze = read(RUN / "selection.json"), read(RUN / "training/freeze_audit.json")
    import matplotlib.pyplot as plt

    training = [json.loads(line) for line in (RUN / "training/training.jsonl").read_text().splitlines()]
    loss = np.asarray([row["loss"] for row in training])
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    axes[0].plot(np.arange(100, len(loss) + 1), np.convolve(loss, np.ones(100) / 100, mode="valid"), color="#087563")
    axes[0].set(xlabel="Optimizer step", ylabel="Flow-matching loss (100-step mean)", title="Training demonstrations only")
    steps = [row["step"] for row in selection["candidates"]]
    values = [100 * row["successes"] / row["episodes"] for row in selection["candidates"]]
    axes[1].plot(steps, values, "o-", color="#087563", label="B: expert LoRA")
    validation_a = read(RUN / "validation_A/aggregate.json")
    axes[1].axhline(100 * validation_a["successes"] / validation_a["episodes"], color="#65727a", linestyle="--", label="A: original")
    axes[1].set(xlabel="Checkpoint step", ylabel="Task success (%)", ylim=(0, 100), title="54 held-out validation layouts")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=.15)
    figure.savefig(RUN / "learning_curve.svg")
    figure.savefig(RUN / "learning_curve.png", dpi=180)
    plt.close(figure)
    manifest = [json.loads(line) for line in (DATA / "manifest.jsonl").read_text().splitlines()]
    accepted = [row for row in manifest if row["accepted"]]
    metrics, gallery, raw = {}, [], {}
    for kind in ("motor", "native", "transfer"):
        metrics[kind], clips, a, b = comparison(kind)
        gallery.extend(clips)
        raw[kind] = (a, b)
    a, b = raw["motor"]
    object_rows = []
    for asset in design["assets"]:
        keys = [key for key in a if a[key]["asset"] == asset]
        object_rows.append({"asset": asset, "demos_accepted": coverage["assets"][asset]["accepted"],
                            "demos_attempted": coverage["assets"][asset]["attempted"], "test_episodes": len(keys),
                            "A_successes": sum(a[key]["task_success"] for key in keys),
                            "B_successes": sum(b[key]["task_success"] for key in keys),
                            "A_grasps": sum(a[key]["ever_grasped"] for key in keys),
                            "B_grasps": sum(b[key]["ever_grasped"] for key in keys),
                            "A_lifts": sum(a[key]["ever_lifted"] for key in keys),
                            "B_lifts": sum(b[key]["ever_lifted"] for key in keys)})
    teacher = read(RUN / "teacher_test/aggregate.json")
    for row in object_rows:
        row["teacher_test_successes"] = sum(result["success"] for result in teacher["results"]
                                            if result["spec"]["asset"] == row["asset"])
    result = {"model": "pi05_libero", "experiment": "B: VLM frozen, action-expert-only LoRA", "probe_trained": False,
              "dataset": {"assets": len(design["assets"]), "attempts": len(manifest), "accepted_demos": len(accepted),
                          "action_timesteps": sum(row["steps"] for row in accepted),
                          "max_replay_state_error": max(row["replay"]["trajectory_state_max_abs_error"] for row in accepted),
                          "excluded_assets": ["battery", "chocolate", "wallet", "wood_bowl", "rice_cake"]},
              "selected_step": selection["selected_step"], "selected_adapter": selection["adapter"], "freeze_audit": freeze,
              "metrics": metrics, "objects": object_rows,
              "teacher_test": {"successes": teacher["successes"], "episodes": teacher["episodes"]},
              "seen_training_diagnostic": {"successes": seen_training["successes"],
                                           "episodes": seen_training["episodes"],
                                           "selection": "first two accepted training episodes per asset",
                                           "scope": "posthoc closed-loop fit diagnostic; not used for checkpoint selection"},
              "interpretation": "This B configuration does not establish robust motor adaptation: little held-out improvement, native regression, and poor closed-loop success on seen training layouts. VLM freezing was verified; causes of motor failure are not isolated by this experiment.",
              "limitations": ["single training seed and 27 existing asset identities; held-out layouts only",
                              "novelty of every asset/category to the full pretrained model corpus is unverified",
                              "neutral pick-and-place training, not physical cutting/pouring or semantic safety training",
                              "VLM unchanged for identical inputs; learned action-expert behavior can change",
                              "38 failed teacher attempts preserved; 5 candidate assets excluded before final evaluation",
                              "unsupported physical operations remain null, not failures or safety refusal"]}
    write_json(RUN / "results.json", result)
    with (RUN / "per_asset.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(object_rows[0]))
        writer.writeheader(); writer.writerows(object_rows)
    teacher_gallery = read(RUN / "teacher_examples/gallery.json")
    payload = json.dumps({"result": result, "gallery": gallery, "teacher_gallery": teacher_gallery},
                         ensure_ascii=False).replace("<", "\\u003c")
    template = (ROOT / "scripts/semantic_safety/motor_b_report_template.html").read_text()
    (RUN / "report.html").write_text(template.replace("__REPORT_DATA__", payload))
    with zipfile.ZipFile(RUN / "report_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for name in ("report.html", "results.json", "per_asset.csv", "selection.json", "provenance.json", "learning_curve.svg"):
            bundle.write(RUN / name, arcname=name)
        for row in gallery:
            for key in ("video_A", "video_B", "poster_A", "poster_B"):
                path = RUN / row[key]
                bundle.write(path, arcname=row[key])
        for row in teacher_gallery:
            for key in ("video", "poster"):
                bundle.write(RUN / row[key], arcname=row[key])
    motor = metrics["motor"]
    report = f"""실험 B 결과 — pi0.5 LIBERO

VLM 전체와 원본 action expert 가중치를 고정하고, action expert 18개 층의 126개 선형 모듈에 LoRA(rank 16, alpha 32)만 학습했습니다.
총 학습 파라미터: 6,930,432. Probe와 실험 C는 수행하지 않았습니다.

Demonstration: 공식 LIBERO oracle policy가 아닙니다. 공식 LIBERO 수집 코드는 사람이 조작하고 성공 판정으로 수집을 끝냅니다.
이번에는 프로젝트의 privileged-state scripted teacher를 사용했습니다. Simulator의 물체/목표 위치를 teacher에만 주고 실제 OSC_POSE action으로 움직였습니다.
학생 입력은 두 RGB 영상과 중립적인 instruction뿐입니다. 물체 pose, teacher 단계, 정답 성공/안전 label은 학생에게 전달하지 않았습니다.
27종 × 24개 학습 배치 = 648회 시도 중 610개 성공 시연, 123,352 action timestep입니다.
610개 모두 별도의 새 reset에서 저장 action을 다시 실행해 성공을 확인했고, 모든 simulator state의 최대 차이는 0입니다.
양쪽 손가락 접촉 → 4 cm 초과 lift → tray 접촉/중심 위치/놓기/낮은 속도 10 timestep 유지가 성공 조건입니다.
No-op 대조군 27회는 모두 실패로 판정됐습니다. 실패 시연 38회도 보존했습니다.

최종 test: A {motor['A_successes']}/{motor['episodes']}, B {motor['B_successes']}/{motor['episodes']}.
기존 LIBERO 유지: A {metrics['native']['A_successes']}/16, B {metrics['native']['B_successes']}/16.
원래 v3 안전 지시의 기하학적 완료: A {metrics['transfer']['A_successes']}/15, B {metrics['transfer']['B_successes']}/15 (20개는 물리적 완료 판정 미지원).
학습 배치 54개의 사후 실행 진단: B {seen_training['successes']}/{seen_training['episodes']}.
현재 B 설정은 안정적인 조작 능력 확보에 미달합니다. 학습에 포함된 초기 배치에서도 실패하므로 새로운 배치 일반화만의 문제로 해석하지 않습니다.
이 결과만으로 원인을 VLM freeze, LoRA 용량, 시연 분포, 최적화 또는 action chunk 실행 중 하나로 확정할 수는 없습니다.
선택된 checkpoint: {selection['selected_step']} step. 선택에는 별도 validation 54개 배치만 사용했습니다.
학습/validation/test는 전체 episode와 초기 layout seed로 나눴고, 동일 trajectory의 timestep을 분할하지 않았습니다.
Test는 새로운 배치 일반화입니다. 새로운 물체 종류나 안전 판단 일반화가 아닙니다.
새 motor A/B는 360 step, 10-action chunk 실행을 함께 적용했습니다. 과거 v3 A 결과와 실행 규칙이 다르므로 새로운 A를 다시 측정했습니다.
기존 v3 안전 지시 전이는 원래의 160 step / replan 1 규칙으로 A와 B를 모두 새로 실행했습니다.
고정 입력에서 VLM hidden이 완전히 같고 모든 원본 파라미터 해시가 유지됐습니다. 이 검사는 전체 모델의 안전 행동 보존을 뜻하지는 않습니다.

보기: report_bundle.zip 전체를 내려받아 압축을 풀고 report.html을 브라우저에서 여세요.
원본 trajectory: 각 평가 episode 폴더의 trajectory.h5, events.jsonl, model.xml, summary.json.
학습 데이터: data/motor_skill_b_v1/demonstrations/manifest.jsonl.
LoRA: {selection['adapter']}.
재현 코드: scripts/semantic_safety/run_motor_b_workflow.py 및 관련 motor_* 스크립트.
"""
    (RUN / "REPORT.txt").write_text(report)
    with zipfile.ZipFile(RUN / "report_bundle.zip", "a", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        bundle.write(RUN / "REPORT.txt", arcname="REPORT.txt")
        bundle.write(ROOT / "data/motor_skill_b_v1/README.txt", arcname="DATA_README.txt")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
