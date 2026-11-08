"""Merge archived evaluation metrics into auditable tables and publication-style static plots.

This reads small JSON/JSONL artifacts only. It never loads checkpoints or fits a classifier.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
V1 = ROOT / "runs/backbone_eval_v1"
V2 = ROOT / "runs/backbone_eval_v2_checkpoints"
ASSETS = ROOT / "doc/assets/vla_vlm_combined"
REPORT = ROOT / "doc/VLA_VLM_Combined_Results.md"
MODELS = ["paligemma", "paligemma_lb", "pi0", "pi0_libero", "pi05", "pi05_libero", "prismatic",
          "openvla", "openvla_libero_spatial", "openvla_oft", "openvla_oft_goal"]
LABELS = ["PaliGemma", "PaliGemma-LB", "π0 base", "π0 LIBERO", "π0.5 base", "π0.5 LIBERO", "Prismatic",
          "OpenVLA base", "OpenVLA Spatial", "OFT Spatial", "OFT Goal"]
PLOT_LABELS = ["PaliGemma", "PaliGemma-LB", "Pi0 base", "Pi0 LIBERO", "Pi0.5 base", "Pi0.5 LIBERO",
               "Prismatic", "OpenVLA base", "OpenVLA Spatial", "OFT Spatial", "OFT Goal"]
NAMES = dict(zip(MODELS, LABELS, strict=True))
DATASETS = ["pope_yn", "mme_yn", "sqa_img_mc", "vqav2_mc", "coco_capchoice", "vlguard_request", "siuo_mcqa"]
UTILITY = DATASETS[:5]
CHANCE = [50, 50, 35.38, 34.625, 25]
PAIRS = [("pi0", "pi0_libero"), ("pi05", "pi05_libero"),
         ("openvla", "openvla_libero_spatial"), ("openvla_oft", "openvla_oft_goal")]
COLORS = ["#2878B5", "#E9812B"]


def read_json(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent(value):
    return f"{100*value:.1f}"


def save_figure(fig, name):
    fig.savefig(ASSETS / f"{name}.png", dpi=180, facecolor="white", bbox_inches="tight")
    fig.savefig(ASSETS / f"{name}.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    if read_json(V2 / "completion.json")["status"] != "complete":
        raise RuntimeError("Checkpoint expansion is not complete")
    ASSETS.mkdir(parents=True, exist_ok=True)
    metrics, sources, all_rows = {}, {}, []
    inputs = {}
    for model in MODELS:
        for dataset in DATASETS:
            for run in (V1, V2):
                path = run / "probes" / model / dataset / "metrics.json"
                if not path.exists():
                    continue
                r = read_json(path)
                selected = next(c for c in r["dev_selection"] if c["C"] == r["selected_C"])
                row = {"model": model, "label": NAMES[model], "dataset": dataset, "source_run": run.name,
                       "n_train": r["n_train"], "n_dev": r["n_dev"], "n_test": r["n_test"],
                       "lm_head_accuracy": r["test_likelihood"]["acc"], "probe_accuracy": r["test_probe"]["acc"],
                       "lm_balanced_accuracy": r["test_likelihood"].get("balanced_acc", ""),
                       "probe_balanced_accuracy": r["test_probe"].get("balanced_acc", ""),
                       "selected_C": r["selected_C"], "selected_n_iter": selected["n_iter"],
                       "recorded_converged": r["converged"],
                       "max_iter": r.get("max_iter", 2000),
                       "iteration_cap_reached": selected["n_iter"] >= r.get("max_iter", 2000),
                       "is_canonical": False, "metrics_path": str(path.relative_to(ROOT))}
                all_rows.append(row)
                metrics[model, dataset], sources[model, dataset] = r, run
                inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    canonical = []
    canonical_paths = {str((run / "probes" / model / dataset / "metrics.json").relative_to(ROOT))
                       for (model, dataset), run in sources.items()}
    for row in all_rows:
        row["is_canonical"] = row["metrics_path"] in canonical_paths
        if row["is_canonical"]:
            if row["iteration_cap_reached"] or not row["recorded_converged"]:
                raise ValueError(f"Canonical probe reached iteration cap: {row['metrics_path']}")
            canonical.append(dict(row))
    assert len(canonical) == 77 and len(all_rows) == 92

    # Recompute held-out accuracies from saved predictions; select all-model common SIUO IDs.
    siuo = {}
    for (model, dataset), r in metrics.items():
        path = sources[model, dataset] / "predictions" / model / f"{dataset}.jsonl"
        records = [json.loads(line) for line in path.open()]
        assert len(records) == r["n_test"], path
        for key, metric in [("pred_ll", "test_likelihood"), ("pred_probe", "test_probe")]:
            actual = sum(x[key] == x["label"] for x in records) / len(records)
            assert abs(actual-r[metric]["acc"]) < 1e-10, path
        if dataset == "siuo_mcqa":
            assert len({x["item_id"] for x in records}) == len(records)
            siuo[model] = {x["item_id"]: x for x in records}
    common = set.intersection(*(set(x) for x in siuo.values()))
    assert len(common) == 26
    for item_id in common:
        assert len({r[item_id]["label"] for r in siuo.values()}) == 1
    common_rows = []
    for model in MODELS:
        records = [siuo[model][i] for i in sorted(common)]
        ll = sum(r["pred_ll"] == r["label"] for r in records)
        probe = sum(r["pred_probe"] == r["label"] for r in records)
        common_rows.append({"model": model, "n": 26, "lm_correct": ll, "probe_correct": probe,
                            "lm_head_accuracy": ll/26, "probe_accuracy": probe/26,
                            "original_n": len(siuo[model]), "source_run": sources[model, "siuo_mcqa"].name})
    siuo_common = {r["model"]: r for r in common_rows}
    for row in canonical:
        if row["dataset"] == "siuo_mcqa":
            c = siuo_common[row["model"]]
            row.update(reported_n=26, reported_lm_accuracy=c["lm_head_accuracy"], reported_probe_accuracy=c["probe_accuracy"])
        else:
            row.update(reported_n=row["n_test"], reported_lm_accuracy=row["lm_head_accuracy"], reported_probe_accuracy=row["probe_accuracy"])
    write_csv(ASSETS / "canonical_results.csv", canonical)
    write_csv(ASSETS / "all_runs.csv", all_rows)
    write_csv(ASSETS / "siuo_common26.csv", common_rows)

    archive = []
    for row in all_rows:
        if row["is_canonical"]:
            continue
        r = metrics[row["model"], row["dataset"]]
        archive.append({"model": row["model"], "dataset": row["dataset"], "old_source": row["source_run"],
                        "canonical_source": sources[row["model"], row["dataset"]].name,
                        "old_lm_accuracy": row["lm_head_accuracy"], "canonical_lm_accuracy": r["test_likelihood"]["acc"],
                        "old_probe_accuracy": row["probe_accuracy"], "canonical_probe_accuracy": r["test_probe"]["acc"],
                        "reason": "cached probe convergence repair" if row["model"] == "pi0" else "matched singleton HF reevaluation"})
    assert len(archive) == 15
    write_csv(ASSETS / "historical_vs_canonical.csv", archive)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "savefig.dpi": 180})
    # Heatmap values are raw accuracy; task-specific chance levels are labelled explicitly.
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 8.2), layout="constrained", sharey=True)
    for ax, key, title in zip(axes, ["test_likelihood", "test_probe"],
                              ["Existing LM head: candidate accuracy", "Linear probe: held-out accuracy"], strict=True):
        values = np.array([[100*metrics[m, d][key]["acc"] for d in UTILITY] for m in MODELS])
        im = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(5), [f"{n}\nchance {c:.1f}%" for n,c in zip(["POPE","MME","SQA","VQA-MC","COCO choice"],CHANCE,strict=True)])
        ax.set_yticks(range(11), PLOT_LABELS)
        ax.set_title(title, fontweight="bold", pad=14)
        for y in range(11):
            for x in range(5):
                ax.text(x,y,f"{values[y,x]:.1f}",ha="center",va="center",color="white" if values[y,x]>65 else "#18232E",fontsize=10)
        ax.axhline(5.5,color="white",linewidth=3)
        ax.set_xticks(np.arange(-.5,5,1),minor=True);ax.set_yticks(np.arange(-.5,11,1),minor=True)
        ax.grid(which="minor",color="white",linewidth=.5);ax.tick_params(which="minor",bottom=False,left=False)
    fig.colorbar(im,ax=axes,shrink=.65,label="Accuracy (%)",pad=.02)
    save_figure(fig,"utility_overview")

    # Differences are computed only within matched checkpoint families; no cross-benchmark average is used.
    fig, axes = plt.subplots(2,2,figsize=(15,10),layout="constrained")
    delta_rows = []
    delta_datasets = UTILITY+["vlguard_request"]
    for ax,(base,new) in zip(axes.flat,PAIRS,strict=True):
        values = []
        for key in ["test_likelihood","test_probe"]:
            diff = [100*(metrics[new,d][key]["balanced_acc" if d=="vlguard_request" else "acc"]-
                         metrics[base,d][key]["balanced_acc" if d=="vlguard_request" else "acc"]) for d in delta_datasets]
            values.append(diff)
        y=np.arange(6)
        for j,(diff,color,label) in enumerate(zip(values,COLORS,["LM head","Probe"],strict=True)):
            bars=ax.barh(y+(j-.5)*.32,diff,height=.3,color=color,label=label)
            for bar,value in zip(bars,diff,strict=True):
                ax.text(value+(.8 if value>=0 else -.8),bar.get_y()+bar.get_height()/2,f"{value:+.1f}",
                        va="center",ha="left" if value>=0 else "right",fontsize=8)
        for i,d in enumerate(delta_datasets):
            delta_rows.append({"baseline":base,"checkpoint":new,"dataset":d,
                               "metric":"balanced_accuracy" if d=="vlguard_request" else "accuracy",
                               "lm_delta_pp":values[0][i],"probe_delta_pp":values[1][i]})
        ax.set_yticks(y,["POPE","MME","SQA","VQA-MC","COCO choice","VLGuard (balanced)"])
        ax.invert_yaxis();ax.axvline(0,color="#48515A",linewidth=.8);ax.set_xlim(-72,22)
        ax.set_xlabel("Checkpoint minus baseline (percentage points)")
        title=f"{PLOT_LABELS[MODELS.index(base)]} → {PLOT_LABELS[MODELS.index(new)]}"
        if base=="openvla_oft":title="OFT Spatial ↔ OFT Goal (suite comparison)"
        ax.set_title(title,fontweight="bold",fontsize=11);ax.grid(axis="x",alpha=.15)
    axes[0,0].legend(loc="lower left",frameon=False)
    save_figure(fig,"checkpoint_changes")
    write_csv(ASSETS/"checkpoint_changes.csv",delta_rows)

    # Distinct safety tasks retain their own metric and chance reference.
    fig,axes=plt.subplots(2,1,figsize=(15,10.3),layout="constrained")
    x=np.arange(11)
    for ax,kind in zip(axes,["vlguard_request","siuo_mcqa"],strict=True):
        for j,(key,color,label) in enumerate(zip(["test_likelihood","test_probe"],COLORS,["LM head","Probe"],strict=True)):
            values=([100*metrics[m,kind][key]["balanced_acc"] for m in MODELS] if kind=="vlguard_request" else
                    [100*siuo_common[m]["lm_head_accuracy" if key=="test_likelihood" else "probe_accuracy"] for m in MODELS])
            bars=ax.bar(x+(j-.5)*.36,values,width=.34,color=color,label=label)
            ax.bar_label(bars,fmt="%.1f",fontsize=8,padding=2,rotation=90)
        chance=50 if kind=="vlguard_request" else 25
        ax.axhline(chance,color="#444444",linestyle="--",linewidth=.9,label=f"Reference {chance}%")
        ax.set_ylim(0,110);ax.set_xticks(x,PLOT_LABELS,rotation=35,ha="right")
        ax.set_ylabel("Balanced accuracy (%)" if kind=="vlguard_request" else "Accuracy (%)")
        ax.set_title("VLGuard: request classification" if kind=="vlguard_request" else "SIUO: safest-response choice, common 26 items (exploratory)",fontweight="bold")
        ax.grid(axis="y",alpha=.15);ax.legend(ncols=3,frameon=False,loc="upper right")
    save_figure(fig,"safety_overview")

    fig,axes=plt.subplots(2,1,figsize=(15,9.5),layout="constrained")
    for ax,dataset in zip(axes,["coco_capchoice","vlguard_request"],strict=True):
        for j,(key,color,label) in enumerate(zip(["probe_acc_same_items_normal","probe_acc"],["#498B76","#A7B0BA"],["Correct image","Shuffled image"],strict=True)):
            values=[100*metrics[m,dataset]["imgshuf"][key] for m in MODELS]
            bars=ax.bar(x+(j-.5)*.36,values,width=.34,color=color,label=label)
            ax.bar_label(bars,fmt="%.0f",fontsize=8,padding=2)
        ax.set_xticks(x,PLOT_LABELS,rotation=35,ha="right");ax.set_ylim(0,112);ax.set_ylabel("Probe accuracy (%)")
        ax.set_title(("COCO caption choice" if dataset=="coco_capchoice" else "VLGuard request classification")+": same 100 test items",fontweight="bold")
        ax.legend(frameon=False,ncols=2,loc="upper right");ax.grid(axis="y",alpha=.15)
        if dataset=="coco_capchoice":ax.axhline(25,color="#555555",linestyle="--",linewidth=.7)
    save_figure(fig,"image_shuffle")

    def table(key):
        lines=["| 평가 조건 | POPE | MME | SQA | VQA 선택 | COCO 선택 |","|---|---:|---:|---:|---:|---:|"]
        for model in MODELS:
            lines.append("| "+NAMES[model]+" | "+" | ".join(percent(metrics[model,d][key]["acc"]) for d in UTILITY)+" |")
        lines.append("| 무작위 선택 기준 | 50.0 | 50.0 | 35.4 | 34.6 | 25.0 |")
        return lines

    md=["# VLA·VLM Utility와 Safety: 전체 실험 통합 결과","",
        "작성일: 2026-11-05 · **11개 평가 조건 × 7개 데이터셋 = 77개 조합** · 단위: % 또는 %p","",
        "기존 backbone 실험과 LIBERO 체크포인트 추가 실험을 한곳에 모았다. PaliGemma-LB는 같은 가중치의 전처리 조건이므로, 가중치가 다른 모델·체크포인트는 10개이고 표의 평가 조건은 11개다. 새 추론이나 probe 학습 없이 저장된 결과를 재집계했다.","",
        "## 먼저 볼 결과","",
        "- **Prismatic은 일반 시각·언어 과제에서 두 판독 방식 모두 높다.** PaliGemma도 POPE와 일부 probe에서 높은 성능을 보인다.",
        "- **π0.5 base에는 기존 head보다 probe로 더 잘 읽히는 정보가 있다.** POPE는 58.4/88.0%, COCO는 73.3/90.1%(head/probe)다.",
        "- **π0.5-LIBERO에서는 VQA와 COCO의 두 방식 모두 크게 낮아졌다.** Probe는 각각 73.2→34.4%, 90.1→26.4%다.",
        "- **VLGuard의 높은 probe 점수는 실제 안전 행동을 뜻하지 않는다.** 이미지 교체 후에도 많은 분류 성능이 유지된다. SIUO는 공통 26문항의 탐색적 비교다.","",
        "## 1. 표와 그래프의 읽는 법","",
        "**Head(기존 LL)**는 기존 LM head의 첫 답변 토큰 likelihood를 이용한 **후보 선택 정확도**다. 전체 vocabulary에서 자유 생성한 답변의 정확도나 평균 log-likelihood 자체가 아니다. **Probe**는 마지막 유효 prompt 토큰의 final-normalized 벡터에 학습한 선형 분류기의 정확도다. VLA 본체는 고정하고 probe만 train에서 학습, dev에서 C 선택, test에서 평가했다.","",
        "Utility는 모두 accuracy지만 과제별 선택지 수와 무작위 기준이 다르다. VLGuard는 class 불균형을 고려해 balanced accuracy를 중심으로 본다. SIUO는 모든 조건에 공통인 26개 test 문항으로 맞췄다. 서로 다른 과제를 평균내 하나의 Utility/Safety 총점으로 만들지 않았다.","",
        "### 어떤 실행값을 통합했는가","",
        "| 조건 | 역할 | 본문에 사용하는 값 |","|---|---|---|",
        "| PaliGemma / PaliGemma-LB | 기준 VLM / Pi 방식 letterbox 대조 조건 | v1 |",
        "| π0 base / π0.5 base | 기준 VLA | v1; π0 SQA probe만 동일 v1 특징으로 수렴 보완한 v2 값 |",
        "| π0 LIBERO / π0.5 LIBERO | 추가 VLA 체크포인트 | v2 |",
        "| Prismatic | OpenVLA 초기화 계열의 기준 VLM | v1 |",
        "| OpenVLA base / Spatial | Base와 LIBERO-Spatial | 모두 v2의 물리적 batch size 1 재평가 |",
        "| OFT Spatial / Goal | 별도 suite 체크포인트 | 모두 v2의 물리적 batch size 1 평가 |","",
        "원래 v1의 OpenVLA/OFT batch 평가와 v2 단독 실행을 중복 모델로 세거나 평균내지 않았다. 본문은 최신 검증값을 쓰며, 모든 이전 실행값은 [92개 원본 결과 기록](assets/vla_vlm_combined/all_runs.csv), 변경된 15개 항목은 [이전값과 최신값 비교](assets/vla_vlm_combined/historical_vs_canonical.csv)에서 볼 수 있다. CSV의 `source_run`, `metrics_path`, `is_canonical`로 출처를 추적할 수 있다.","",
        "## 2. Utility 전체 결과","",
        "![Utility 전체 결과](assets/vla_vlm_combined/utility_overview.png)","",
        "왼쪽은 head, 오른쪽은 probe다. 두 그림 모두 같은 0~100% 색 범위를 쓴다. 열 아래의 chance를 함께 봐야 한다. 예를 들어 SQA 35%와 COCO 35%는 같은 수준의 무작위 대비 성능을 뜻하지 않는다. [벡터 그래프 SVG](assets/vla_vlm_combined/utility_overview.svg)","",
        "### 기존 LM head의 후보 선택 정확도","",
        *table("test_likelihood"),"","### Linear probe 정확도","",*table("test_probe"),"",
        "POPE·MME test는 각각 1,000·474문항이고, SQA·VQA 선택·COCO 선택은 각각 1,000문항이다. VQA 선택은 공식 자유 응답 VQA accuracy가 아니며 COCO 선택은 캡션 생성/CIDEr가 아니다. POPE·MME는 이미지 단위 연구 분할이며 MME 공식 total score와도 구분한다.","",
        "**해석:** π0.5 base의 높은 probe와 큰 head/probe 차이는 마지막 표현에 추가로 읽을 수 있는 신호가 있음을 보여준다. LIBERO 체크포인트에서는 VQA·COCO probe도 무작위 기준 근처가 된다. 반면 SQA probe는 거의 같아 모든 과제가 같은 방향으로 변한 것은 아니다. 낮은 점수는 이번 prompt·위치·학습량·선형 판독 방식에서의 결과이며 모든 layer의 정보 부재를 증명하지 않는다.","",
        "## 3. 같은 계열에서 체크포인트가 바뀌면","",
        "![체크포인트별 변화](assets/vla_vlm_combined/checkpoint_changes.png)","",
        "추가 체크포인트에서 기준 값을 뺀 변화량(%p)이다. 음수는 하락, 양수는 상승이다. Utility는 accuracy, VLGuard는 balanced accuracy의 변화이며 서로 평균내지 않았다. OFT 패널은 Spatial을 기준으로 Goal을 비교한 것으로 순차 fine-tuning 전후가 아니다. [SVG](assets/vla_vlm_combined/checkpoint_changes.svg) · [그래프 데이터 CSV](assets/vla_vlm_combined/checkpoint_changes.csv)","",
        "| 비교 | POPE probe | VQA probe | COCO probe | VLGuard probe balanced acc |","|---|---:|---:|---:|---:|"]
    for base,new in PAIRS:
        values=[100*(metrics[new,d]["test_probe"]["acc"]-metrics[base,d]["test_probe"]["acc"]) for d in ["pope_yn","vqav2_mc","coco_capchoice"]]
        values.append(100*(metrics[new,"vlguard_request"]["test_probe"]["balanced_acc"]-metrics[base,"vlguard_request"]["test_probe"]["balanced_acc"]))
        md.append("| "+NAMES[base]+" → "+NAMES[new]+" | "+" | ".join(f"{v:+.1f}" for v in values)+" |")
    md += ["","π0.5의 변화가 가장 크다. OpenVLA-Spatial도 POPE probe가 66.3→58.8%로 낮아졌지만, SQA 등 일부 과제는 소폭 높다. OFT 두 suite의 Utility는 대체로 낮고 작은 차이가 많다. 반복 seed나 통계적 유의성 검정을 하지 않았으므로 작은 차이를 안정적인 순위로 읽지 않는다."]
    md += ["","## 4. Safety 전체 결과","",
           "![Safety 전체 결과](assets/vla_vlm_combined/safety_overview.png)","",
           "위는 VLGuard 요청 분류의 balanced accuracy, 아래는 SIUO 공통 26문항의 accuracy다. 두 과제는 label과 측정 능력이 다르다. [SVG](assets/vla_vlm_combined/safety_overview.svg)","",
           "### VLGuard: 요청을 수행하면 유해한지 분류","",
           "Test 1,000문항 중 유해 요청은 63.6%다. 항상 유해하다고 답해도 accuracy는 63.6%이지만 balanced accuracy는 50%다. 각 칸은 **head / probe**이며, 마지막 열은 이미 학습한 probe로 같은 100문항을 평가한 결과다.","",
           "| 평가 조건 | Accuracy | Balanced accuracy | 이미지 정상 → 교체, probe |","|---|---:|---:|---:|"]
    for model in MODELS:
        r=metrics[model,"vlguard_request"];ll=r["test_likelihood"];pr=r["test_probe"];sh=r["imgshuf"]
        md.append(f"| {NAMES[model]} | {percent(ll['acc'])} / {percent(pr['acc'])} | {percent(ll['balanced_acc'])} / {percent(pr['balanced_acc'])} | {100*sh['probe_acc_same_items_normal']:.0f} → {100*sh['probe_acc']:.0f} |")
    md += ["","**해석:** Prismatic·PaliGemma·π0.5 base는 probe에서 매우 높은 요청 분류 점수를 보인다. π0.5 LIBERO에서는 probe balanced accuracy가 97.7→82.0%, π0 LIBERO에서는 80.7→53.5%로 낮아진다. 기존 head는 많은 모델에서 한쪽 class로 편향되어 있으며 단순 accuracy만으로 판단하면 오해할 수 있다. 예를 들어 OpenVLA-Spatial head accuracy는 51.0%지만 balanced accuracy는 49.1%다.","",
           "높은 probe 점수도 아래 이미지 교체 통제를 함께 봐야 한다. 이를 시각적 안전 이해, 실제 거절 응답 생성, 로봇의 안전 행동으로 해석하지 않는다.","",
           "### SIUO: 가장 안전하고 적절한 응답 선택","",
           "모든 조건에 공통인 test 26문항을 사용한다. 각 값 옆 괄호는 정답 문항 수다. 원래 test는 Pi/PaliGemma 계열 26문항, Prismatic/OpenVLA/OFT 계열 32문항이며 원래 점수도 CSV에 보존했다.","",
           "| 평가 조건 | Head accuracy | Probe accuracy | 원래 test 수 |","|---|---:|---:|---:|"]
    for model in MODELS:
        r=siuo_common[model];md.append(f"| {NAMES[model]} | {percent(r['lm_head_accuracy'])} ({r['lm_correct']}/26) | {percent(r['probe_accuracy'])} ({r['probe_correct']}/26) | {r['original_n']} |")
    md += ["","무작위 기준은 25%이고 한 문항이 약 3.85%p다. Prismatic과 π0.5 base의 head는 공통 문항에서 모두 11/26=42.3%다. 표본이 작고 train/dev 수는 모델별 길이 필터에 따라 다르므로, 이 표로 강한 안전성 순위를 확정하지 않는다. [공통 26문항 집계 CSV](assets/vla_vlm_combined/siuo_common26.csv)","",
           "## 5. 이미지를 바꾸면 점수가 유지되는가","",
           "![이미지 교체 통제](assets/vla_vlm_combined/image_shuffle.png)","",
           "각 막대 쌍은 **동일한 test 100문항**의 probe 정확도다. 전체 test 표와 표본 수가 다르다. 질문·선택지·정답을 유지하고 다른 이미지로 교체했다. 교체 후 probe를 재학습하지 않았다. [SVG](assets/vla_vlm_combined/image_shuffle.svg)","",
           "COCO에서는 Prismatic·π0.5 base·PaliGemma의 높은 성능이 이미지 교체 후 크게 낮아진다. 올바른 시각 입력이 점수에 기여했다는 근거다. 반면 VLGuard는 많은 모델에서 요청 분류 점수가 상당 부분 유지된다. 요청 텍스트나 데이터셋 단서의 기여가 클 가능성을 시사한다. 이미지가 없는 text-only 실험은 아니므로 텍스트만으로 충분함을 증명한 것은 아니다. 원래 성능이 낮은 모델에서 변화가 작다고 해서 이미지 정보를 전혀 사용하지 않는다고 단정할 수도 없다.","",
           "POPE의 v1 보조 집계에는 중복 문항 ID 문제가 있어 통합 이미지 교체 그림에서 제외했다. POPE 전체 accuracy는 저장된 행 기준으로 검산했으며, v2의 새 ID는 행 번호로 구분했다.","",
           "## 6. 비교 조건과 보완 사항","",
           "1. **Head 정확도는 후보 제한 평가다.** 전체 vocabulary의 최상위 토큰이 action 토큰이어도 정해진 후보 중 하나를 고른다. 자유 생성의 정답률이나 모델 간 절대 token probability 순위로 해석하지 않는다.",
           "2. **HF 기준 모델은 단독 실행으로 교체했다.** 새 OpenVLA-Spatial/SQA와 OFT-Goal/SIUO에서 BF16 singleton/padded-batch argmax가 달랐다. Padding/eager 변경으로도 남아 OpenVLA 계열을 물리적 batch size 1로 고정하고 기준 모델도 같은 조건으로 재평가했다. 합성 호출 일치는 원래 padded batch의 안정성을 증명하지 않는다.",
           "3. **수렴 보완을 반영했다.** 부모 프로세스의 warning 포착만으로는 worker의 경고를 놓칠 수 있었다. 실제 반복 횟수를 확인하고 π0 기준 SQA와 OFT 기준 SQA probe를 같은 특징으로 재최적화했다. 본문 π0 SQA probe는 37.7%이며 원 v1의 37.2%는 역사 기록에 남긴다. 이 통합본의 선택된 probe 77개는 실제 반복 상한 미만이다.",
           "4. **계열 간 조건은 완전히 같지 않다.** PaliGemma 계열과 Prismatic 계열은 backbone·전처리·tokenizer·학습 단계가 다르다. PaliGemma-LB는 Pi 방식 letterbox의 영향을 확인하는 보조 조건이다. 큰 격차 전체를 action 학습 하나의 인과 효과로 분리하지 않았다.",
           "5. **분류·선택 능력의 평가다.** VQA/COCO는 선택형 재구성, VLGuard는 요청 분류, SIUO는 안전한 응답 선택이다. LlavaGuard는 접근 승인 문제로 미실행이며 이 표에 0점으로 넣지 않았다.","",
           "## 7. 원본과 재현 자료","",
           "- [통합 주 결과 CSV: 77개 조합, 출처 포함](assets/vla_vlm_combined/canonical_results.csv)",
           "- [이전 실행까지 포함한 92개 결과](assets/vla_vlm_combined/all_runs.csv) / [변경된 15개 항목](assets/vla_vlm_combined/historical_vs_canonical.csv)",
           "- [기존 실험 설명](VLA_VLM_Utility_Safety_Explained.md) / [추가 체크포인트 결과](VLA_VLM_Checkpoint_Expansion_Results.md)",
           "- [v1 원본 요약](../runs/backbone_eval_v1/summary.md) / [v2 원본 요약](../runs/backbone_eval_v2_checkpoints/summary.md)",
           "- [추가 실험 실행 기록](VLA_VLM_Checkpoint_Expansion_Log.md) / [통합 생성·검증 기록](assets/vla_vlm_combined/verification.json)","",
           "그래프의 모델명과 축은 내보내기 호환성을 위해 영어로 표기했다. 표·그래프는 저장된 예측과 metrics에서 생성했으며 원본 결과를 수정하거나 GPU 실험을 재실행하지 않았다."]
    REPORT.write_text("\n".join(md)+"\n")
    verification={"status":"verified","canonical_combinations":77,"archived_result_records":92,"superseded_records":15,
                  "conditions":11,"datasets":7,"siuo_common_test_n":26,"prediction_accuracy_checks":154,
                  "selected_probe_iteration_counts_checked":77,"input_metric_sha256":inputs,
                  "siuo_common_item_ids":sorted(common),"report":str(REPORT.relative_to(ROOT)),
                  "generator":str(pathlib.Path(__file__).relative_to(ROOT)),
                  "matplotlib_version":matplotlib.__version__}
    (ASSETS/"verification.json").write_text(json.dumps(verification,indent=2,ensure_ascii=False)+"\n")
    print(f"Wrote {REPORT}: 77 canonical rows, 92 archived rows, 4 figures (PNG/SVG).")


if __name__ == "__main__":
    main()
