"""Create auditable tables and standalone figures for the completed probe study."""

import csv
from datetime import datetime
import html
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from representation_common import MODELS, RUN, TEACHER, now
from rollout_common import write_json

DISPLAY = {"baseline": "LIBERO base", "ae": "AE-only", "vlm": "VLM-only", "all": "Both"}
VIEWS = (("both", "initial"), ("both", "all"), ("teacher", "all"))


def main():
    report = RUN / "report"
    report.mkdir(parents=True, exist_ok=True)
    results, audit = [], []
    for source, view in VIEWS:
        completion = json.loads((RUN / "probes" / source / view / "completion.json").read_text())
        assert completion["complete"] and completion["sites"] == 312
        rows = json.loads((RUN / "features" / source / view / "rows.json").read_text())
        for model in MODELS:
            folder = RUN / "features" / source / view
            hook = json.loads((folder / f"{model}_hook_audit.json").read_text())
            assert hook["hook_action_max_abs_difference"] == 0
            with h5py.File(folder / f"{model}.h5", "r") as features:
                assert features.attrs["complete"] and features["written"][:].all()
                assert len(rows) == len(features["written"])
            if model == "ae":
                assert hook["all_vlm_features_exact_to_baseline"]
            audit.append({"source": source, "view": view, "model": model,
                          "inputs": len(rows), "hook_max_abs_difference": 0.})
            paths = list((RUN / "probes" / source / view / model).rglob("metrics.json"))
            assert len(paths) == 78
            for path in paths:
                result = json.loads(path.read_text())
                assert all(c["converged"] for f in result["folds"] for c in f["candidates"])
                results.append(result)
    lookup = {(row["source"], row["view"], row["model"], row["site"]): row for row in results}
    for source, view in VIEWS:
        for layer in range(18):
            site = f"vlm/layer_{layer:02d}"
            a, b = [lookup[source, view, model, site] for model in ("baseline", "ae")]
            assert a["fold_mean_AUROC"] == b["fold_mean_AUROC"]
    fields = ["source", "view", "model", "site", "fold_mean_AUROC", "fold_sd_AUROC", "fold_mean_balanced_accuracy"]
    with (report / "all_sites.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in results)
    images = []
    for source, view in VIEWS:
        figure, axes = plt.subplots(1, 4, figsize=(17, 4), sharey=True)
        for axis, module in zip(axes, ("vlm", "ae/flow_01", "ae/flow_05", "ae/flow_10"), strict=True):
            for model in MODELS:
                values = [lookup[source, view, model, f"{module}/layer_{layer:02d}"] for layer in range(18)]
                means = np.asarray([row["fold_mean_AUROC"] for row in values])
                errors = np.asarray([row["fold_sd_AUROC"] for row in values])
                axis.plot(range(18), means, label=DISPLAY[model], marker=".", linewidth=1.5,
                          linestyle="--" if model == "ae" else "-")
                axis.fill_between(range(18), np.clip(means - errors, 0, 1), np.clip(means + errors, 0, 1), alpha=.10)
            axis.axhline(.5, color="gray", linestyle="--", linewidth=1)
            axis.set(title=module, xlabel="Block layer", ylim=(0, 1), xticks=[0, 4, 8, 12, 17])
            axis.grid(alpha=.2)
        axes[0].set_ylabel("Test AUROC, fold mean ± fold SD")
        axes[-1].legend(fontsize=8, loc="lower right")
        figure.suptitle(f"{source} / {view}: family-held-out linear probes")
        figure.tight_layout()
        name = f"{source}_{view}_layers.png"
        figure.savefig(report / name, dpi=180)
        plt.close(figure)
        images.append(name)
        if view == "all":
            figure, axes = plt.subplots(1, 3, figsize=(15, 4))
            for axis, site in zip(axes[:2], ("vlm/layer_17", "ae/flow_10/layer_17"), strict=True):
                for model in MODELS:
                    values = lookup[source, view, model, site]["decision_timestep_oof"]
                    valid = [row for row in values if row["valid_folds"] == 5]
                    axis.plot([row["timestep"] for row in valid], [row["fold_mean_AUROC"] for row in valid],
                              label=DISPLAY[model])
                axis.axhline(.5, color="gray", linestyle="--", linewidth=1)
                axis.set(title=site, xlabel="Re-inference control timestep", ylabel="Fold mean AUROC", ylim=(0, 1))
                axis.grid(alpha=.2)
            axes[0].legend(fontsize=8)
            values = lookup[source, view, "baseline", "vlm/layer_17"]["decision_timestep_oof"]
            axes[2].plot([row["timestep"] for row in values], [row["samples"] for row in values], color="black")
            axes[2].set(title="Available reference episodes", xlabel="Control timestep", ylabel="Episode count")
            figure.suptitle(f"{source}: held-out predictions of the all-timestep probe; no per-timestep refitting")
            figure.tight_layout()
            name = f"{source}_decision_timesteps.png"
            figure.savefig(report / name, dpi=180)
            plt.close(figure)
            images.append(name)
    selected, success_selected = [], []
    # These sites were fixed before inspecting test performance; no test-best-layer selection.
    for source, view in VIEWS:
        for model in MODELS:
            row = {"source": source, "view": view, "model": model}
            success_row = dict(row)
            for key, site in (("VLM_17", "vlm/layer_17"), ("AE_17_flow10", "ae/flow_10/layer_17"),
                              ("velocity_pre_flow10", "velocity_pre/flow_10"),
                              ("velocity_post_flow10", "velocity_post/flow_10")):
                entry = lookup[source, view, model, site]
                row[key] = {"AUROC": entry["fold_mean_AUROC"], "fold_sd": entry["fold_sd_AUROC"],
                            "balanced_accuracy": entry["fold_mean_balanced_accuracy"]}
                success_values = [value["AUROC"] for value in entry["both_successful_pair_subset_folds"]
                                  if value["AUROC"] is not None]
                success_row[key] = {"AUROC": float(np.mean(success_values)) if success_values else None,
                    "fold_sd": float(np.std(success_values, ddof=1)) if len(success_values) > 1 else None,
                    "valid_folds": len(success_values)}
                success_row["episodes"] = entry["both_successful_pair_subset_oof"].get("episodes", 0)
                success_row["layout_pairs"] = entry["both_successful_pair_subset_oof"].get("paired_episodes", 0)
            selected.append(row)
            success_selected.append(success_row)
    teacher = json.loads((TEACHER / "completion.json").read_text())
    integrity = json.loads((RUN / "input_integrity_audit.json").read_text())
    protocol = json.loads((RUN / "experiment_protocol.json").read_text())
    assert teacher["complete"] and teacher["accepted"] == 676
    completed_utc = now()
    started_utc = protocol.get("execution_started_utc")
    elapsed_seconds = ((datetime.fromisoformat(completed_utc) - datetime.fromisoformat(started_utc)).total_seconds()
                       if started_utc else None)
    summary = {"complete": True, "completed_utc": completed_utc, "feature_and_probe_conditions": len(results),
        "five_fold_selected_probe_fits": len(results) * 5,
        "regularization_candidate_evaluations": len(results) * 5 * 3,
        "teacher": teacher, "audits": audit, "fixed_site_results": selected,
        "matched_both_success_subset_fixed_sites": success_selected,
        "input_integrity": integrity, "execution_started_utc": started_utc,
        "execution_wall_seconds": elapsed_seconds, "policy_config": protocol["policy_config"],
        "policy_training": False, "lda": False,
        "limitations": ["Authored semantic task labels; motor success is a separate proxy, not observed harm.",
            "Same semantic scenarios appeared in LoRA motor training; families are held out only from probe fitting.",
            "No instruction-only, modality, state/action-only, or random-label controls were run, as requested.",
            "Linear decodability does not establish causal warning transmission or use by the action expert.",
            "Velocity post features contain all 32 internal channels; the executed action has 7 dimensions.",
            "Fold SD is descriptive variation across folds, not a confidence interval.",
            "Episode termination changes the available task mix at later absolute timesteps; episode counts are plotted.",
            "VLM residual outputs are not direct measurements of the per-layer KV tensors consumed by the expert.",
            "The original pi05_libero config uses pi05=True and discrete_state_input=False: robot state passes through policy preprocessing but is not embedded as a VLM or AE state token. This setting was preserved across all four models.",
            "Two semantic pairs have identical initial model inputs with opposite safety labels (four layout-pairs, eight episodes); those initial labels are not inferable from the supplied input. All conditions are retained and flagged.",
            "Timesteps and layout variants are correlated observations, not additional independent semantic scenarios: 169 pairs and 35 families.",
            "Low accuracy of a token-mean linear probe does not establish absence of localized or nonlinear safety information.",
            "Mean-pooled logits across time summarize an episode retrospectively; decision-time metrics are also retained."]}
    write_json(report / "summary.json", summary)
    def table_rows(items):
        cells = []
        for row in items:
            values = [html.escape(str(row[key])) for key in ("source", "view", "model")]
            for key in ("VLM_17", "AE_17_flow10", "velocity_pre_flow10", "velocity_post_flow10"):
                value = row[key]
                values.append(f"{value['AUROC']:.3f} ± {value['fold_sd']:.3f}"
                              if value["AUROC"] is not None and value["fold_sd"] is not None else "N/A")
            cells.append("<tr>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>")
        return "".join(cells)
    document = """<!doctype html><html lang="ko"><meta charset="utf-8"><title>Semantic safety linear probes</title>
<style>body{font-family:system-ui;max-width:1250px;margin:32px auto;line-height:1.6}table{border-collapse:collapse;width:100%}
th,td{border:1px solid #bbb;padding:7px}img{width:100%}code{background:#eee}</style>
<h1>안전·위험 representation: Linear probe</h1>
<p>공식 LIBERO FT π0.5 및 AE-only·VLM-only·Both 16k LoRA 정책을 고정하고,
공통 입력에서 0–17층 residual output과 velocity projection 전후를 추출했다.
Both 평가 trajectory와 같은 676개 초기 배치에서 새로 수집한 Teacher trajectory를 각각 분석했다.
최초 입력은 독립 검증에서 정확히 일치하여 초기 시점 분석은 공유한다. 시간축은 평균하지 않았다.</p>
<p>저장된 이미지·지시문·robot state와 공통 초기 noise로 각 시점에서 독립 추론했다.
Teacher의 정답 action은 모델 입력으로 주지 않았다. 원래 pi05_libero 설정의
<code>pi05=True, discrete_state_input=False</code>를 유지하여, robot state는 전처리를 통과하지만
VLM 또는 AE의 별도 state token으로 들어가지는 않는다.</p>
<p>35개 family의 5-fold 분할을 모든 모델·출처·층에 공유한다. 각 fold는 학습 21개, validation 7개,
test 7개 family를 사용한다. 표준화는 학습 표본에만 맞추고 C∈{0.01,0.1,1}를 validation AUROC로 선택했다.
상황 family별·에피소드별 기여도를 맞추며, 한 trajectory의 모든 시점과 안전·위험 대조쌍은 같은 분할에 속한다.
VLA 추가 학습과 LDA는 수행하지 않았다.</p>
<p>기존 정책 평가의 test 배치 676개를 probe용 학습·validation·test로 다시 분할한 것이다.
초기 이미지·지시문·state·noise가 완전히 같지만 label이 반대인 semantic 대조쌍 2개(배치 포함 4쌍, 8개 에피소드)를
입력 검증에서 확인했다. 전체 조건을 유지하고, 이 조건은 초기 입력만으로 구분할 수 없음을 결과 해석에 반영한다.</p>
<h2>사전에 지정한 위치의 test AUROC</h2><p>평균 ± fold 표준편차. Test에서 가장 잘 나온 층을 선택한 표가 아니다.</p>
<table><thead><tr><th>출처</th><th>시점</th><th>모델</th><th>VLM 17</th><th>AE 17 / flow 10</th>
<th>Velocity 직전 / flow 10</th><th>Velocity 직후 / flow 10</th></tr></thead><tbody>"""
    document += table_rows(selected) + "</tbody></table>"
    document += ("<h2>Both 참조에서 양쪽이 성공한 대조쌍의 결과</h2><p>안전·위험 양쪽의 motor proxy가 성공한 "
        "189개 배치 대조쌍(378개 에피소드, 118개 semantic 대조쌍)을 선택했다. 같은 조건을 Teacher에도 적용했다. "
        "전체 조건으로 학습한 동일 probe의 held-out 예측을 이 부분집합에서 평가했으며 재학습하지 않았다. "
        "성공은 실제 위해 발생이나 지시문의 모든 물리적 효과를 뜻하지 않는다.</p>"
        "<table><thead><tr><th>출처</th><th>시점</th><th>모델</th><th>VLM 17</th><th>AE 17 / flow 10</th>"
        "<th>Velocity 직전 / flow 10</th><th>Velocity 직후 / flow 10</th></tr></thead><tbody>")
    document += table_rows(success_selected) + "</tbody></table><h2>전체 layer 비교</h2>"
    document += "".join(f'<img src="{name}" alt="layer-wise AUROC">' for name in images)
    document += "<h2>해석 범위</h2><ul>" + "".join(f"<li>{html.escape(value)}</li>" for value in summary["limitations"])
    document += "</ul><p><a href=\"all_sites.csv\">전체 위치 CSV</a> · <a href=\"summary.json\">검증·설정·요약 JSON</a></p></html>"
    (report / "index.html").write_text(document)
    print(json.dumps({"report": str(report / "index.html"), "conditions": len(results)}))


if __name__ == "__main__":
    main()
