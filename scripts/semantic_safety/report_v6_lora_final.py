"""Build paired comparisons, a local video review, and standalone learning curves."""

import argparse
from collections import defaultdict
import csv
import html
import json
import os
from pathlib import Path
import shutil

import h5py
import numpy as np

from rollout_common import ROOT, write_json
from v6_lora_data import DATA, RUN

MODELS = ["baseline", "ae", "vlm", "all"]
KEYS = ["ever_grasped", "ever_lifted_4cm", "transport_success", "sustained_transport_success",
        "xy_only_transport_success", "xy_only_sustained_success", "native_success"]
EXPORT = ROOT / "checkpoints/pi05_libero_v6_lora_v1"


def load_results():
    completion = json.loads((RUN / "final_evaluation_complete.json").read_text())
    assert completion["complete"]
    results = {}
    for result in completion["rows"]:
        folder = Path(result["folder"])
        summary = json.loads((folder / "motor_summary.json").read_text())
        keep = set(result["evaluated_item_ids"])
        rows = {row["item_id"]: row for row in summary["rows"] if row["item_id"] in keep}
        assert len(rows) == result["overall"]["n"]
        results[result["benchmark"], result["model"]] = dict(folder=folder, rows=rows, overall=result["overall"])
    return results


def comparisons(results):
    output = []
    eligible = []
    cross_audit = []
    for bench in ["confirmation", "test", "v6", "native"]:
        base = results[bench, "baseline"]
        for scope in MODELS[1:]:
            target = results[bench, scope]
            assert base["rows"].keys() == target["rows"].keys()
            checks = []
            for key in base["rows"]:
                with h5py.File(base["folder"] / "episodes" / key / "trajectory.h5") as before, \
                        h5py.File(target["folder"] / "episodes" / key / "trajectory.h5") as after:
                    for name in ["main_rgb_raw", "wrist_rgb_raw", "state"]:
                        assert np.array_equal(before["policy_inputs/" + name][0], after["policy_inputs/" + name][0])
                    assert before["flow_noise_seeds"][0] == after["flow_noise_seeds"][0]
                    assert before.attrs["instruction"] == after.attrs["instruction"]
                checks.append(key)
            cross_audit.append(dict(benchmark=bench, model=scope, episodes=len(checks), identical_initial_inputs=True,
                                    identical_initial_noise_seed=True, identical_instructions=True))
            for metric in (["native_success"] if bench == "native" else KEYS[:-1]):
                gains = sum(not base["rows"][key][metric] and row[metric] for key, row in target["rows"].items())
                losses = sum(base["rows"][key][metric] and not row[metric] for key, row in target["rows"].items())
                output.append(dict(benchmark=bench, model=scope, metric=metric, n=len(target["rows"]),
                                   baseline=sum(bool(r[metric]) for r in base["rows"].values()),
                                   adapted=sum(bool(r[metric]) for r in target["rows"].values()), gains=gains, losses=losses))
        if bench == "native":
            continue
        for scope in MODELS:
            groups = defaultdict(list)
            for row in results[bench, scope]["rows"].values():
                groups[row["pair_group"]].append(row)
            for pair, rows in groups.items():
                assert len(rows) == 2 and {r["task_is_unsafe"] for r in rows} == {False, True}
                if all(row["transport_success"] for row in rows):
                    eligible.append(dict(benchmark=bench, model=scope, pair_group=pair,
                                         family_id=rows[0]["pair_id"],
                                         sustained_both=all(row["sustained_transport_success"] for row in rows),
                                         episodes=[dict(item_id=row["item_id"], unsafe=row["task_is_unsafe"],
                                                        condition_index=row["condition_index"],
                                                        instruction=row["instruction"], first_transport_step=row["first_transport_step"],
                                                        trajectory=str((results[bench, scope]["folder"] / "episodes" /
                                                                        row["item_id"] / "trajectory.h5").relative_to(ROOT)))
                                                   for row in rows]))
    write_json(RUN / "final_cross_model_input_audit.json", dict(passed=True, comparisons=cross_audit))
    write_json(RUN / "paired_model_comparisons.json", output)
    shared = {}
    for bench in ["confirmation", "test", "v6"]:
        sets = {scope: {row["pair_group"] for row in eligible if row["benchmark"] == bench and row["model"] == scope}
                for scope in MODELS}
        shared[bench] = dict(three_adapted_models=sorted(set.intersection(*(sets[scope] for scope in MODELS[1:]))),
                             baseline_and_three_models=sorted(set.intersection(*sets.values())))
    write_json(RUN / "representation_candidate_pairs.json", dict(pairs=eligible, features_extracted=False,
               shared_pairs=shared,
               caution="Success-filtered candidate trajectories only; motor success does not establish semantic safety awareness."))
    with (RUN / "paired_model_comparisons.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    return eligible


def curves():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = json.loads((RUN / "validation_grid_complete.json").read_text())["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for scope in MODELS[1:]:
        training = [json.loads(line) for line in (RUN / f"train_{scope}/training.jsonl").read_text().splitlines()]
        extension_log = RUN / f"train_{scope}_extension/training.jsonl"
        if extension_log.exists():
            training += [json.loads(line) for line in extension_log.read_text().splitlines()]
        value = np.array([row["active7_loss"] for row in training])
        window = min(100, len(value))
        smooth = np.convolve(value, np.ones(window) / window, mode="valid")
        axes[0].plot(np.arange(window, len(value) + 1), smooth, label=scope)
        rows = [row for row in grid if row["scope"] == scope]
        axes[1].plot([row["step"] for row in rows], [100 * row["transport_success"] / row["n"] for row in rows],
                     marker="o", label=scope)
    baseline = json.loads((RUN / "baseline_selection66_balanced/motor_summary.json").read_text())["overall"]
    axes[1].axhline(100 * baseline["transport_success"] / baseline["n"], color="black", linestyle=":",
                    linewidth=1, label="baseline")
    axes[0].set(xlabel="Optimizer steps", ylabel="Active 7D flow loss (100-step mean)", yscale="log")
    axes[1].set(xlabel="Optimizer steps", ylabel="Validation transport success (%)", ylim=(0, 100))
    if (RUN / "extension_protocol.json").exists():
        for axis in axes:
            axis.axvline(6000, color="gray", linestyle="--", linewidth=1)
    for axis in axes:
        axis.legend()
        axis.grid(alpha=.2)
    fig.savefig(RUN / "learning_curves.png", dpi=180)
    fig.savefig(RUN / "learning_curves.svg")
    plt.close(fig)


def condition_summary(results):
    with (DATA / "condition_coverage.csv").open() as file:
        coverage = {int(row["condition"]): row for row in csv.DictReader(file)}
    output = []
    for condition, source in sorted(coverage.items()):
        for model in MODELS:
            rows = [row for row in results["test", model]["rows"].values()
                    if row["condition_index"] == condition]
            assert len(rows) == 4
            assert all(row["pair_id"] == source["pair_id"] for row in rows)
            output.append(dict(condition=condition, family=source["pair_id"],
                               unsafe=rows[0]["task_is_unsafe"], actor=source["actor"],
                               destination=source["destination"], instruction=source["instruction"],
                               training_demonstrations=int(source["demonstrations"]), model=model,
                               n=len(rows), **{key: sum(bool(row[key]) for row in rows) for key in KEYS[:-1]}))
    write_json(RUN / "test_condition_summary.json", output)
    with (RUN / "test_condition_summary.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


def review(results):
    rows = []
    for bench in ["test", "v6", "confirmation", "native"]:
        for key, original in results[bench, "baseline"]["rows"].items():
            item = dict(benchmark=bench, item_id=key, instruction=original["instruction"],
                        unsafe=original.get("task_is_unsafe"), models={})
            for scope in MODELS:
                result = results[bench, scope]
                row = result["rows"][key]
                folder = result["folder"] / "episodes" / key
                item["models"][scope] = dict(video=os.path.relpath(folder / "views.mp4", RUN),
                                             initial=os.path.relpath(folder / "initial.png", RUN),
                                             success=bool(row.get("native_success") if bench == "native" else row["transport_success"]),
                                             sustained=bool(row.get("sustained_transport_success")),
                                             xy=bool(row.get("xy_only_transport_success")))
            rows.append(item)
    data = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")
    document = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>v6 π0.5 LoRA 비교</title><style>
body{font:16px system-ui;margin:24px;background:#f5f7fb;color:#172033}main{max-width:1440px;margin:auto}
h1{font-size:26px}select,button{font:inherit;padding:8px;margin:4px}label{display:inline-flex;align-items:center;max-width:100%}select{min-width:0;max-width:100%}#grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
article{background:white;padding:14px;border-radius:10px;min-width:0}video{width:100%;background:#111}small{display:block;color:#556}
.yes{color:#167247}.no{color:#a53232}#instruction{min-height:2em}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ccd}
@media(max-width:700px){body{margin:12px}#grid{grid-template-columns:1fr}}
</style><main><h1>v6 π0.5 LoRA 수행 비교</h1>
<p>동일 장면·지시·초기 noise seed의 기본 모델 / AE-only / VLM-only / 공동 LoRA 결과입니다.
안전·위험 task의 집기·운반을 평가합니다. 붓기·절단·타격 완료나 위해 발생 여부를 뜻하지 않습니다.</p>
<label>평가 집합 <select id="bench"><option>test</option><option>v6</option><option>confirmation</option><option>native</option></select></label>
<label>장면 <select id="scene"></select></label><button id="previous">이전</button><button id="next">다음</button>
<button id="play">동시에 처음부터 재생</button><p id="instruction"></p><div id="grid"></div>
<p><a href="learning_curves.svg">학습 곡선</a> · <a href="paired_model_comparisons.csv">동일 장면 비교 CSV</a> ·
<a href="representation_candidate_pairs.json">양쪽 운반에 성공한 safety/unsafety 쌍</a></p>
<small>새 모델의 모터 평가 영상은 정책 입력 시점의 2Hz 기록이며 모든 20Hz action/state는 trajectory.h5에 있습니다.
v6는 설계에 사용한 개발 benchmark이고 test는 동일 물체 종류의 새 배치입니다.</small></main>
<script>const rows=DATA;let current=[];const names={baseline:'기본 π0.5',ae:'AE-only',vlm:'VLM-only',all:'공동 LoRA'};
const bench=document.querySelector('#bench'),scene=document.querySelector('#scene');
function fill(){current=rows.filter(r=>r.benchmark===bench.value);scene.replaceChildren();current.forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=r.item_id;scene.append(o)});show()}
function show(){const row=current[Number(scene.value)];document.querySelector('#instruction').textContent=(row.benchmark==='native'?'native':row.unsafe?'위험 지시':'안전 지시')+' · '+row.instruction;const grid=document.querySelector('#grid');grid.replaceChildren();for(const [scope,m] of Object.entries(row.models)){const a=document.createElement('article'),h=document.createElement('h2'),s=document.createElement('p'),v=document.createElement('video');h.textContent=names[scope];s.textContent=row.benchmark==='native'?'공식 task '+(m.success?'성공':'실패'):(m.success?'운반 성공':'운반 실패')+' · 지속 성공 '+(m.sustained?'O':'X')+' · XY 운반 '+(m.xy?'O':'X');s.className=m.success?'yes':'no';v.controls=true;v.preload='metadata';v.poster=m.initial;v.src=m.video;a.append(h,s,v);grid.append(a)}}
bench.onchange=fill;scene.onchange=show;document.querySelector('#previous').onclick=()=>{scene.value=(Number(scene.value)-1+current.length)%current.length;show()};document.querySelector('#next').onclick=()=>{scene.value=(Number(scene.value)+1)%current.length;show()};document.querySelector('#play').onclick=()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})});fill();</script></html>'''
    (RUN / "review.html").write_text(document.replace("const rows=DATA;", "const rows=" + data + ";"))


def report(results, eligible):
    selection = json.loads((RUN / "selection_frozen.json").read_text())
    dataset = json.loads((DATA / "training_data_audit.json").read_text())
    protocol = json.loads((RUN / "experiment_protocol.json").read_text())
    extension = protocol.get("extension")
    grid = json.loads((RUN / "validation_grid_complete.json").read_text())["rows"]
    names = dict(baseline="기본 π0.5", ae="AE-only", vlm="VLM-only", all="공동 LoRA")
    labels = dataset.get("covered_conditions_by_label", dict(safe=31, unsafe=31))
    asset_names = dict(spoon="스푼", baby="아기", battery="버튼 배터리", cup="컵", wood_bowl="나무 그릇",
                       mat="매트", candle_lit="촛불")
    missing = ", ".join(asset_names.get(row["actor"], row["actor"]) + "→" +
                        asset_names.get(row["destination"], row["destination"])
                        for row in dataset["missing_conditions"])

    def score(bench, model, metric="transport_success"):
        row = results[bench, model]["overall"]
        return f"{row[metric]}/{row['n']} ({100 * row[metric] / row['n']:.1f}%)"

    lines = ["# v6 π0.5 모터 적응: VLM / AE / 공동 LoRA", "",
             "상태: **학습 및 최종 평가 완료**. 공식 LIBERO FT π0.5에서 출발한 세 LoRA 모델을 비교했다. "
             "아래 모터 성공은 **집기 → 들어올리기 → 목적지 부근 운반**의 성공이다. "
             "최종 붓기·절단·타격이나 위해 발생을 뜻하지 않으며, semantic safety awareness 자체는 아직 측정하지 않았다.", "",
             "## 최종 수행 결과", "",
             "Checkpoint는 방향 균형 validation 66개로 선택한 뒤 고정했다. Test 결과로 다시 학습하거나 checkpoint를 바꾸지 않았다.", "",
             "| 모델 | 새 배치 test 운반 | 원본 v6 운반 | Native LIBERO 성공 | 선택 step |",
             "|---|---:|---:|---:|---:|"]
    for model in MODELS:
        step = "—" if model == "baseline" else f"{selection['selected'][model]['step']:,}"
        lines.append(f"| {names[model]} | {score('test', model)} | {score('v6', model)} | "
                     f"{score('native', model, 'native_success')} | {step} |")
    lines += ["", "Native는 Spatial 8개 task × 초기 상태 2개인 16-episode 회귀 검사이다. "
              "전체 LIBERO benchmark 성능으로 일반화하지 않는다. 새 test도 동일한 물체 종류의 새로운 배치이며, 미관측 물체/위험 종류 평가가 아니다.", "",
              "| 모델 | Test 지속 운반 | Test XY-only 운반 | 확인용 validation 운반 | Test 양쪽 성공 쌍 (상황 수) |",
              "|---|---:|---:|---:|---:|"]
    for model in MODELS:
        pairs = sum(row["benchmark"] == "test" and row["model"] == model for row in eligible)
        families = len({row["family_id"] for row in eligible
                        if row["benchmark"] == "test" and row["model"] == model})
        lines.append(f"| {names[model]} | {score('test', model, 'sustained_transport_success')} | "
                     f"{score('test', model, 'xy_only_transport_success')} | {score('confirmation', model)} | "
                     f"{pairs}/132 ({families}/33) |")
    lines += ["", "| 모델 | 안전 지시 test 운반 | 위험 지시 test 운반 | "
              f"시연 있는 {dataset['conditions_with_demonstrations']}조건 | 시연 없는 {len(dataset['missing_conditions'])}조건 |",
              "|---|---:|---:|---:|---:|"]
    missing_indices = {row["condition"] for row in dataset["missing_conditions"]}
    subgroup_rows = []
    for model in MODELS:
        rows = list(results["test", model]["rows"].values())
        groups = {
            "safe": [r for r in rows if not r["task_is_unsafe"]],
            "unsafe": [r for r in rows if r["task_is_unsafe"]],
            "with_demonstrations": [r for r in rows if r["condition_index"] not in missing_indices],
            "without_demonstrations": [r for r in rows if r["condition_index"] in missing_indices],
        }
        cells = []
        for group, items in groups.items():
            count = sum(r["transport_success"] for r in items)
            cells.append(f"{count}/{len(items)} ({100 * count / len(items):.1f}%)")
            subgroup_rows.append(dict(model=model, group=group, n=len(items), transport_success=count))
        lines.append(f"| {names[model]} | " + " | ".join(cells) + " |")
    write_json(RUN / "test_subgroups.json", subgroup_rows)
    lines += ["", "| 모델 | Test 집기 관측 | Test 4cm lift 관측 | 기본 모델 대비 운반 추가 성공 / 상실 |",
              "|---|---:|---:|---:|"]
    baseline_rows = results["test", "baseline"]["rows"]
    for model in MODELS:
        rows = results["test", model]["rows"]
        gains = sum(not baseline_rows[key]["transport_success"] and row["transport_success"]
                    for key, row in rows.items())
        losses = sum(baseline_rows[key]["transport_success"] and not row["transport_success"]
                     for key, row in rows.items())
        difference = "—" if model == "baseline" else f"+{gains} / −{losses}"
        lines.append(f"| {names[model]} | {score('test', model, 'ever_grasped')} | "
                     f"{score('test', model, 'ever_lifted_4cm')} | {difference} |")
    lines += ["", "집기와 lift는 episode 중 한 번 이상 해당 상태가 관측됐는지를 센다. "
              "추가 성공/상실은 동일한 test 장면에서 기본 모델의 실패가 성공으로, 또는 성공이 실패로 바뀐 수이다. "
              "각 모델의 운반 성공은 현재 grasp와 연속 운반 조건까지 별도로 충족해야 한다."]
    lines += ["", "동일 장면의 개선/퇴보 수는 [paired_model_comparisons.csv](../runs/semantic_motor_v6_lora_v1/paired_model_comparisons.csv)에 있다. "
              "합계뿐 아니라 안전/위험 쌍을 모두 수행했는지도 집계했다. 본 비교는 각 조건을 한 번씩 학습한 결과이며 독립적인 반복 실행의 분산은 측정하지 않았다.", "",
              "## 학습 데이터", "",
              f"- 신규 시연 **{dataset['episodes']:,}개 / {dataset['control_steps']:,} control steps**. "
              f"66개 조건 중 {dataset['conditions_with_demonstrations']}개에서 최소 48개씩 확보했다. "
              f"조건 단위 안전/위험 수는 {labels['safe']}/{labels['unsafe']}이다.",
              "- Teacher의 모든 action은 실제 물리 step으로 실행했다. 채택한 시연은 독립 환경에서 전 action을 재실행하고 "
              "상태 및 성공을 확인했다. 이미지·state·action 캐시는 원본 배열과 정확히 일치한다.",
              "- 모터 조건 → 시연 → 10-step 시작점을 균등 추출하며, 네 가지 지시 표현은 위치/방향과 독립적으로 추출한다.",
              "- 학습 샘플의 10%는 공식 모델의 native 성공 시연 50개에서 뽑는다. "
              "Spatial 10개 task의 초기 상태 2–6을 사용하며, 새 native 회귀 평가의 초기 상태 0–1은 이 추가 학습에서 제외했다. "
              "이는 공식 checkpoint의 과거 LIBERO 학습에서도 미관측이었다는 뜻은 아니다.",
              "- 학생의 실제 모델 계산에는 main/wrist RGB와 지시가 사용된다. Privileged 물체 pose와 이진 안전 label은 입력하지 않는다. "
              "공식 π0.5 정규화 통계를 유지했다.",
              "- Train, validation 132개, test 264개는 별도 seed로 만들었으며 정확한 배치 중복을 배제했다. "
              "평가 장면은 teacher/model 성공 여부로 필터하지 않았다.",
              f"- {missing}의 **{len(dataset['missing_conditions'])}개 조건은 성공 시연을 확보하지 못했다**. "
              "이 조건도 평가에는 남아 있다. Teacher 탐색 실패를 물리적 불가능의 증거로 해석하지 않는다.",
              "- 앞쪽 목적지의 일부 train 배치는 도달 가능성을 고려해 조정했다. 지갑 시연 48개에는 세운 초기 자세를 사용했다. "
              "원본 v6와 holdout 배치는 변경하지 않았다.", "",
              "전체 조건별 확보 수와 검증 기록: [데이터 안내](../data/semantic_motor_v6_lora_v1/README.md), "
              "[coverage CSV](../data/semantic_motor_v6_lora_v1/condition_coverage.csv), "
              "[data audit](../data/semantic_motor_v6_lora_v1/training_data_audit.json).", "",
              "## 모델과 학습 설정", "",
              "모든 원본 가중치는 고정하고 선택한 모듈의 LoRA만 학습했다. "
              "Rank 16, alpha 32, dropout 0, batch 16, AdamW, 최대 learning rate 2e-4, "
              "warmup/cosine schedule, gradient clip 1, seed 20261111이다. "
              "세 조건은 동일 manifest와 샘플링/flow noise seed 설정을 사용한다.", "",
              "| 조건 | LoRA 범위 | 학습 파라미터 | 학습 steps | 원본 가중치 고정 |",
              "|---|---|---:|---:|---|"]
    descriptions = dict(ae="Action transformer·조건부 norm 선형층·action/time projection",
                        vlm="PaliGemma 영상·언어·멀티모달 선형층", all="VLM 및 AE의 위 범위")
    for scope in MODELS[1:]:
        config = json.loads((RUN / f"train_{scope}/config.json").read_text())
        audit = json.loads((RUN / f"train_{scope}/freeze_audit.json").read_text())
        assert audit["all_original_weights_unchanged"]
        steps_text = f"{config['steps_planned']:,}"
        if extension:
            steps_text += f" + {extension['additional_steps_per_scope']:,}"
        lines.append(f"| {names[scope]} | {descriptions[scope]} | {audit['trainable_parameters']:,} | "
                     f"{steps_text} | 통과 |")
    if extension:
        lines += ["", "VLM validation 운반이 4k의 38/66에서 6k의 43/66으로 증가해, test를 열기 전에 "
                  "세 조건 모두에 동일한 추가 2,000-step 단계를 정했다. 각자의 6k adapter에서 시작하며, "
                  "새 AdamW optimizer, 최대 learning rate 2e-5, warmup 200, cosine 최저 learning rate 2e-6, "
                  "seed 20271111을 사용했다. 기존 optimizer 상태를 이어받은 연속 재개는 아니다. "
                  "본 비교의 총 학습량은 각 8,000 updates이고, 선택 후보는 누적 2k/4k/6k/7k/8k이다. "
                  "추가 단계 adapter 내부의 `metadata.step`은 해당 단계 안의 step이며 export의 `training_history.json`에 누적 step을 기록했다.", ""]
    lines += ["", "AE-only에서는 동일 입력의 VLM hidden state도 학습 전후 정확히 같았다. "
              "VLM/공동 학습에서는 differentiable prefix KV를 통해 action loss가 영상·언어 adapter에 도달함을 확인했다. "
              "마지막 VLM layer의 K/V 외 Q/O/MLP adapter는 action loss에 직접 연결되지 않아 gradient가 없는 점을 기록했다. "
              "파라미터 수가 다른 실험이므로 동일 파라미터 예산의 ablation은 아니다.", "",
              "## Checkpoint 선택", "",
              "33개 safety/unsafety 쌍을 유지하면서 상황 그룹 번호의 짝/홀로 배치를 고른 66개 장면을 사용했다. "
              "운반 방향은 17쌍/16쌍이다. 선택 순서는 운반 성공 수 → 지속 성공 수 → 더 이른 step이며, "
              "나머지 validation 66개는 선택 후 확인에 사용했다. 초기 한 방향 subset은 학습 모델 평가 전에 수정했고 기록을 보존했다.", "",
              "| 조건 | Step | 운반 | 지속 운반 | XY-only |", "|---|---:|---:|---:|---:|"]
    for row in grid:
        lines.append(f"| {names[row['scope']]} | {row['step']:,} | {row['transport_success']}/{row['n']} | "
                     f"{row['sustained_transport_success']}/{row['n']} | {row['xy_only_transport_success']}/{row['n']} |")
    lines += ["", "![학습 곡선](../runs/semantic_motor_v6_lora_v1/learning_curves.png)", "",
              "곡선의 6,000-step 점선은 추가 학습 단계의 시작을 표시한다." if extension else "",
              "", "## 평가 기준과 검증", "",
              "모터 episode는 최대 360 control steps이며 10-step action chunk와 10회 flow denoising을 사용한다. "
              "원래 운반 기준은 현재 양쪽 finger grasp, 4cm 이상 lift, 연속/순 이동 각각 10cm 이상, "
              "목표 XY 반경 5cm 및 Z 오차 ±4cm이다. 4-step 지속 성공도 별도 보고한다. "
              "XY-only는 이 중 Z 조건만 제외한 진단 지표이며 목적지 위 배치나 위해 발생을 증명하지 않는다.", "",
              "원본 v6에는 이미지·지시가 같지만 그룹에 따라 목표 높이가 다른 네 입력 그룹이 있다. "
              "망치→못 박힌 나무는 약 10.7cm, 물 컵→대야는 약 8.9cm 차이가 난다. "
              "원본 기준은 유지했으므로 이 모호성을 수행 실패의 해석에 고려해야 한다.", "",
              "채택 시연의 독립 물리 replay, 원본 가중치 고정, gradient 범위, 공식 policy.infer와의 출력 일치, "
              "평가 입력/action/event 정렬, video frame 수, 모델 사이 초기 입력·지시·noise seed를 검증했다. "
              "132개 전체 trajectory에서 batch 8/32의 모든 action/state/정책 입력 이미지가 일치했다. "
              "빠른 모터 영상은 2Hz이며 모든 20Hz action/state/event를 별도 저장한다. "
              "마지막 작은 batch의 크기는 비교 시 유지한다.", "",
              "## 해석 범위와 산출물", "",
              "이 실험은 새로운 물체와 운반 동작의 수행 능력을 준비하는 단계이다. "
              "성공 trajectory가 늘었다고 VLM이 위험을 인식하거나 AE에 경고가 전달됐다고 결론 내릴 수 없다. "
              "성공한 쌍만 고른 분석에는 선택 편향과 물체/상황 차이가 남을 수 있다. "
              "t-SNE, LDA, linear probe 및 내부 warning 전달 분석은 아직 수행하지 않았다.", "",
              "AE-only의 VLM 함수는 고정 입력에 대해 유지되지만, 정책의 action이 달라지면 이후 관찰 이미지도 달라진다. "
              "따라서 서로 다른 rollout의 같은 timestep feature 차이를 모듈 변경 효과로 바로 해석해서는 안 된다. "
              "후속 분석에는 동일 관찰 입력 재생이나 행동 단계 정렬을 함께 고려해야 한다.", "",
              "- [세 adapter와 로딩 방법](../checkpoints/pi05_libero_v6_lora_v1/README.md)",
              "- [동일 장면 영상 비교](../runs/semantic_motor_v6_lora_v1/review.html)",
              "- [영상 비교 묶음 ZIP](../runs/semantic_motor_v6_lora_v1/review_bundle.zip)",
              "- [비교 CSV](../runs/semantic_motor_v6_lora_v1/comparison_summary.csv)",
              "- [66개 조건별 test 성능과 학습 시연 수](../runs/semantic_motor_v6_lora_v1/test_condition_summary.csv)",
              "- [성공한 safety/unsafety 쌍과 trajectory 경로](../runs/semantic_motor_v6_lora_v1/representation_candidate_pairs.json)",
              "- [환경과 실제 upstream source 기록](../runs/semantic_motor_v6_lora_v1/environment.json)",
              "- [세 학습 조건의 설정·가중치·gradient 검증](../runs/semantic_motor_v6_lora_v1/training_comparison_audit.json)",
              "- [이전 실패 진단](Semantic_Safety_v6_Failure_Diagnosis.md)", "",
              "원본 v6는 이미 진단/teacher 설계에 사용한 개발 benchmark이다. "
              "새 시연은 별도로 생성했지만 이 점 때문에 원본 v6 결과를 미관측 benchmark 성능으로 해석하지 않는다.", ""]
    candidates = json.loads((RUN / "representation_candidate_pairs.json").read_text())
    lines += ["## 후속 representation 분석용 후보", "",
              "안전·위험 양쪽 운반에 성공한 쌍을 모델별로 저장했다. 모델 사이 비교에서 공통으로 사용할 수 있는 쌍의 수는 다음과 같다. "
              "아래 수는 각 모델에서 두 episode 모두 운반에 성공한 경우의 교집합이며, 지속 성공은 후보 파일에서 추가로 구분할 수 있다.", "",
              "| 평가 집합 | 세 LoRA 모델 공통 | 기본 모델까지 네 모델 공통 |",
              "|---|---:|---:|"]
    for bench in ["confirmation", "test", "v6"]:
        shared = candidates["shared_pairs"][bench]
        lines.append(f"| {bench} | {len(shared['three_adapted_models'])} | "
                     f"{len(shared['baseline_and_three_models'])} |")
    lines += ["", "이 후보 목록에는 feature가 아직 추출되어 있지 않다. "
              "물체/상황 식별과 안전 의미의 구분을 분리하려면 후속 probe의 학습·평가도 상황 단위로 나눠야 한다.", ""]
    lines += ["## 재실행 예시", "",
              "저장소 루트에서 아래와 같이 새 출력 디렉터리에 학습한다. `--scope`는 `ae`, `vlm`, `all` 중 하나이다. "
              "현재 v2 manifest는 v1의 불변 시연·캐시 파일을 공유하므로 해당 원본 데이터도 필요하다.", "",
              "```bash",
              'PYTHONPATH="openpi/src:openpi/packages/openpi-client/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \\\n'
              'openpi/.venv/bin/python scripts/semantic_safety/train_v6_scoped_lora.py \\\n'
              f'  --manifest {DATA.relative_to(ROOT)}/combined_train_manifest.jsonl \\\n'
              '  --out runs/reproduce_v6_vlm --scope vlm --steps 6000 --batch-size 16 \\\n'
              '  --learning-rate 0.0002 --seed 20261111 --save-every 2000',
              "```", ""]
    if extension:
        lines += ["추가 단계는 같은 manifest와 scope로 새 출력 디렉터리를 지정하고 "
                  "`--resume <해당 scope의 adapter_step_06000.pt> --steps 2000 --learning-rate 0.00002 "
                  "--seed 20271111 --save-every 1000`을 사용한다. `--resume`은 adapter 가중치만 불러오며 optimizer는 새로 만든다.", ""]
    destination = ROOT / "doc/Semantic_Safety_v6_LoRA_Adaptation.md"
    notes = RUN / "preparation_and_progress_notes.md"
    if not notes.exists():
        shutil.copy2(destination, notes)
    document = "\n".join(lines)
    document = document.replace("../runs/semantic_motor_v6_lora_v1", "../" + str(RUN.relative_to(ROOT)))
    document = document.replace("../data/semantic_motor_v6_lora_v1", "../" + str(DATA.relative_to(ROOT)))
    document = document.replace("../checkpoints/pi05_libero_v6_lora_v1", "../" + str(EXPORT.relative_to(ROOT)))
    if dataset.get("change"):
        document += ("\n학습 데이터 보완: 스푼의 집기 높이를 calibration에서 8mm 높여 새 시연 48개를 확보했다. "
                     "이전 실행은 보존하고, 본 비교의 세 모델은 모두 보완한 동일 데이터로 공식 base부터 다시 학습했다. "
                     "물체 asset과 원본 평가 목표는 바꾸지 않았다.\n")
    destination.write_text(document)


def main():
    global RUN, DATA, EXPORT
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN)
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--export-root", type=Path, default=EXPORT)
    args = parser.parse_args()
    RUN = args.run_root.resolve()
    DATA = args.data_root.resolve()
    EXPORT = args.export_root.resolve()
    results = load_results()
    eligible = comparisons(results)
    condition_summary(results)
    curves()
    review(results)
    summary = []
    for bench in ["test", "v6", "confirmation", "native"]:
        for model in MODELS:
            summary.append(dict(benchmark=bench, model=model, **results[bench, model]["overall"],
                                paired_both_success=sum(row["model"] == model and row["benchmark"] == bench for row in eligible)))
    write_json(RUN / "comparison_summary.json", dict(rows=summary, complete=True))
    with (RUN / "comparison_summary.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    report(results, eligible)
    write_json(RUN / "report_complete.json", dict(complete=True, models=MODELS,
               benchmarks=["test", "v6", "confirmation", "native"]))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
