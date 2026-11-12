"""Summarize completed v5 adaptation without conflating proxy contact with semantic safety."""

from collections import Counter
import csv
import html
import json
from pathlib import Path

from rollout_common import ROOT, write_json
from encode_expansion_v5_review import encode_all

DATA=ROOT/"data/semantic_expansion_v5_lora_v1"
RUN=ROOT/"runs/semantic_expansion_v5_lora_v1"
EXPORT=ROOT/"checkpoints/pi05_libero_expansion_v5_lora_v1"


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def percentage(n,d):
    return f"{n}/{d} ({100*n/d:.1f}%)"


def main():
    audit=json.loads((DATA/"training_data_audit.json").read_text())
    native_audit=json.loads((DATA/"native_rehearsal_audit.json").read_text())
    protocol=json.loads((RUN/"experiment_protocol.json").read_text())
    selection=json.loads((RUN/"frozen_selection.json").read_text())["selected"]
    conditions={s["condition_index"]:s for s in load(DATA/"conditions.jsonl")}
    rows=[];by_skill=[];paired={};models={}
    for model in ["baseline","ae","vlm","all"]:
        motor=load(RUN/f"final/{model}/motor/episodes.jsonl")
        native=load(RUN/f"final/{model}/native/episodes.jsonl")
        models[model]={r["item_id"]:r for r in motor}
        pairs={}
        for item in motor:pairs.setdefault(item["pair_group"],[]).append(item)
        assert all(len(items)==2 for items in pairs.values())
        both=[key for key,items in pairs.items() if all(r["success"] for r in items)]
        paired[model]=[dict(pair_group=key,episodes=[str((RUN/f"final/{model}/motor/episodes"/r["item_id"]/"trajectory.h5").relative_to(ROOT))
                                                    for r in pairs[key]]) for key in both]
        row=dict(model=model,selected_step=0 if model=="baseline" else selection[model]["cumulative_step"],
            motor_successes=sum(r["success"] for r in motor),motor_episodes=len(motor),
            native_successes=sum(r["success"] for r in native),native_episodes=len(native),
            paired_both_successes=len(both),pair_trials=len(pairs))
        for side in ["safe","unsafe"]:
            subset=[r for r in motor if conditions[r["condition_index"]]["side"]==side]
            row[side+"_successes"]=sum(r["success"] for r in subset)
            row[side+"_episodes"]=len(subset)
        rows.append(row)
        for skill in sorted({r["skill"] for r in motor}):
            subset=[r for r in motor if r["skill"]==skill]
            by_skill.append(dict(model=model,skill=skill,successes=sum(r["success"] for r in subset),episodes=len(subset)))
        for suite in sorted({r["suite"] for r in native}):
            subset=[r for r in native if r["suite"]==suite]
            by_skill.append(dict(model=model,skill=suite,successes=sum(r["success"] for r in subset),episodes=len(subset)))
    for filename,records in [("comparison_summary.csv",rows),("skill_suite_summary.csv",by_skill)]:
        with (RUN/filename).open("w",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write_json(RUN/"representation_candidate_pairs.json",dict(feature_extraction_performed=False,
        selected_by_motor_proxy_success=True,selection_bias_must_be_accounted_for=True,models=paired))
    configs={s:json.loads((RUN/f"train_{s}/config.json").read_text()) for s in ["ae","vlm","all"]}
    freezes={s:json.loads((RUN/f"train_{s}/freeze_audit.json").read_text()) for s in configs}
    keys=["manifest_sha256","seed","steps_planned","batch_size","micro_batch_size","learning_rate_peak","code_sha256"]
    assert all(all(c[k]==configs["ae"][k] for c in configs.values()) for k in keys)
    assert all(r["all_original_weights_unchanged"] for r in freezes.values())
    assert freezes["ae"]["fixed_input_vlm_hidden_exact"]
    assert len({r["native_sample_fraction"] for r in freezes.values()})==1
    comparison=dict(passed=True,matched_fields=keys,identical_sample_seeds=True,all_original_weights_unchanged=True,
        ae_only_fixed_input_vlm_hidden_exact=True,native_fraction=freezes["ae"]["native_sample_fraction"],
        trainable_parameters={s:configs[s]["lora"]["trainable_parameters"] for s in configs},
        adapter_tensors_without_gradient={s:freezes[s]["last_gradient"]["no_gradient"] for s in configs},
        parameter_counts_not_matched=True,independent_training_seed_replicates=1)
    extension_info=json.loads((RUN/"extension_decision.json").read_text())
    extended=extension_info["extend_all_scopes"]
    comparison["extended_all_scopes"]=extended
    if extended:
        extension_configs={s:json.loads((RUN/f"train_{s}_extension/config.json").read_text()) for s in configs}
        extension_audits={s:json.loads((RUN/f"train_{s}_extension/freeze_audit.json").read_text()) for s in configs}
        assert all(all(c[k]==extension_configs["ae"][k] for c in extension_configs.values()) for k in keys)
        assert all(a["all_original_weights_unchanged"] for a in extension_audits.values())
        assert extension_audits["ae"]["fixed_input_vlm_hidden_exact"]
        assert len({a["native_sample_fraction"] for a in extension_audits.values()})==1
        comparison["extension_audits_passed"]=True
    write_json(RUN/"training_comparison_audit.json",comparison)
    labels={"baseline":"기본 π0.5 LIBERO","ae":"AE-only LoRA","vlm":"VLM-only LoRA","all":"공동 LoRA"}
    table="\n".join(f"| {labels[r['model']]} | {percentage(r['motor_successes'],r['motor_episodes'])} | "
        f"{percentage(r['native_successes'],r['native_episodes'])} | {r['paired_both_successes']}/{r['pair_trials']} | {r['selected_step']:,} |" for r in rows)
    missing=", ".join(audit["missing_conditions"]) or "없음"
    if extended:
        extension_text="사전 validation 조건에 따라 세 조건 모두 8,000 updates를 추가했다. 추가 단계는 LR 2e-5와 새 optimizer를 사용했다."
    elif not extension_info.get("extension_allowed",True):
        extension_text="사용자의 시간 제약에 따라 8,000-update 추가 학습은 보류했다. 이 변경은 최종 test 실행 전에 기록했으며, 각 모델의 16,000-step 이하 체크포인트에서 선택했다."
    else:
        extension_text="사전 validation 연장 조건을 충족하지 않아 추가 학습 없이 초기 단계의 체크포인트에서 선택했다."
    report=f"""# 확장 v5 π0.5 LoRA 적응

상태: **세 모델 학습·checkpoint 선택·최종 평가 완료**.

## 결과

| 모델 | 새 배치 모터 proxy | Native LIBERO | 양쪽 성공 대조쌍 | 선택 step |
|---|---:|---:|---:|---:|
{table}

모터 성공은 실제 집기와 3.5 cm lift, 연속으로 잡고 이동한 XY 경로 12 cm 이상 및 순변위 10 cm 이상을 먼저 확인한 뒤,
도구를 잡은 상태의 접촉·수신자 옆 전달·목적지 위 50도 이상 기울이기 중 해당 조건을 연속 충족한 것이다.
놓기·제공·덮기·채우기 지시는 검증된 운반 후 목적지와 접촉하면 놓은 상태도 인정한다. 도구 사용 지시는 마지막까지 잡은 상태를 요구한다.
전자레인지는 원본 LIBERO Long 환경에서 전면 접근 또는 접촉을 확인한다. 실제 절단·문 닫기·가열·액체 이동·위해 발생의 성공률이 아니다.

## 시연 생성과 검증

- v5의 **169개 상황, 338개 안전/위험 조건**을 유지했다. 물리 teacher가 물체 pose를 이용해 실제 `env.step` action을 생성했다.
- 신규 학습 시연 **{audit['episodes']:,}개 / {audit['control_steps']:,} control steps**, 시연 확보 조건 {audit['conditions_with_demonstrations']}/{audit['conditions_total']}. 미확보 조건: {missing}.
- 채택한 모든 시연을 별도 환경에서 action 단위로 재생하고 상태 및 성공을 검증했다. 기록 중 물체를 순간 이동시키지 않았다. 불안정한 물리 rollout은 채택하지 않았다.
- 위치 변화에 취약한 집기는 학습 배치에서 추가로 자세를 탐색했다({audit['extra_grasp_search_episodes']:,}개 시연). 물리 환경·controller·채점 기준은 유지했으며 이전 실패 기록도 보존했다. 평가 배치는 이 탐색에 사용하지 않았다.
- 수집 완료 후 모든 파일 hash, RGB·state·action의 시간 정렬, 유효한 action 범위, 관찰 shape·dtype, episode별 시작·중간·종료 카메라 관찰을 검사했다. `trajectory_quality_audit.jsonl`에 결과를 기록하고 실제 학습 배치의 표본 관찰을 `training_quality_review.html`에 제공한다. 전체 영상을 사람이 검수했다는 뜻은 아니다.
- 칼은 세운 상태로 절삭면을 아래로 수정했다. 이동 물체에 free joint를 추가하고 목적지는 고정했다. 정적 자산의 관성 보완, 보이는 집기 받침, 장갑 scan의 충돌 형상 분할, 작은 유리 조각의 12 g 질량 설정은 학습 환경 변환 기록에 남겼다.
- 모델에는 공식 π0.5 입력 경로의 RGB 두 시점과 원래 v5 지시를 사용했다. privileged pose, 안전 label, 근거 설명은 모델에 전달하지 않았다.

## LIBERO 재학습

기본 π0.5의 성공 rollout **{native_audit['episodes']}개**, {native_audit['tasks']}개 task를 독립 재생 검증해 사용했다.
Spatial/Object/Goal/Long의 초기 상태 2–5에서 수집했다. 이는 원본 사람 시연 파일을 재사용한 것이 아니라 공식 모델의 성공 시연이다.
전체 **학습 샘플의 10%**를 이 데이터에서 추출한다. 시연 개수의 10%라는 뜻은 아니다. 측정된 첫 학습 단계의 비율은 {100*comparison['native_fraction']:.2f}%이다.

## 학습과 선택

공식 LIBERO FT π0.5에서 세 조건을 각각 시작했다. 원본 가중치는 모두 고정하고 VLM, AE, 두 모듈의 LoRA만 각각 학습했다.
Rank 16, alpha 32, dropout 0, AdamW, 최대 LR 2e-4, 유효 batch 32이다. GPU 메모리를 위해 microbatch 16 두 번의 gradient를 누적한다.
동일 manifest, 예제·flow noise seed, optimizer update 수를 사용했다. 모듈별 학습 파라미터 수는 달라 파라미터 수를 맞춘 비교는 아니다.
복구용 optimizer·LoRA 상태를 200 updates마다 저장한다. 중단 시 저장된 단계부터 동일 예제와 noise로 재개하며 미저장 구간의 중복 로그는 별도로 보존한다.
VLM 마지막 층의 일부 LoRA는 AE에 전달되는 KV 이후의 연산에 있어 action loss의 gradient가 없다. 실제 gradient와 해당 tensor 목록을 감사 파일에 기록했다.

초기 16,000 updates의 4k/8k/12k/16k를 비교했다. {extension_text}
checkpoint는 native validation 성능 유지 조건을 우선 적용하고 모터 성공 수로 선택했다. 선택 결과를 고정한 후 test를 실행했다.
추가 학습 여부와 모든 후보 점수는 `extension_decision.json`, `checkpoint_candidates.json`, `frozen_selection.json`에 있다.

## 평가 집합과 해석 범위

- 모터 selection: {protocol['selection_episodes']}조건. 나머지 validation {protocol['confirmation_episodes']}조건은 선택 후 확인용이다.
- 모터 test: **{protocol['final_test_episodes']} episode = 169상황 × 안전/위험 × 새로운 배치 2개**. 동일 물체·상황의 새 배치이며 미관측 위험 종류의 평가가 아니다.
- Native validation: 4개 suite × 10 task × 초기 상태 7번 = 40 episode.
- Native test: 4개 suite × 10 task × 초기 상태 0–1번 = 80 episode. 모든 LIBERO 초기 상태를 평가한 공식 benchmark 점수로 일반화하지 않는다.
- train과 추가 학습의 평가 배치는 분리했다. 공식 checkpoint의 과거 학습에서도 미관측이었다는 뜻은 아니다.
- 평가 조건은 teacher나 모델의 성공 여부로 걸러내지 않았다. 독립 seed의 반복 학습 분산은 측정하지 않았다.

이 실험은 지시를 수행할 수 있는 모터 능력을 적응시키는 단계다. 안전/위험 양쪽 행동을 학습했으며 거부 능력이나 위험 인식 자체를 평가한 결과가 아니다.
representation 비교에는 원본 모델 대조, 고정 관찰 입력 재생, 상황 단위 probe split을 함께 고려해야 한다.
`representation_candidate_pairs.json`은 성공한 쌍의 경로 목록이며 feature 분석은 아직 수행하지 않았다.

## 산출물

- 학습 데이터: `data/semantic_expansion_v5_lora_v1/combined_train_manifest.jsonl`, `condition_coverage.csv`, `training_data_audit.json`.
- 실제 학습 시연 검토: `data/semantic_expansion_v5_lora_v1/training_quality_review.html`.
- adapter: `checkpoints/pi05_libero_expansion_v5_lora_v1/{{ae,vlm,all}}/adapter.pt`.
- 결과와 영상: `runs/semantic_expansion_v5_lora_v1/final/`, `comparison_summary.csv`, `skill_suite_summary.csv`, `review.html`.
- 학습 곡선: `learning_curves.png`. 검사: `training_comparison_audit.json`.

원본 v4와 기존 v6 실험·모델은 유지했다.
"""
    (ROOT/"doc/Semantic_Safety_Expansion_v5_LoRA_Adaptation.md").write_text(report)
    (RUN/"README.md").write_text(report)
    (DATA/"README.md").write_text("# 확장 v5 시연 데이터\n\n"+report.split("## 시연 생성과 검증\n",1)[1].split("## 학습과 선택",1)[0]+
        "\n전체 방법과 결과: `doc/Semantic_Safety_Expansion_v5_LoRA_Adaptation.md`.\n")
    (EXPORT/"README.md").write_text("# π0.5 LIBERO 확장 v5 LoRA\n\n원본 checkpoint에 adapter를 추가해 로딩한다. adapter만으로 추론할 수 없다.\n\n"
        "```python\nfrom motor_lora import load_policy\nfrom v6_scoped_lora import load\npolicy, config = load_policy('pi05_libero')\n"
        "metadata = load(policy._model, 'checkpoints/pi05_libero_expansion_v5_lora_v1/all/adapter.pt')\n```\n\n"
        "저장소 루트에서 `scripts/semantic_safety`, `openpi/src`, `openpi/packages/openpi-client/src`를 PYTHONPATH에 추가한다.\n")
    # Link original trajectories; do not embed semantic labels into policy images.
    encode_all(sorted((RUN/"final").glob("*/motor/episodes/*/views.mp4")))
    demonstration_review=DATA/"demonstration_review.html"
    if demonstration_review.exists():
        collection=Path(load(DATA/"train_manifest.jsonl")[0]["path"]).parts[2]
        encode_all(sorted((DATA/collection/"calibration").glob("*/views.mp4")))
        demonstration_review.write_text(demonstration_review.read_text().replace("/views.mp4","/views_browser.mp4"))
    cards=[]
    for item_id in models["baseline"]:
        row=models["baseline"][item_id]
        cells=[]
        for model in ["baseline","ae","vlm","all"]:
            result=models[model][item_id]
            path=f"final/{model}/motor/episodes/{item_id}"
            cells.append(f'<div><b>{labels[model]} · {"성공" if result["success"] else "실패"}</b>'
                f'<a href="{path}/views_browser.mp4"><img loading="lazy" src="{path}/final.png" alt="final observation"></a></div>')
        cards.append(f'<article><h3>{html.escape(item_id)}</h3><p>{html.escape(row["instruction"])}</p><section>{"".join(cells)}</section></article>')
    (RUN/"review.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>확장 v5 LoRA 비교</title>'
        '<style>body{font-family:sans-serif;margin:24px;background:#f5f5f5}article{background:white;padding:16px;margin:16px 0}'
        'section{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:12px}img{display:block;width:100%;max-width:224px}'
        '@media(max-width:700px){section{grid-template-columns:1fr 1fr}}</style><h1>확장 v5 LoRA 비교</h1>'
        '<p>모터 proxy 수행 비교. 이미지를 누르면 해당 실제 시뮬레이션 영상을 연다.</p>'+''.join(cards)+'</html>')
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    for scope in ["ae","vlm","all"]:
        records=load(RUN/f"train_{scope}/training.jsonl")
        if extended:
            records.extend({**r,"step":r["step"]+protocol["steps_initial"]}
                           for r in load(RUN/f"train_{scope}_extension/training.jsonl"))
        points={r["step"]:r for r in records}
        points=[points[k] for k in sorted(points)]
        for offset in range(0,len(points),100):
            part=points[offset:offset+100]
            if part:ax.plot(sum(p["step"] for p in part)/len(part),sum(p["active7_loss"] for p in part)/len(part),
                marker=".",color={"ae":"#377eb8","vlm":"#4daf4a","all":"#e41a1c"}[scope],
                label=scope if offset==0 else None)
    ax.set(xlabel="Optimizer update",ylabel="Flow loss (7 active action dimensions)")
    ax.legend();fig.tight_layout();fig.savefig(RUN/"learning_curves.png",dpi=180);plt.close(fig)
    print(json.dumps(dict(completed=True,models=rows,comparison_audit=comparison)),flush=True)


if __name__=="__main__":
    main()
