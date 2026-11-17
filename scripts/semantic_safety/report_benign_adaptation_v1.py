"""Audit matched adapted probes and write reproducible Korean comparative results."""

import csv
import json
from pathlib import Path

import h5py
import joblib
import numpy as np

import benign_adaptation_v1 as b
from representation_common import feature_sites
from rollout_common import write_json

MODELS=["base","libero","ae","vlm","all"]
NAMES={"base":"π0.5 base","libero":"LIBERO FT","ae":"AE-only","vlm":"VLM-only","all":"Both"}
PRIMARY=["vlm/layer_17","ae/flow_10/layer_17","velocity_pre/flow_10","velocity_post/flow_10"]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out=b.RUN/"report";out.mkdir(exist_ok=True)
    protocol=json.loads((b.RUN/"experiment_protocol.json").read_text())
    data=json.loads((b.DATA/"training_data_audit.json").read_text())
    selection=json.loads((b.RUN/"frozen_selection.json").read_text())
    assert data["passed"] and b.sha(b.BASELINE/"manifest.json")==data["probe_manifest_sha256"]
    assert (b.RUN/"manifest.json").read_bytes()==(b.BASELINE/"manifest.json").read_bytes()
    numerics=[];all_metrics={};rows=[]
    with h5py.File(b.BASELINE/"features/libero.h5") as baseline:
        for model in MODELS:
            with h5py.File(b.RUN/"features"/(model+".h5")) as features:
                assert features.attrs["complete"] and features["written"][:].all()
                assert np.array_equal(features["input_sha256"][:],baseline["input_sha256"][:])
                for site,_ in feature_sites():
                    assert np.isfinite(features[site][:]).all()
                    if model=="ae" and site.startswith("vlm/"):assert np.array_equal(features[site][:],baseline[site][:]),site
            for track in ["A","B"]:
                for site,_ in feature_sites():
                    folder=b.RUN/"probes"/track/model/site
                    result=json.loads((folder/"metrics.json").read_text())
                    assert result["complete"]
                    all_metrics[model,track,site]=result
                    for fold in range(5):
                        fit=joblib.load(folder/f"fold_{fold}.joblib")
                        for array in [fit["scaler"].mean_,fit["scaler"].scale_,fit["classifier"].coef_,fit["classifier"].intercept_]:
                            assert np.isfinite(array).all(),(model,track,site,fold)
                        numerics.append(dict(model=model,track=track,site=site,fold=fold,passed=True))
                    rows.append(dict(model=model,track=track,site=site,AUROC=result["fold_mean_AUROC"],
                        AUROC_SD=result["fold_sd_AUROC"],balanced_accuracy=result["fold_mean_balanced_accuracy"],
                        accuracy=result["oof"]["accuracy"],predicted_unsafe_fraction=result["oof"]["predicted_unsafe_fraction"],
                        confusion_matrix=result["oof"]["confusion_matrix"]))
    write_json(out/"numerical_audit.json",dict(passed=True,classifiers=len(numerics),
        all_parameters_and_features_finite=True,original_inputs_and_folds_unchanged=True,
        ae_only_vlm_features_exactly_unchanged=True))
    from contrast_initial_linear import metrics
    manifest=json.loads((b.RUN/"manifest.json").read_text())
    selected=[]
    for track in ["A","B"]:
        track_rows=[r for r in manifest["rows"] if r["track"]==track]
        for model in MODELS:
            for module in ["vlm","ae"]:
                choices=sorted([v for (m,t,s),v in all_metrics.items() if m==model and t==track and s.startswith(module+"/")],key=lambda v:v["site"])
                scores=np.full(len(track_rows),np.nan);folds=[]
                for fold in range(5):
                    winner=max(choices,key=lambda v:v["folds"][fold]["validation_AUROC"])
                    with np.load(b.RUN/"probes"/track/model/winner["site"]/"oof_predictions.npz") as values:
                        mask=values["fold"]==fold;scores[mask]=values["decision_score"][mask]
                    folds.append(dict(fold=fold,site=winner["site"],C=winner["folds"][fold]["C"],
                                      validation_AUROC=winner["folds"][fold]["validation_AUROC"],test=winner["folds"][fold]["test"]))
                assert np.isfinite(scores).all()
                selected.append(dict(track=track,model=model,module=module,folds=folds,
                    fold_mean_AUROC=float(np.mean([v["test"]["AUROC"] for v in folds])),
                    fold_mean_balanced_accuracy=float(np.mean([v["test"]["balanced_accuracy"] for v in folds])),
                    oof=metrics(track_rows,scores)))
    write_json(out/"validation_selected_sites.json",selected)
    with (out/"all_sites.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(2,2,figsize=(13,9),sharex=True,sharey=True)
    for r,track in enumerate(["A","B"]):
        for c,prefix in enumerate(["vlm/layer_","ae/flow_10/layer_"]):
            ax=axes[r,c]
            for model in MODELS:
                values=[all_metrics[model,track,prefix+f"{layer:02d}"]["fold_mean_AUROC"] for layer in range(18)]
                ax.plot(range(18),values,label=NAMES[model],linewidth=1.7)
            ax.axhline(.5,ls="--",c="gray",lw=1);ax.set_ylim(.35,1.01);ax.grid(alpha=.2)
            ax.set_title(f"Track {track}: "+("VLM" if c==0 else "AE / flow 10"));ax.set_xlabel("Layer")
            ax.set_ylabel("Mean held-out fold AUROC");ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/"layer_auroc.png",dpi=170);plt.close(fig)
    assets=b.ROOT/"doc/assets/benign_adaptation_v1";assets.mkdir(parents=True,exist_ok=True)
    import shutil
    shutil.copy2(out/"layer_auroc.png",assets/"layer_auroc.png")
    lines=["# 안전한 물체·조작 적응 전후 Linear Probe 비교", "",
           "## 1. 연구 질문", "",
           "새 물체와 조작을 안전한 과제로 익힌 뒤, 첫 장면의 안전·위험 정보를 VLM과 Action Expert에서 얼마나 선형적으로 구분할 수 있는지 비교했다. 이전 안전·위험 대조 과제를 행동 시연으로 그대로 재사용하지 않았다.", "",
           "FT가 기존 representation을 혼란시켰다는 설명은 검증 대상 가설이다. 이번 결과만으로 과거 실험 성능 저하의 원인을 확정하지 않는다. 첫 장면 probe 자체에는 실제 조작 성공이 필수 조건이 아니다.", "",
           "## 2. 정책 적응 데이터", "",
           f"- 별도 과제: {data['conditions']}개, 성공 시연 {data['motor_episodes']:,}개, 제어 action {data['motor_control_steps']:,}개.",
           f"- LIBERO 재학습 시연: {data['native_episodes']}개, 네 suite의 {data['native_tasks']}개 task. 원본 사람 시연이 아니라 공식 LIBERO 정책의 성공 trajectory를 검증한 데이터다.",
           "- minibatch 샘플의 90%는 새 benign 시연, 10%는 LIBERO 시연이다. 에피소드 수를 90:10으로 맞춘다는 의미가 아니다.",
           "- 새 시연 안에서는 이동 50% / 나머지 동작 50%로 샘플링하고, 후자는 verb를 균등하게 뽑은 뒤 과제·episode·시점을 뽑는다. 이동 과제 수가 많아 다른 조작이 묻히는 것을 줄인다.",
           "- 물체 어휘·시각 asset의 공유는 의도했다. 새 학습 대상 fixture와 배우는 목표 조합은 기존 probe 조건과 분리했다. 이는 미관측 물체에 대한 평가가 아니다.",
           "- 곰팡이 식품, 사람·영아 위해, 생물 대상 오염, 가열·통전 손상 과제는 시연에 넣지 않았다. 식품은 신선한 상태, 다리미·금속판·주전자는 차갑거나 전원이 꺼진 변형을 사용했다.",
           "- privileged teacher는 시뮬레이터 물체 pose를 읽는다. 학생의 입력에는 RGB 두 장, robot state, 작업 지시만 전달하며 물체 pose와 안전 라벨·이유는 전달하지 않는다.",
           "- 매 시연을 별도 환경에서 action replay하고, 실제 잡기·들기·이동·접촉, 유한한 상태, 관측/action 시간 정렬과 train/holdout 초기 배치 비중복을 확인했다.", "",
           "- 천 공·천 책·놀이 블록은 각각 원래 크기의 0.70/0.60/0.75배인 학습용 변형이다. 원본 probe asset은 보존했다. 종이로 프레임을 덮는 후보 1개만 물리 시연 안정성 때문에 제외했고, 모든 원래 probe 조건은 평가에 유지했다.", "",
           "[시연 초기·종료 이미지 검토](../data/semantic_benign_adaptation_v1/preview.html) · [중복 검사](../data/semantic_benign_adaptation_v1/semantic_overlap_audit.json) · [데이터 검증](../data/semantic_benign_adaptation_v1/training_data_audit.json)", "",
           "## 3. 조작 성공 기준과 한계", "",
           "- place: 집어 올려 운반한 후 목적지에 내려놓고 gripper를 연다.",
           "- wipe/cut/sand/brush/saw/grate/scratch: 잡고 운반한 도구가 대상과 접촉하면서 수평으로 3 cm 이상 움직인다.",
           "- cover: 빈 연습 프레임의 작은 개구부를 덮도록 겹쳐 놓고 놓아준다.",
           "- press: 잡고 운반한 도구가 연습 표면에 일정 시간 접촉한다.",
           "- pour: 잡아 운반한 물병·주전자가 계량 용기 위에서 50도 이상 기울어진다. 실제 액체 이동은 구현하지 않는다.",
           "시뮬레이션의 강체 asset은 실제 절단·표면 제거·세척·천 변형·유체 이동을 구현하지 않는다. 성공률은 위 motor 기준의 충족률이다. knife는 정적 검토 asset의 모양을 유지하고 회전각 단위·바닥 정렬·질량·접촉 강성을 수정해 동적 조작을 안정화했다.", "",
           "평가 중 pour의 actor 좌표가 stroke용 단일 좌표 변수로 덮여쓰이는 구현 오류를 수정했다. 평가 전용 oracle에서 해당 변수 이름만 분리했으며, 기존 생성 코드·시연·학습 가중치와 성공 판정식·임계값은 유지했다. 원래 오류 재현, 잘못된 성공 판정 방지, 수정 범위를 확인하는 회귀 검사 5개를 통과했다. [수정 검증 기록](../runs/semantic_benign_adaptation_v1/pour_coordinate_recovery_audit.json)", "",
           "## 4. LoRA 학습 recipe", "",
           "| 항목 | 설정 |", "|---|---|",
           "| 출발 체크포인트 | 공식 LIBERO FT π0.5 |",
           "| 학습 범위 | VLM-only / AE-only / Both |",
           "| VLM 범위 | PaliGemma vision·language·multimodal linear 모듈; 생성 head 제외 |",
           "| AE 범위 | expert Transformer와 action·time 입력/출력 projection |",
           "| LoRA | rank 16, alpha 32, dropout 0; 원래 가중치는 freeze |",
           "| 손실 | action flow-matching MSE; 안전·위험 분류 손실 없음 |",
           "| Optimizer | AdamW, weight decay 0.01, gradient clip 1 |",
           "| Learning rate | peak 2e-4, warmup 300 updates, cosine로 peak의 10%까지 감소 |",
           "| Batch | 유효 32 = microbatch 16 × gradient accumulation 2 |",
           "| 학습 길이 | 각 16,000 optimizer updates; 추가 8,000 updates 없음 |",
           "| Action chunk | 10 control timesteps, 내부 32차원 / 실제 로봇 action 7차원 |",
           "| 비교 통제 | 세 방식에 같은 샘플링 seed와 minibatch 순서, 공유 모듈의 같은 LoRA 초기화 |", "",
           "4k/8k/12k/16k 체크포인트를 benign motor validation 및 LIBERO validation으로 선택했다. 안전·위험 probe test AUROC는 선택에 사용하지 않았다. LIBERO validation은 init state 7, 최종 평가는 0/1이며 rehearsal은 2–5다.",
           "새 환경은 기존 첫 장면 probe와 동일한 upright RGB를 사용한다. LIBERO rehearsal/유지 평가는 기존 공식 평가 파이프라인의 180도 convention을 유지한다. 이를 데이터별로 명시적으로 처리한다.", "",
           "| 모델 | 선택 step | motor validation | LIBERO validation | 유지 조건 충족 |",
           "|---|---:|---:|---:|---|" ]
    for scope,row in selection["selected"].items():
        lines.append(f"| {NAMES[scope]} | {row['cumulative_step']:,} | {row['motor_successes']}/{row['motor_episodes']} | {row['native_successes']}/{row['native_episodes']} | {row['retention_constraint_met']} |")
    lines += ["", "## 5. 조작 및 LIBERO 유지 성능", "",
              "새 조작 test는 같은 benign 과제의 별도 초기 배치다. semantic safety benchmark에서 위험 행동을 실행한 점수가 아니다. LIBERO 80개는 네 suite × 10 task × 2 초기 상태이며 전체 공식 benchmark 점수와 구분한다.", "",
              "| 모델 | benign motor test | LIBERO 80 test |", "|---|---:|---:|"]
    motor_results={}
    for scope in ["baseline","ae","vlm","all"]:
        values={kind:json.loads((b.RUN/"final"/scope/kind/"completion.json").read_text()) for kind in ["motor","native"]}
        motor_results[scope]=values
        lines.append(f"| {NAMES.get(scope,'LIBERO FT baseline')} | "+" | ".join(f"{v['successes']}/{v['episodes']} ({v['successes']/v['episodes']:.1%})" for v in values.values())+" |")
    lines += ["", "## 6. Linear Probe 학습·평가", "",
              "기존 336개 의미 조건, A 336개/B 121개 입력, 첫 장면 이미지·지시·state, 연결 group 20개와 5-fold 분할을 그대로 사용했다. Base와 LIBERO baseline은 완료된 동일 실험의 feature/probe를 읽기 전용으로 재사용했다.",
              "각 적응 모델의 가중치를 freeze하고 78개 위치를 추출했다. VLM 0–17 residual, AE 0–17 residual × flow 1/5/10, 각 flow의 velocity projection 전후다. 같은 초기 noise와 10회 내부 denoising을 사용하며 환경 action은 실행하지 않는다. Token 축에만 mean pooling한다.",
              "각 모델·위치·fold마다 StandardScaler와 L2 logistic probe를 독립적으로 새로 학습했다. train group 크기와 클래스 질량을 보정하는 sample weight를 적용했다. C=0.01/0.1/1 중 validation AUROC로 선택하고 dev를 합쳐 재학습하지 않는다. 판정은 logit≥0이다.",
              "주 AUROC와 balanced accuracy는 5개 test fold의 평균이다. Accuracy·confusion matrix·위험 예측 비율은 모든 OOF test 예측을 합친 값이다. 기존 보고서에 fold별 실제 표본 수와 weighting 수식이 있다.",
              "[기존 상세 프로토콜](Semantic_Safety_Contrast_Initial_Linear_v1_Results.md)", "",
              "## 7. 주요 위치 결과", "",
              "표의 AUROC는 fold 평균±표준편차이며 신뢰구간이 아니다. 마지막 layer 결과를 사전에 지정한 주 비교로 사용한다.", ""]
    for track in ["A","B"]:
        lines += [f"### Track {track}","","| 모델 | VLM layer17 AUROC | VLM accuracy | AE flow10 layer17 AUROC | AE accuracy |","|---|---:|---:|---:|---:|"]
        for model in MODELS:
            a=all_metrics[model,track,PRIMARY[0]];e=all_metrics[model,track,PRIMARY[1]]
            lines.append(f"| {NAMES[model]} | {a['fold_mean_AUROC']:.3f}±{a['fold_sd_AUROC']:.3f} | {a['oof']['accuracy']:.1%} | {e['fold_mean_AUROC']:.3f}±{e['fold_sd_AUROC']:.3f} | {e['oof']['accuracy']:.1%} |")
        lines += ["", "| 모델·위치 | balanced accuracy | 위험 예측 비율 | [[TN,FP],[FN,TP]] |", "|---|---:|---:|---|"]
        for model in MODELS:
            for site,label in [(PRIMARY[0],"VLM17"),(PRIMARY[1],"AE17/flow10")]:
                value=all_metrics[model,track,site]
                lines.append(f"| {NAMES[model]} {label} | {value['fold_mean_balanced_accuracy']:.1%} | {value['oof']['predicted_unsafe_fraction']:.1%} | {value['oof']['confusion_matrix']} |")
        lines.append("")
    lines += ["### Velocity projection 전후", "",
              "아래는 flow 10의 AUROC다. 전/후 차원은 각각 1024/32이며, 수치 차이만으로 위험 경고가 사라졌다고 판단하지 않는다.", "",
              "| 모델 | A pre | A post | B pre | B post |","|---|---:|---:|---:|---:|"]
    for model in MODELS:
        values=[all_metrics[model,track,site]["fold_mean_AUROC"] for track in ["A","B"] for site in PRIMARY[2:]]
        lines.append(f"| {NAMES[model]} | "+" | ".join(f"{v:.3f}" for v in values)+" |")
    lines += ["", "### Validation으로 위치를 선택한 보조 비교", "",
              "각 fold에서 VLM 18개/AE 54개 후보 중 validation AUROC로 위치를 고른 뒤 해당 test fold 점수를 집계했다. Test에서 가장 높은 위치를 고른 값이 아니며, fold마다 선택 위치가 다를 수 있다. 추가 probe 학습은 없다.", "",
              "| 모델 | A VLM | A AE | B VLM | B AE |","|---|---:|---:|---:|---:|"]
    for model in MODELS:
        values=[next(v for v in selected if v["model"]==model and v["track"]==track and v["module"]==module)["fold_mean_AUROC"] for track in ["A","B"] for module in ["vlm","ae"]]
        lines.append(f"| {NAMES[model]} | "+" | ".join(f"{v:.3f}" for v in values)+" |")
    lines += ["", "![Layer별 AUROC](assets/benign_adaptation_v1/layer_auroc.png)","",
              "## 8. 해석", "",
              "AE-only에서 VLM 18개 layer의 feature는 LIBERO baseline과 입력별로 완전히 같음을 확인했다. 따라서 이 비교의 VLM probe 변화는 없어야 한다. AE 측 결과 변화와 분리해 해석한다.", ""]
    for track in ["A","B"]:
        for module,site in [("VLM",PRIMARY[0]),("AE",PRIMARY[1])]:
            baseline=all_metrics["libero",track,site]["fold_mean_AUROC"]
            deltas=[f"{NAMES[m]} {all_metrics[m,track,site]['fold_mean_AUROC']-baseline:+.3f}" for m in ["ae","vlm","all"]]
            lines.append(f"- Track {track} {module}의 LIBERO baseline 대비 AUROC 차이: "+", ".join(deltas)+".")
    lines += ["", "조작 성공률 상승과 함께 probe 성능이 유지되면, 새 물체·조작 적응과 선형적으로 읽히는 위험 정보가 공존한다는 근거가 된다. probe 성능이 감소하면 benign 적응에서도 위험 라벨과 관련된 선형 구분 가능성이 약해졌다는 관찰이다. 정보 자체가 사라졌다거나 과거 실패 원인을 입증했다는 결론으로 확대하지 않는다.",
              "이 실험은 위험 정보의 선형 접근 가능성을 측정한다. 위험이 VLM에서 AE로 인과적으로 전달되는지, 행동 선택에 사용되는지는 검증하지 않는다. A/B는 서로 다른 크기의 부분집합이며 이미지 기여의 인과 비교가 아니다. 라벨은 아직 독립적인 사람 검증을 마친 gold label이 아니고, 별도 위험 유형 holdout도 아니다.", "",
              "## 9. 재현 및 근거 파일", "",
              "```bash", "openpi/.venv/bin/python scripts/semantic_safety/run_benign_adaptation_v1.py --supervise", "```", "",
              "완료된 단계는 검증 기록을 읽어 재사용하며 중단된 학습은 optimizer/LoRA 복구 상태에서 이어간다. 같은 run을 중복 실행하면 lock으로 차단한다.", "",
              "- [실험 설정](../runs/semantic_benign_adaptation_v1/experiment_protocol.json)",
              "- [체크포인트 선택](../runs/semantic_benign_adaptation_v1/frozen_selection.json)",
              "- [전체 위치 CSV](../runs/semantic_benign_adaptation_v1/report/all_sites.csv)",
              "- [수치·입력 동일성 검증](../runs/semantic_benign_adaptation_v1/report/numerical_audit.json)",
              "- 모델 adapter: `checkpoints/pi05_libero_benign_adaptation_v1/{ae,vlm,all}/adapter.pt`", ""]
    path=b.ROOT/"doc/Semantic_Safety_Benign_Adaptation_v1_Results.md"
    path.write_text("\n".join(lines))
    write_json(out/"summary.json",dict(complete=True,rows=rows,policy_evaluation=motor_results,
                                     validation_selected_sites=selected,document=str(path)))
    write_json(b.RUN/"verification.json",dict(passed=True,numerical_audit=str(out/"numerical_audit.json"),
                                             report=str(path),all_input_hashes_match=True))
    print(json.dumps(dict(complete=True,report=str(path))),flush=True)


if __name__=="__main__":main()
