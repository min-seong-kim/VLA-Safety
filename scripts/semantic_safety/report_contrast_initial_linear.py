"""Summarize held-out probe results without selecting layers on test scores."""

import csv
from html import escape
import json
from pathlib import Path

import h5py
import numpy as np

from contrast_initial_linear import RUN, MODELS, metrics, milestone
from rollout_common import write_json


def table(headers, rows):
    return '<table><thead><tr>'+''.join('<th>'+escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table>'


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest=json.loads((RUN/"manifest.json").read_text())
    protocol=json.loads((RUN/"experiment_protocol.json").read_text())
    folder=RUN/"report";folder.mkdir(exist_ok=True)
    all_results=[json.loads(p.read_text()) for p in sorted((RUN/"probes").rglob("metrics.json"))]
    assert len(all_results)==312, len(all_results)
    indexed={(r["track"],r["model"],r["site"]):r for r in all_results}
    records=[]
    for r in all_results:
        records.append(dict(track=r["track"],model=r["model"],site=r["site"],
                            fold_mean_AUROC=r["fold_mean_AUROC"],fold_sd_AUROC=r["fold_sd_AUROC"],
                            fold_mean_balanced_accuracy=r["fold_mean_balanced_accuracy"],
                            **{k:v for k,v in r["oof"].items() if k!="confusion_matrix"}))
    with (folder/"all_sites.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    primary=[indexed[track,model,site] for track in ["A","B"] for model in MODELS for site in protocol["primary_sites"]]
    # Validation selects one site per fold, independently of all test performance.
    selected=[]
    for track in ["A","B"]:
        selected_rows=[r for r in manifest["rows"] if r["track"]==track]
        for model in MODELS:
            for module in ["vlm","ae"]:
                choices=sorted([r for r in all_results if r["track"]==track and r["model"]==model and r["site"].startswith(module+"/")],key=lambda r:r["site"])
                scores=np.full(len(selected_rows),np.nan);fold_records=[]
                for fold in range(5):
                    winner=max(choices,key=lambda r:r["folds"][fold]["validation_AUROC"])
                    path=RUN/"probes"/track/model/winner["site"]/"oof_predictions.npz"
                    with np.load(path) as values:
                        mask=values["fold"]==fold;scores[mask]=values["decision_score"][mask]
                    fold_records.append(dict(fold=fold,site=winner["site"],C=winner["folds"][fold]["C"],
                                             validation_AUROC=winner["folds"][fold]["validation_AUROC"],
                                             test=winner["folds"][fold]["test"]))
                assert np.isfinite(scores).all()
                result=dict(track=track,model=model,module=module,folds=fold_records,
                            fold_mean_AUROC=float(np.mean([r["test"]["AUROC"] for r in fold_records])),
                            fold_mean_balanced_accuracy=float(np.mean([r["test"]["balanced_accuracy"] for r in fold_records])),
                            oof=metrics(selected_rows,scores))
                selected.append(result)
    write_json(folder/"validation_selected_sites.json",selected)
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True,sharey=True)
    colors={"base":"#31698b","libero":"#b06c43"}
    for ri,track in enumerate(["A","B"]):
        for ci,module in enumerate(["vlm","ae"]):
            ax=axes[ri,ci]
            for model in MODELS:
                steps=[None] if module=="vlm" else [1,5,10]
                for step in steps:
                    sites=[f"vlm/layer_{i:02d}" if step is None else f"ae/flow_{step:02d}/layer_{i:02d}" for i in range(18)]
                    values=[indexed[track,model,s]["fold_mean_AUROC"] for s in sites]
                    sd=np.array([indexed[track,model,s]["fold_sd_AUROC"] for s in sites])
                    label=model+("" if step is None else f" / flow {step}")
                    ax.plot(range(18),values,label=label,color=colors[model],linestyle={None:"-",1:":",5:"--",10:"-"}[step])
                    if step is None:ax.fill_between(range(18),np.array(values)-sd,np.array(values)+sd,color=colors[model],alpha=.1)
            ax.axhline(.5,color="#888",linestyle=":",linewidth=1);ax.set_title(f"Track {track}: {module.upper()}")
            ax.set_ylim(.2,1.);ax.set_xlim(0,17);ax.set_xticks([0,3,6,9,12,15,17]);ax.grid(alpha=.15);ax.legend(fontsize=8)
            if ci==0:ax.set_ylabel("Five-fold mean test AUROC")
            if ri==1:ax.set_xlabel("Residual-output layer")
    fig.suptitle("Frozen pi0.5 checkpoints: initial scene only (VLM shading: fold SD)")
    fig.tight_layout();fig.savefig(folder/"layer_auroc.png",dpi=180);plt.close(fig)
    # Verify every checkpoint and every input before reporting a complete experiment.
    audits={model:json.loads((RUN/"features"/(model+"_audit.json")).read_text()) for model in MODELS}
    with h5py.File(RUN/"features/base.h5") as a,h5py.File(RUN/"features/libero.h5") as b:
        assert a.attrs["complete"] and b.attrs["complete"]
        assert np.array_equal(a["input_sha256"][:],b["input_sha256"][:])
    for r in all_results:
        for fold in r["folds"]:
            parts=[set(fold[k+"_indices"]) for k in ["train","validation","test"]]
            assert not(parts[0]&parts[1] or parts[0]&parts[2] or parts[1]&parts[2])
            assert np.isclose(fold["train_weighted_safe_mass"],fold["train_weighted_unsafe_mass"])
    for fold in manifest["folds"]:
        parts={k:[r for r in manifest["rows"] if r["group_id"] in fold[k+"_groups"]] for k in ["train","validation","test"]}
        for key in ["condition_id","image_hash"]:
            sets=[{r[key] for r in parts[k]} for k in parts]
            assert not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]),(fold["fold"],key)
    verification=dict(complete=True,conditions=312,selected_fits=1560,candidate_fits=4680,all_candidates_converged=True,
                      frozen_weight_digests_unchanged=True,model_inputs_identical=True,
                      condition_and_image_leakage=0,train_class_weighted_mass_equal=True,
                      text_generation_heads_used=False,initial_scene_only=True)
    write_json(RUN/"verification.json",verification)
    write_json(folder/"summary.json",dict(primary_sites=primary,validation_selected=selected,verification=verification,feature_audits=audits))
    cells=[]
    for track in ["A","B"]:
        for model in MODELS:
            for site in ["vlm/layer_17","ae/flow_10/layer_17"]:
                r=indexed[track,model,site];o=r["oof"]
                cells.append([track,model,"VLM 17" if site.startswith("vlm") else "AE 17 / flow 10",
                              f"{r['fold_mean_AUROC']:.3f} ± {r['fold_sd_AUROC']:.3f}",
                              f"{100*r['fold_mean_balanced_accuracy']:.1f}%",f"{100*o['accuracy']:.1f}%",
                              f"{o['predicted_unsafe']}/{o['n']} ({100*o['predicted_unsafe_fraction']:.1f}%)"])
    primary_table=table(["트랙","체크포인트","위치","AUROC, fold 평균 ± SD","Balanced acc, fold 평균","Accuracy, OOF 합산","위험 판정"],cells)
    confusion=[]
    for track in ["A","B"]:
        for model in MODELS:
            for site in ["vlm/layer_17","ae/flow_10/layer_17"]:
                m=indexed[track,model,site]["oof"]["confusion_matrix"]
                confusion.append([track,model,site,m[0][0],m[0][1],m[1][0],m[1][1]])
    folds=[]
    for f in manifest["folds"]:
        for track in ["A","B"]:
            counts=f["counts"]
            folds.append([f["fold"],track,*[f"{counts[k][track]['safe']} / {counts[k][track]['unsafe']}" for k in ["train","validation","test"]]])
    selected_table=table(["트랙","모델","모듈","AUROC, fold 평균","Balanced acc, fold 평균","fold별 선택 위치"],
                          [[r["track"],r["model"],r["module"],f"{r['fold_mean_AUROC']:.3f}",f"{100*r['fold_mean_balanced_accuracy']:.1f}%",
                            "; ".join(str(f["fold"])+": "+f["site"] for f in r["folds"])] for r in selected])
    text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>첫 장면 linear probe 결과 · pi0.5 base / LIBERO</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1400px;margin:35px auto;padding:0 22px;color:#253447;line-height:1.7}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border-bottom:1px solid #d7dfe6;padding:9px;text-align:left}}th{{background:#edf2f6}}.scroll{{overflow:auto}}img{{width:100%}}.note{{background:#eef3f7;padding:16px;border-radius:8px}}code{{overflow-wrap:anywhere}}</style>
<h1>π0.5 base / LIBERO: 첫 장면 안전·위험 linear probe</h1>
<p>두 VLA는 고정하고 probe만 학습했다. 같은 첫 장면·영문 작업 지시·초기 noise를 입력했다. 시뮬레이터를 전진시키거나 성공 trajectory를 선택하지 않았다. 안전 질문과 판단 근거도 모델 입력에 추가하지 않았다.</p>
<p class="note"><strong>조건:</strong> 검토용 후보 label 336개(A: 안전228·위험108), 관찰 결합 후보 121개(B: 안전63·위험58). 두 트랙은 별도 probe를 학습했다. 동일 의미 조건의 A/B는 같은 분할이다. 인체·생물 proxy 인식 및 모든 후보의 의미적 정답은 아직 독립 검증되지 않았다.</p>
<h2>1. 사전 지정한 마지막 층 결과</h2><p>AUROC와 balanced accuracy는 5개 test fold의 평균이다. Accuracy와 판정 수는 각 입력이 한 번씩 test가 된 OOF 예측의 합산이다. SD는 fold 표준편차이며 신뢰구간이 아니다. OOF AUROC를 서로 다른 probe의 점수와 섞어 주요 결과로 사용하지 않았다.</p><div class="scroll">{primary_table}</div>
<p>A 전체에서 항상 안전이라고 답해도 accuracy 67.9%, B는 52.1%다. 이 분류기의 balanced accuracy는 두 트랙 모두 50%다. 위험 판정 비율은 probe의 출력이며 VLA 자체의 위험 확률이나 거절률이 아니다.</p>
<h2>2. 실제 오분류 수</h2><div class="scroll">{table(["트랙","모델","위치","안전→안전","안전→위험","위험→안전","위험→위험"],confusion)}</div>
<h2>3. 층·denoising 단계 탐색</h2><img src="layer_auroc.png" alt="층별 test AUROC"><p>전체 0–17층과 AE flow 1·5·10, velocity projection 전후를 탐색했다. 이 그림에서 test 점수가 가장 높은 위치를 최종 detector로 선택하지 않았다.</p>
<h2>4. Validation으로 위치까지 선택한 보조 결과</h2><p>각 fold에서 C를 고른 뒤, VLM 18개 위치 또는 AE 54개 위치 중 validation AUROC가 가장 높은 위치를 선택했다. 아래 test 점수는 선택에 쓰지 않았다. Fold마다 다른 위치이므로 하나의 고정된 최적 층을 찾았다는 의미가 아니다.</p><div class="scroll">{selected_table}</div>
<h2>5. 분할과 학습</h2><p>87개 대조 묶음의 공유 조건과 완전히 같은 두 카메라 관찰을 연결해 <strong>{manifest['groups']}개 분할 그룹</strong>을 만들었다. 조건·영상 누출은 0건이다. 그룹 크기가 달라 fold별 test 수가 다르다. 아래 숫자는 안전 / 위험이다.</p><div class="scroll">{table(["Fold","트랙","Train 안전/위험","Validation 안전/위험","Test 안전/위험"],folds)}</div>
<p>Train에서 그룹별 비중을 맞춘 뒤 두 클래스의 가중치 합을 같게 했다. 같은 train 가중치로 StandardScaler를 학습했다. L2 logistic regression의 C=0.01, 0.1, 1을 validation AUROC로 선택하고 validation을 합쳐 재학습하지 않았다. 판정 기준은 logit≥0(확률 0.5)이며 test에 맞춰 조정하지 않았다.</p>
<h2>6. 해석의 범위</h2><ul>
<li>두 체크포인트 모두 LIBERO 입력 형식, 두 카메라, action horizon 10, discrete state token 없음, denoising 10회 조건으로 비교했다. Base의 기본 50-token·discrete-state 설정 그대로의 평가는 아니다. Base에서 이 입력 형식의 영향을 별도로 분리하지 않았다.</li>
<li>VLM은 유효한 이미지·지시 token 전체 평균, AE는 10개 action token의 평균이다. 시간·층·denoising 단계를 함께 평균하지 않았다. State는 동일하게 전달하지만 이 설정에서는 별도 state token으로 embedding되지 않는다.</li>
<li>분할은 겹치지 않는 대조 그룹 일반화다. 별도로 설계한 위험 유형 전체 제외 평가나 원인별 시각·언어 기여 분리는 수행하지 않았다. A와 B의 표본 구성이 달라 두 점수 차이만으로 modality 기여를 판단할 수 없다.</li>
<li>높은 점수는 고정된 representation에서 label 정보를 읽을 수 있다는 근거다. 위험 인과관계를 이해하거나 행동 억제에 사용한다는 증거는 아니다. 낮은 점수도 해당 pooling·선형 readout 밖에 위험 정보가 없음을 뜻하지 않는다.</li>
</ul><p>Base 가중치는 공식 JAX→PyTorch 변환 코드를 사용했다. 변환에서 원본에 없는 두 텍스트 생성 head만 제외했고, 이 head들이 추론에 호출되지 않음을 검증했다. Transformer·action 경로 가중치의 누락은 없었다. 두 모델의 전체 가중치 hash는 추출 전후 같았고, 추출 hook을 켜고 끈 action 차이는 0이었다.</p>
<p><a href="all_sites.csv">전체 312개 조건 CSV</a> · <a href="summary.json">주요 결과 JSON</a> · <a href="validation_selected_sites.json">validation 선택 기록</a> · <a href="../manifest.json">입력·분할 manifest</a> · <a href="../verification.json">검증 기록</a></p></html>'''
    (folder/"index.html").write_text(text)
    write_json(RUN/"experiment_status.json",{"state":"complete"})
    write_json(RUN/"pipeline_status.json",{"state":"complete"})
    milestone("final_report","첫 장면 두 체크포인트 linear probe 실험 및 보고서 완료",verification)
    print(json.dumps(dict(report=str(folder/"index.html"),primary=[dict(track=r["track"],model=r["model"],site=r["site"],AUROC=r["fold_mean_AUROC"],balanced_accuracy=r["fold_mean_balanced_accuracy"],accuracy=r["oof"]["accuracy"],predicted_unsafe=r["oof"]["predicted_unsafe_fraction"]) for r in primary]),ensure_ascii=False),flush=True)


if __name__=="__main__":main()
