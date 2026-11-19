"""Audit the complete crossed experiment, summarize OOF scores and paired LIBERO outcomes, and render the report."""

from collections import Counter, defaultdict
import csv
import hashlib
from itertools import product
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np

from ae_output_nullspace_pilot import read
from modality_ablation import AE, COMMON, CONDITIONS, REFERENCE, RUN, SITES, STATE_BASE, VLM, conditions
from probe_modality_ablation import csv_rows
from rollout_common import ROOT, write_json
from rollout_modality_ablation import panel

DOC = ROOT / "doc/2026-11-19_Semantic_Safety_Modality_Ablation_v1_Results.md"
FIG = ROOT / "doc/assets/semantic_safety_modality_ablation_v1/probe_delta.png"
DISPLAY = dict(base="Base", libero="LIBERO FT", ae="AE-only", vlm="VLM-only", all="Both", base_state="Base 상태 입력")
CN = dict(intact="전체 입력", global_text="지시 차단", global_image="영상 차단", ae_text="AE 지시 KV 차단",
          ae_image="AE 영상 KV 차단", global_state="상태 차단", ae_state="AE 상태 KV 차단",
          read_image="영상 위치만 읽기", read_text="지시 위치만 읽기")


def index_csv(path):
    with path.open() as stream:
        return {(r["model"],r["condition"],r["track"],r["site"],r["method"]): r for r in csv.DictReader(stream)}


def matched_scores(rows):
    results = []
    for m in COMMON + (STATE_BASE,):
        for cond, track, site, method in product(conditions(m), ("A","B"), SITES, ("fixed","refit")):
            folder = RUN / "conditions" / cond / "probes" / track / m / site
            pred = folder / ("fixed_oof.npz" if method=="fixed" else "oof_predictions.npz")
            with np.load(pred) as obj:
                scores, indices, fold = obj["decision_score"], obj["global_indices"], obj["fold"]
            key = "image_hash" if track=="A" else "instruction"
            groups = defaultdict(lambda: defaultdict(list))
            for i,idx in enumerate(indices):
                r = rows[idx]
                groups[(int(fold[i]), r[key])][r["label"]].append(scores[i])
            deltas, means, ties = [], [], 0
            for cls in groups.values():
                if len(cls) < 2:
                    continue
                delta = np.subtract.outer(cls[1], cls[0]).ravel()
                tie = np.abs(delta) <= 1e-10
                wins = np.where(tie, .5, delta > 0)
                means.append(float(np.mean(wins)))
                deltas.extend(wins.tolist())
                ties += int(tie.sum())
            assert len(deltas) == (18 if track=="A" else 82)
            results.append(dict(model=m,condition=cond,track=track,site=site,method=method,shared_input=key,
                                pairs=len(deltas),fold_key_groups=len(means),tied=ties,
                                unsafe_ranked_higher_pair_mean=float(np.mean(deltas)),
                                unsafe_ranked_higher_group_mean=float(np.mean(means))))
    csv_rows(RUN / "matched_pairs.csv", results)
    return {(r["model"],r["condition"],r["track"],r["site"],r["method"]): r for r in results}


def rollouts():
    episodes, archive, counts = [], [], {}
    specs = panel()
    for m in COMMON + (STATE_BASE,):
        for cond in conditions(m):
            result = []
            baseline = []
            for s in specs:
                def one(c, model=m):
                    return read(RUN / "rollout" / model / c / (s["item_id"] + ".json"))
                item, intact, anchor = one(cond), one("intact"), one("intact","libero")
                assert item["initial_observation_sha256"] == anchor["initial_observation_sha256"]
                assert item["noise_seed"] == anchor["noise_seed"] == s["noise_seed"]
                assert item["steps"] <= s["horizon"] and item["instruction"] == anchor["instruction"]
                assert item["model"] == m and item["condition"] == cond
                if cond == "intact" and m in ("libero","ae","vlm","all"):
                    tag = "baseline" if m == "libero" else m
                    old = read(REFERENCE/"final"/tag/"native/episodes"/s["item_id"]/"summary.json")
                    assert old["initial_image_sha256"] == item["initial_main_sha256"]
                    archive.append(dict(model=m, item=s["item_id"], archive_success=old["success"],
                                        current_success=item["success"], archive_steps=old["steps"], current_steps=item["steps"]))
                episodes.append(item)
                result.append(item["success"])
                baseline.append(intact["success"])
            condition = read(RUN / "rollout" / m / cond / "completion.json")
            wins = int(np.sum(np.array(result) & ~np.array(baseline)))
            losses = int(np.sum(~np.array(result) & np.array(baseline)))
            assert sum(result) == condition["successes"] and len(result)==12
            counts[m,cond] = dict(model=m,condition=cond,successes=sum(result),n=len(result),
                                  rate=sum(result)/len(result),gained_vs_intact=wins,lost_vs_intact=losses,
                                  physics_unstable=condition["unstable"])
    assert len(episodes) == 336 and len(counts)==28
    csv_rows(RUN / "rollout_episodes.csv", episodes)
    csv_rows(RUN / "rollout_summary.csv", list(counts.values()))
    assert len(archive) == 48
    csv_rows(RUN / "archive_native_audit.csv", archive)
    return counts


def figure(probe):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    arms = CONDITIONS[1:]
    fig, axes = plt.subplots(2,2,figsize=(10.8,6.5), sharex=True, sharey=True, constrained_layout=True)
    arrays = {}
    for i, site in enumerate((VLM,AE)):
        for j, track in enumerate(("A","B")):
            arrays[i,j] = np.array([[float(probe[m,c,track,site,"refit"]["delta_vs_intact"])
                                     for c in arms] for m in COMMON])
    lim = max(.05, max(np.max(np.abs(a)) for a in arrays.values()))
    for (i,j), data in arrays.items():
        ax = axes[i,j]
        im = ax.imshow(data, norm=TwoSlopeNorm(0,-lim,lim), cmap="RdBu", aspect="auto")
        ax.set_title(("VLM layer 17" if i==0 else "AE layer 17, flow 10") + "  |  Track " + ("A","B")[j], loc="left")
        ax.set_yticks(range(5),labels=[DISPLAY[m] for m in COMMON])
        ax.set_xticks(range(4),labels=["No text","No vision","AE: no\ntext KV","AE: no\nvision KV"])
        for y,x in np.ndindex(data.shape):
            ax.text(x,y,f"{data[y,x]:+.3f}",ha="center",va="center",fontsize=9,
                    color="white" if abs(data[y,x])>.62*lim else "#121820")
    fig.colorbar(im, ax=axes, label="Change in fold-mean AUROC vs. intact", shrink=.83)
    FIG.parent.mkdir(exist_ok=True,parents=True)
    fig.savefig(FIG,dpi=175)
    plt.close(fig)


def table(headers, data):
    return "\n".join(["| " + " | ".join(headers) + " |","|" + "|".join(["---"]*len(headers))+"|"] +
                     ["| "+" | ".join(map(str,row))+" |" for row in data])


def main():
    p = index_csv(RUN / "probe_summary.csv")
    pool = index_csv(RUN / "pooling_summary.csv")
    assert len(p)==224 and len(pool)==40
    inputs = read(RUN / "input_audit.json")
    assert inputs["unique_initial_states"] == 1
    manifest = read(REFERENCE / "manifest.json")
    protocol = read(RUN / "protocol.json")
    assert hashlib.sha256((REFERENCE / "manifest.json").read_bytes()).hexdigest() == protocol["manifest_sha256"]
    nfit = 0
    for m in COMMON + (STATE_BASE,):
        audit = read(RUN / "audits" / (m + ".json"))
        assert audit["complete"] and audit["rows"] == 457
        if m in COMMON:
            assert read(RUN / "checks" / (m+".json"))["complete"]
            assert not any(audit["perturb_intact"].values())
        else:
            assert not any(audit["perturb_global_state"].values()) and audit["perturb_intact"]["physical_action"]>0
        for c in conditions(m):
            with h5py.File(RUN/"conditions"/c/"features"/(m+".h5")) as f:
                assert f.attrs["complete"] and f["written"][:].all()
            for track, site in product(("A","B"), SITES):
                met = read(RUN/"conditions"/c/"probes"/track/m/site/"metrics.json")
                indices = [i for fold in met["folds"] for i in fold["test_indices"]]
                assert len(set(indices))==len(indices)==(336 if track=="A" else 121)
                nfit += 5
    assert nfit == 560
    for condition in CONDITIONS + ("read_image", "read_text"):
        folder = RUN/"conditions"/condition/"features"
        with h5py.File(folder/"libero.h5") as baseline, h5py.File(folder/"ae.h5") as ae_only:
            np.testing.assert_array_equal(baseline[VLM][:], ae_only[VLM][:])
    matched = matched_scores(manifest["rows"])
    counts = rollouts()
    replay = read(RUN/"checks/native_ae_repeat/audit.json")
    figure(p)
    def auc(model,condition,track,site,method="refit"):
        return float(p[model,condition,track,site,method]["fold_mean_auroc"])
    def median(c,t,s,method="refit"):
        return np.median([auc(m,c,t,s,method) for m in COMMON])
    def cells(model,c,t,site):
        return f"{auc(model,c,t,site,'fixed'):.3f} / {auc(model,c,t,site):.3f}"
    def link(label,file):
        return f"[{label}](../runs/{RUN.name}/{file})"
    sections = ["# π0.5에서 지시·영상·로봇 상태를 가렸을 때의 안전 probe와 LIBERO 과제 성공률"]
    sections.append(f"다섯 모델의 VLM 마지막 층에서 새로 학습한 probe의 AUROC 중앙값은 "
        f"트랙 A에서 전체 입력 {median('intact','A',VLM):.3f}, 지시 차단 {median('global_text','A',VLM):.3f}, "
        f"영상 차단 {median('global_image','A',VLM):.3f}였다. 트랙 B에서는 각각 "
        f"{median('intact','B',VLM):.3f}, {median('global_text','B',VLM):.3f}, {median('global_image','B',VLM):.3f}였다. "
        f"공식 LIBERO FT 정책의 실제 과제 성공은 같은 12개 초기 배치에서 전체 입력 "
        f"{counts['libero','intact']['successes']}/12, 지시 차단 {counts['libero','global_text']['successes']}/12, "
        f"영상 차단 {counts['libero','global_image']['successes']}/12였다.")
    sections += ["## 1. 무엇을 비교했는가",
        "가중치를 고정한 공식 Base·LIBERO FT, benign 과제에 적응한 AE-only 8k·VLM-only 12k·Both 8k를 사용했다. "
        "주 분석에서는 다섯 모델에 동일한 LIBERO 전처리·정규화와 10개 action token, 10회 denoising을 적용했다. "
        "Base의 상태 입력과 horizon 50은 5절에서 별도로 평가했다.",
        table(["조건","VLM에서 읽을 수 있는 입력","AE가 읽을 수 있는 VLM KV"],[
            ["전체 입력","외부·손목 영상과 지시 전체","모든 유효 위치"],
            ["지시 차단","두 카메라 영상","영상 위치"],
            ["영상 차단","지시","지시 위치"],
            ["AE 지시 KV 차단","영상과 지시 전체","영상 위치"],
            ["AE 영상 KV 차단","영상과 지시 전체","지시 위치"]]),
        "전체 입력 단계의 차단은 지정한 prefix 위치를 attention의 유효 토큰에서 제외하고 위치 번호를 다시 계산한다. "
        "지시 차단에는 BOS와 줄바꿈도 포함했다. 영상 차단은 외부 카메라 256개와 손목 카메라 256개 위치에 적용했다. "
        "AE 전용 차단은 VLM이 원래 KV를 만든 뒤 열 방향으로 적용하며, 10번의 denoising 모두에서 위치 번호와 action token 사이의 attention을 보존했다. "
        "이때 영상 위치의 KV에도 지시 정보가, 지시 위치의 KV에도 영상 정보가 섞일 수 있다.",
        "안전 probe는 행동 전 첫 장면 457개에서 측정했다. 지시에 위험 판단에 필요한 사실을 명시한 A는 안전 228·위험 108개, "
        "일부 단서를 관찰에서 얻도록 설계한 B는 안전 63·위험 58개다. 기존 20개 연결 그룹의 5-fold 분할과 트랙별 학습을 유지했다. "
        "훈련 fold 안에서 그룹·클래스 가중치로 표준화와 L2 로지스틱 회귀를 학습하고, 검증 AUROC로 C ∈ {0.01, 0.1, 1}을 골랐다. "
        "아래 AUROC는 test fold 다섯 개의 산술평균이다. 전체 모델 간 중앙값은 같은 데이터에서 얻은 기술 통계다.",
        "VLM 17층의 유효 prefix 토큰과 AE 17층 마지막 flow의 action 토큰을 각각 토큰 축으로 평균했다. 각 장면의 초기 noise는 seed 20261116으로 고정했다. "
        "‘고정’은 전체 입력으로 학습한 각 fold의 표준화와 probe를 마스킹된 test 특징에 그대로 적용한 결과다. "
        "‘재학습’은 같은 마스크를 train·validation·test 모두에 적용하여 해당 조건에서 읽을 수 있는 정보를 측정한다."]
    sections += ["## 2. 첫 장면에서 안전·위험을 구분한 결과",
        "각 칸은 **고정 probe / 재학습 probe**의 AUROC다. AE에서만 차단한 두 조건의 VLM 특징과 점수는 전체 입력과 일치해 VLM 표에는 한 번만 실었다. "
        "AE-only 적응에서는 VLM 가중치가 동결되어, 공통 입력에서 LIBERO FT와 AE-only의 VLM 특징이 모든 조건에서 원소별로 같았다."]
    for site, sname, conds in ((VLM,"VLM 17층",CONDITIONS[:3]),(AE,"AE 17층 · flow 10",CONDITIONS)):
        sections += ["### "+sname,table(["모델","트랙"]+[CN[c] for c in conds],
                          [[DISPLAY[m],t]+[cells(m,c,t,site) for c in conds] for t in ("A","B") for m in COMMON])]
    sections += ["![조건별 재학습 probe의 AUROC 변화](assets/semantic_safety_modality_ablation_v1/probe_delta.png)",
        "그림은 각 모델의 전체 입력 AUROC에서 얼마나 변했는지 표시한다. "
        "판정 임계값 0에서의 정확도와 안전·위험별 오분류도 저장했다. "
        +link("전체 수치와 혼동행렬","probe_summary.csv")+" · "+link("fold별 수치와 선택한 C","probe_folds.csv"),
        "LIBERO FT의 트랙 A에서 영상을 가리자 원래 VLM probe의 AUROC는 0.888에서 0.605로 떨어졌고, "
        "임계값 0에서 336개 중 위험 판정은 113개에서 3개로 줄었다. 같은 지시만 남긴 특징에 새 probe를 학습하면 AUROC는 0.928이었다. "
        "이는 원래 probe의 점수 변화와 남은 입력의 분류 가능성이 서로 다른 측정임을 보여준다.",
        "### 같은 입력 하나를 공유하는 안전·위험 쌍",
        "A에서 두 카메라 영상이 정확히 같은 안전·위험 쌍 18개, B에서 지시 문장이 정확히 같은 안전·위험 쌍 82개를 추렸다. "
        "같은 test fold 안에서 같은 probe가 매긴 점수만 비교했다. 아래는 위험 예시 점수가 더 높은 비율이며, 동점은 0.5점으로 셌다. "
        "같은 표본이 여러 쌍에 쓰일 수 있어 비율은 기술 통계로 해석한다.",
        table(["모델 · 재학습 probe","트랙과 고정한 입력","전체 입력","지시 차단","영상 차단"],
              [[DISPLAY["libero"]+" · "+s,t+(" · 동일 영상 (18쌍)" if t=="A" else " · 동일 문장 (82쌍)")]+[
                  f"{matched['libero',c,t,site,'refit']['unsafe_ranked_higher_pair_mean']:.3f}" for c in CONDITIONS[:3]]
               for t in ("A","B") for site,s in ((VLM,"VLM"),(AE,"AE"))]),
        "B의 121개 예시는 28개 문장을 사용했다. 그중 23개 문장에는 두 라벨이 모두 있고 여기에 92개 예시가 속한다. "
        "문장마다 한 라벨만 내는 결정론적 분류기는 정답 분포를 모두 알아도 전체 B에서 최대 87/121(71.9%)까지 맞힐 수 있다. "
        "이 상한은 정확도에 대한 값이다. 같은 test fold에 안전·위험 예시가 모두 있는 문장만 남기면 86개 예시·82쌍이다. "
        +link("전 모델·조건의 쌍별 순위 집계","matched_pairs.csv"),
        "전체 입력의 LIBERO FT VLM은 B의 82쌍 중 43쌍(52.4%)에서 위험 예시에 더 높은 점수를 매겼다. "
        "서로 다른 문장끼리 비교할 수 있는 전체 B의 AUROC는 0.839였다. 같은 82쌍도 문장에 따라 만들어지는 쌍의 수가 다르다. "
        "각 fold 안의 문장 묶음 27개에 동일한 가중치를 주면 65.5%였다. "
        "동일 문장에서 시각 단서를 활용하는 정도는 비교 단위와 모델에 따라 달랐다.",
        table(["모델","B 전체 입력 VLM · 82쌍","B 전체 입력 AE · 82쌍"], [
            [DISPLAY[m]]+[f"{matched[m,'intact','B',s,'refit']['unsafe_ranked_higher_pair_mean']:.1%}" for s in (VLM,AE)]
            for m in COMMON]),
        "### VLM에서 읽는 위치만 바꾸면",
        "원래 영상과 지시를 모두 처리한 뒤 마지막 층에서 평균낼 위치만 바꿨다. 아래는 다섯 모델의 재학습 VLM probe AUROC 중앙값이다. "
        "같은 열에서 읽는 위치가 같은 두 행을 비교하면, 한 입력을 처음부터 제거한 효과와 마지막에 평균낼 위치를 고른 효과를 구분할 수 있다.",
        table(["VLM이 받은 입력","읽은 위치","트랙 A","트랙 B"],
              [["전체","영상 위치"]+[f"{np.median([float(pool[m,'read_image',t,VLM,'refit']['fold_mean_auroc']) for m in COMMON]):.3f}" for t in ("A","B")],
               ["영상만","영상 위치"]+[f"{median('global_text',t,VLM):.3f}" for t in ("A","B")],
               ["전체","지시 위치"]+[f"{np.median([float(pool[m,'read_text',t,VLM,'refit']['fold_mean_auroc']) for m in COMMON]):.3f}" for t in ("A","B")],
               ["지시만","지시 위치"]+[f"{median('global_image',t,VLM):.3f}" for t in ("A","B")]]),
        "트랙 A에서 같은 영상 위치만 읽어도 지시와 함께 처리한 경우의 중앙값은 0.882, 지시를 처음부터 차단한 경우는 0.799였다. "
        "영상 위치의 마지막 층 특징에도 앞서 함께 처리한 지시가 영향을 준다는 관찰이다. "
        +link("pooling 대조군 전체 수치","pooling_summary.csv")]
    sections += ["## 3. 실제 LIBERO 과제 성공률",
        "네 공식 suite(spatial, object, goal, LIBERO-10)에서 성능을 보기 전에 정한 task ID 0·4·8과 초기 배치 0을 썼다. "
        "모델·조건마다 같은 12개 과제를 1회씩 실행했다. 전체 40개 과제에 대한 추정에는 표본이 작다. 모든 조건에서 "
        "두 영상·로봇 상태의 첫 관측 해시와 첫 instruction이 일치했다. 각 재추론 시점의 noise seed를 과제·초기 배치·제어 step에 고정했다. "
        "10 step마다 새 관측을 받고, 각 제어 step에서 공식 성공 조건을 검사했다. 제한은 460 step, LIBERO-10은 600 step이다. "
        "집계한 336회 모두 기존 evaluator와 같은 FP32 제어 명령을 사용했다.",
        table(["모델"]+[CN[c] for c in CONDITIONS],
              [[DISPLAY[m]]+[f"{counts[m,c]['successes']}/12" for c in CONDITIONS] for m in COMMON]),
        "LIBERO FT는 지시를 가린 상태에서 공간 배치 suite의 그릇 옮기기 세 과제에 모두 성공했지만, "
        "가운데 서랍 열기와 케첩을 바구니에 담기는 실패했다. 일부 배치에서는 관측만으로 익숙한 행동을 선택해 성공할 수 있다. "
        "이 패널에서는 같은 배치에 여러 지시를 번갈아 주지 않아, 지시를 따르는 능력을 8/12라는 수치만으로 추정하기 어렵다.",
        "각 칸에서 1개 과제는 8.3%p다. 마스크가 달라지면 처음 행동부터 경로가 달라질 수 있다. "
        "같은 과제가 성공에서 실패로, 또는 실패에서 성공으로 바뀐 횟수와 개별 종료 step도 기록했다. "
        +link("성공률·짝지은 변화","rollout_summary.csv")+" · "+link("336개 에피소드","rollout_episodes.csv"),
        "LIBERO FT의 AE가 지시 위치의 KV만 읽지 못하게 하면 11/12, 영상 위치의 KV만 읽지 못하게 하면 0/12였다. "
        "이 과제들에서 제어에 필요한 정보가 어느 위치를 통해 전달되는지에 대한 결과다. "
        "영상 위치도 VLM의 지시를 이미 받아들였으므로 11/12라는 성공률로 언어가 무관하다고 단정하기 어렵다.",
        "이전 실험에도 기록이 있는 전체 입력 48회를 대조했다. 첫 외부 카메라 영상은 48/48, 성공 판정은 47/48이 같았다. "
        "AE-only의 LIBERO-10 task 0은 이전 기록에서 600 step 실패, 이번에는 279 step 성공이었다. "
        f"이번 설정으로 그 사례를 별도 반복했을 때 {replay['repeat_steps']} step에서 "
        f"{'성공' if replay['repeat_success'] else '실패'}했다. 이전 평가의 추론 배치 크기는 진행 중인 에피소드 수에 따라 줄었고 "
        "이번에는 12로 고정했다. 배치 차이가 과거 기록과의 차이를 유발했는지는 별개로 검증해야 한다. "
        +link("이전 전체 입력 기록과의 대조","archive_native_audit.csv"),
        "## 4. 공통 설정에서 로봇 상태 입력을 바꾼 결과",
        "다섯 모델의 공통 LIBERO 설정은 discrete_state_input=False이며 π0.5 AE에도 별도의 연속 상태 토큰이 없다. "
        "각 모델에서 첫 12개 원시 8차원 로봇 상태를 정규화·토큰화 전에 실제로 바꿨다. "
        "60개 모델·입력 조합 모두에서 지시 토큰, VLM·AE 특징, 역정규화한 7차원 행동의 최대 절대 차이는 0이었다. "
        "손목 카메라 등 영상에 보이는 로봇 자세는 정책이 계속 사용할 수 있다.",
        "## 5. 상태 토큰을 실제로 받는 Base 보조 실험",
        "같은 공식 Base 가중치에 π0.5의 이산 상태 지시문 형식과 horizon 50을 적용했다. 입력 카메라·LIBERO 정규화는 공통 실험과 동일하고, "
        "rollout은 50개 행동 중 처음 10개를 실행한 뒤 재추론했다. 숫자와 숫자 사이 공백에 해당하는 SentencePiece 위치만 정확히 확인해 가렸다. "
        "전체 입력 단계에서 상태를 가릴 때 남은 문장 토큰을 같은 슬롯에 정렬해 숫자 길이 단서도 제거했다.",
        table(["조건","VLM A","VLM B","AE A","AE B","과제 성공"], [
            [CN[c]]+[cells(STATE_BASE,c,t,s) for s,t in ((VLM,"A"),(VLM,"B"),(AE,"A"),(AE,"B"))]+
            [f"{counts[STATE_BASE,c]['successes']}/12"] for c in conditions(STATE_BASE)]),
        f"첫 장면 457개의 원래 로봇 상태는 한 종류였다. 첫 12개에서 상태값을 바꾸면 전체 입력의 7축 행동 최대 절대 차이는 "
        f"{read(RUN/'audits/base_state.json')['perturb_intact']['physical_action']:.3f}, 상태 전체 차단 뒤에는 0이었다. "
        "이 자료에서 상태 차단 후의 probe 변화는 같은 상태 필드를 넣거나 뺀 데 따른 계산 변화다. 장면별 로봇 자세로 라벨을 판독했는지는 검증할 수 없다. "
        + ("상태 형식 Base는 세 조건 모두 과제 성공이 0/12여서, 상태 정보가 과제 성공에 얼마나 중요한지는 이 패널에서 구분하기 어렵다."
           if all(counts[STATE_BASE,c]["successes"]==0 for c in conditions(STATE_BASE)) else ""),
        "## 6. 해석과 재현",
        "관찰 결합 트랙에서 영상을 모두 가린 LIBERO FT의 재학습 VLM probe는 AUROC 0.811을 기록했지만, "
        "같은 문장 82쌍에는 전부 같은 점수를 줬고 공식 LIBERO 과제는 0/12였다. "
        "전체 B에서 문장별 라벨 분포를 활용할 수 있다는 점과 실제 행동 제어에 시각 정보가 필요하다는 점이 함께 드러난다. "
        "첫 장면의 안전성 연구에서 전체 AUROC와 동일 문장·서로 다른 관찰 안에서의 위험 순위를 함께 보는 근거다.",
        "고정 probe의 하락은 원래 분류 경계가 새 특징 분포에서 얼마나 유지되는지를 보여준다. 재학습 후의 AUROC는 남은 특징으로 새 경계를 찾았을 때의 결과다. "
        "마스킹은 학습 때와 다른 입력을 만들며 VLM 전체 차단은 pooling의 구성도 바꾼다. 문장을 가리고 영상만으로 라벨을 구분해도 물체·배경 단서의 기여가 포함된다. "
        "probe가 위험을 읽을 수 있는지, 모델이 위험을 알아차려 행동을 중단하는지, 일반 LIBERO 작업을 성공하는지는 각각 별도의 질문이다.",
        "정적 입력의 전체 입력 특징은 공통 모델 5종·457개·두 위치에서 기존 실험과 원소별로 일치했다. "
        "각 모델의 12개 입력에서 텍스트 또는 영상을 서로 바꾸면 원래 행동이 변했고, 해당 입력을 전체 차단하면 특징과 행동 변화가 모두 0이었다. "
        "pooling 대조에서는 전체 토큰 평균이 기존 VLM 특징과 일치했다. "
        +link("고정한 실험 설정","protocol.json")+" · "+link("입력 중복·상태 감사","input_audit.json")
        +" · "+link("환경별 라이브러리 버전","runtime_versions.json"),
        "저장소 루트에서 다음 명령으로 실행한다. GPU 추론은 openpi 환경, probe·보고서는 scikit-learn 환경, 시뮬레이션 worker는 전용 LIBERO 환경을 사용한다.",
        "```bash\nexport PYTHONPATH=\"$PWD/openpi/src:$PWD/openpi/packages/openpi-client/src\"\n"
        "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu WANDB_MODE=disabled\n"
        "s=scripts/semantic_safety\ngpu=openpi/.venv/bin/python\ncpu=envs/representation-probe/bin/python\n"
        "\"$cpu\" \"$s/modality_ablation.py\" prepare\n"
        "for model in libero base_state; do\n  \"$gpu\" \"$s/modality_ablation.py\" extract --model \"$model\" --limit 16\ndone\n"
        "for model in libero ae vlm all base base_state; do\n  \"$gpu\" \"$s/modality_ablation.py\" extract --model \"$model\"\ndone\n"
        "\"$gpu\" \"$s/check_modality_ablation.py\"\n\"$cpu\" \"$s/probe_modality_ablation.py\"\n"
        "\"$gpu\" \"$s/modality_pooling_control.py\"\n\"$cpu\" \"$s/probe_modality_ablation.py\" --pooling\n"
        "\"$gpu\" \"$s/rollout_modality_ablation.py\" --smoke\n\"$gpu\" \"$s/rollout_modality_ablation.py\"\n"
        "\"$gpu\" \"$s/rollout_modality_ablation.py\" --repeat-native-case\n"
        "\"$cpu\" \"$s/report_modality_ablation.py\"\n```"]
    DOC.write_text("\n\n".join(sections)+"\n")
    sources = [Path(__file__), *(Path(__file__).parent/n for n in
        ("modality_ablation.py","check_modality_ablation.py","probe_modality_ablation.py","modality_pooling_control.py",
         "rollout_modality_ablation.py","ae_output_nullspace_pilot.py","contrast_initial_linear.py","motor_lora.py",
         "evaluate_expansion_v5_lora.py","benign_policy_v1.py","v6_scoped_lora.py")),
        ROOT/"openpi/src/openpi/models_pytorch/pi0_pytorch.py", ROOT/"openpi/src/openpi/models/tokenizer.py",
        ROOT/"openpi/src/openpi/training/config.py", ROOT/"openpi/src/openpi/models/pi0_config.py",
        RUN/"protocol.json", REFERENCE/"frozen_selection.json",
        REFERENCE/"manifest.json", RUN/"pooling_control_protocol.json"]
    write_json(RUN/"sources.json", {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    runtimes = {}
    for env, packages in {"openpi/.venv": ["torch","transformers","sentencepiece","numpy","h5py"],
                          "envs/libero-render": ["torch","mujoco","numpy"],
                          "envs/representation-probe": ["numpy","scikit-learn","h5py","matplotlib"]}.items():
        snippet = ("from importlib.metadata import version; import json,sys; print(json.dumps("
                   "{'python':sys.version.split()[0], **{p:version(p) for p in sys.argv[1:]}}))")
        runtimes[env] = json.loads(subprocess.check_output([str(ROOT/env/"bin/python"), "-c", snippet, *packages]))
    write_json(RUN/"runtime_versions.json", runtimes)
    status = dict(state="complete", initial_decisions=457, main_model_condition_pairs=25, state_model_condition_pairs=3,
                  primary_probe_metrics=len(p), pooling_probe_metrics=len(pool), native_rollouts=336,
                  unstable_rollouts=sum(v["physics_unstable"] for v in counts.values()), paired_tasks=12,
                  report=str(DOC.relative_to(ROOT)))
    for path in ("experiment_status.json","pipeline_status.json"):
        write_json(RUN/path,status)
    from experiment_notifications import notify_progress
    notify_progress(RUN)
    print(json.dumps(status,ensure_ascii=False))


if __name__ == "__main__":
    main()
