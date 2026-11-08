"""Compare likelihood and linear readouts using held-out robot-scene groups.

All fitting, hyperparameter selection and projection normalization use only the
outer training/development groups. Plots contain out-of-fold predictions only.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

MODELS = ("pi05", "pi05_libero", "openvla_oft")
CONDITIONS = ("query", "instruction", "neutral_query")
METHODS = ("unsafe_token_logp", "likelihood_margin", "lda", "linear_probe", "output_probe")
C_GRID = (0.01, 0.1, 1.0, 10.0)
SHRINKAGES = (0.9, 0.5, 0.1, 0.01)


def fit_lda(x: np.ndarray, y: np.ndarray, shrinkage: float) -> tuple[np.ndarray, float]:
    """Balanced-prior LDA with spherical covariance shrinkage via Woodbury.

    Equivalent to solving [(1-lambda) S + lambda trace(S)/d I] w = mu1-mu0.
    The dual solve avoids constructing a 4096-by-4096 covariance matrix.
    """
    means = [x[y == label].mean(axis=0) for label in (0, 1)]
    if any(np.sum(y == label) == 0 for label in (0, 1)):
        raise ValueError("LDA training needs both classes")
    centered = np.concatenate([(x[y == label] - means[label]) / np.sqrt(2 * np.sum(y == label))
                               for label in (0, 1)])
    tau = float(np.sum(centered * centered) / x.shape[1])
    ridge = shrinkage * tau
    if ridge <= 0:
        raise ValueError("Degenerate LDA covariance")
    u = np.sqrt(1 - shrinkage) * centered
    delta = means[1] - means[0]
    w = (delta - u.T @ np.linalg.solve(ridge * np.eye(len(u)) + u @ u.T, u @ delta)) / ridge
    intercept = -0.5 * float((means[1] + means[0]) @ w)
    return w, intercept


def folds_for(groups: np.ndarray, seed: int = 42) -> list[dict]:
    unique = np.array(sorted(set(groups)))
    if len(unique) < 10:
        raise ValueError("At least ten scene groups are required for five-fold evaluation")
    np.random.default_rng(seed).shuffle(unique)
    folds = []
    for fold_id, test in enumerate(np.array_split(unique, 5)):
        rest = np.array([group for group in unique if group not in set(test)])
        np.random.default_rng(seed + 100 + fold_id).shuffle(rest)
        n_dev = max(2, int(round(len(rest) * 0.2)))
        development, train = rest[:n_dev], rest[n_dev:]
        if set(train) & set(development) or set(train) & set(test) or set(development) & set(test):
            raise ValueError("Scene leakage across fold partitions")
        folds.append({"fold": fold_id, "train": train.tolist(), "dev": development.tolist(), "test": test.tolist()})
    return folds


def select_lr(x_train, y_train, x_dev, y_dev):
    best = None
    for c in C_GRID:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = LogisticRegression(C=c, solver="liblinear", class_weight="balanced", max_iter=5000, random_state=42)
            model.fit(x_train, y_train)
        converged = (not any(issubclass(w.category, ConvergenceWarning) for w in caught)
                     and int(np.max(model.n_iter_)) < model.max_iter)
        if not converged:
            raise ValueError("Linear probe did not converge")
        score = roc_auc_score(y_dev, model.decision_function(x_dev))
        if best is None or score > best[0] + 1e-12:
            best = (score, c, model)
    return best[2], {"C": best[1], "dev_auroc": best[0]}


def score_metrics(y, score):
    return {"auroc": float(roc_auc_score(y, score)), "auprc_unsafe": float(average_precision_score(y, score)),
            "balanced_accuracy_at_zero": float(balanced_accuracy_score(y, score >= 0))}


def summarize_scores(y, score, groups, fold_ids, bootstrap=200):
    per_fold = [score_metrics(y[fold_ids == fold], score[fold_ids == fold]) for fold in range(5)]
    per_scene = [roc_auc_score(y[groups == group], score[groups == group]) for group in sorted(set(groups))]
    rng = np.random.default_rng(42)
    estimates = []
    for _ in range(bootstrap):
        values = []
        for fold in range(5):
            available = sorted(set(groups[fold_ids == fold]))
            sampled = rng.choice(available, size=len(available), replace=True)
            selected = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
            values.append(roc_auc_score(y[selected], score[selected]))
        estimates.append(float(np.mean(values)))
    return {"mean_fold_auroc": float(np.mean([m["auroc"] for m in per_fold])),
            "mean_fold_auprc_unsafe": float(np.mean([m["auprc_unsafe"] for m in per_fold])),
            "mean_fold_balanced_accuracy": float(np.mean([m["balanced_accuracy_at_zero"] for m in per_fold])),
            "macro_scene_auroc": float(np.mean(per_scene)), "per_fold": per_fold,
            "scene_bootstrap_auroc_ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
            "ci_scope": "Scene resampling within held-out folds; fitted models fixed; no training-seed uncertainty"}


def evaluate_one(data, folds, output):
    x, y, groups = data["h"].astype(np.float64), data["label"], data["group_id"]
    n = len(y)
    output_scores = np.column_stack([data["cand_logp"], data["entropy"], data["n_prompt_tokens"]])
    scores = {method: np.full(n, np.nan) for method in METHODS}
    projection = np.full(n, np.nan)
    fold_ids = np.full(n, -1, dtype=int)
    choices = []
    output.mkdir(parents=True, exist_ok=False)
    for fold in folds:
        train, dev, test = [np.flatnonzero(np.isin(groups, fold[key])) for key in ("train", "dev", "test")]
        for indices in (train, dev, test):
            if len(set(y[indices])) != 2:
                raise ValueError("Each fold partition must contain both source labels")
        scaler = StandardScaler().fit(x[train])
        xt, xd, xe = [scaler.transform(x[index]) for index in (train, dev, test)]
        best = None
        for shrinkage in SHRINKAGES:
            w, intercept = fit_lda(xt, y[train], shrinkage)
            dev_auc = roc_auc_score(y[dev], xd @ w + intercept)
            if best is None or dev_auc > best[0] + 1e-12:
                best = (dev_auc, shrinkage, w, intercept)
        dev_auc, shrinkage, w, intercept = best
        lda_score = xe @ w + intercept
        scores["lda"][test] = lda_score
        training_projection = xt @ w + intercept
        projection[test] = (lda_score - training_projection.mean()) / max(training_projection.std(), 1e-12)
        lr, lr_choice = select_lr(xt, y[train], xd, y[dev])
        scores["linear_probe"][test] = lr.decision_function(xe)
        output_scaler = StandardScaler().fit(output_scores[train])
        output_lr, output_choice = select_lr(output_scaler.transform(output_scores[train]), y[train],
                                             output_scaler.transform(output_scores[dev]), y[dev])
        scores["output_probe"][test] = output_lr.decision_function(output_scaler.transform(output_scores[test]))
        scores["likelihood_margin"][test] = data["cand_logp"][test, 1] - data["cand_logp"][test, 0]
        scores["unsafe_token_logp"][test] = data["cand_logp"][test, 1]
        fold_ids[test] = fold["fold"]
        choices.append({"fold": fold["fold"], "lda_shrinkage": shrinkage, "lda_dev_auroc": dev_auc,
                        "linear_probe": lr_choice, "output_probe": output_choice})
        joblib.dump({"scaler": scaler, "lda_w": w, "lda_intercept": intercept, "linear_probe": lr,
                     "output_scaler": output_scaler, "output_probe": output_lr, "fold": fold},
                    output / f"fold_{fold['fold']}.joblib")
    if (fold_ids < 0).any() or any(not np.isfinite(score).all() for score in scores.values()):
        raise ValueError("Missing out-of-fold prediction")
    summary = {method: summarize_scores(y, score, groups, fold_ids) for method, score in scores.items()}
    np.savez(output / "oof_predictions.npz", item_id=data["item_id"], group_id=groups, label=y, fold=fold_ids,
             lda_projection=projection, **scores)
    return {"n": n, "n_scenes": len(set(groups)), "class_counts": np.bincount(y).tolist(),
            "methods": summary, "dev_choices": choices}


def evaluate_text(data, folds, output):
    text = np.char.add(np.char.add(data["context"], "\n"), data["instruction"])
    y, groups = data["label"], data["group_id"]
    scores, fold_ids = np.full(len(y), np.nan), np.full(len(y), -1, dtype=int)
    for fold in folds:
        train, dev, test = [np.flatnonzero(np.isin(groups, fold[key])) for key in ("train", "dev", "test")]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=10000)
        xt = vectorizer.fit_transform(text[train])
        model, _ = select_lr(xt, y[train], vectorizer.transform(text[dev]), y[dev])
        scores[test] = model.decision_function(vectorizer.transform(text[test]))
        fold_ids[test] = fold["fold"]
    np.savez(output / "text_baseline_oof.npz", item_id=data["item_id"], label=y, group_id=groups, fold=fold_ids, score=scores)
    return summarize_scores(y, scores, groups, fold_ids)


def make_figure(run, report):
    names = {"pi05": "π0.5 base", "pi05_libero": "π0.5 LIBERO", "openvla_oft": "OpenVLA-OFT Spatial"}
    fig = plt.figure(figsize=(15, 7.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)
    axis = fig.add_subplot(grid[0, :])
    methods = ("unsafe_token_logp", "likelihood_margin", "lda", "linear_probe")
    labels = ("−NLL(yes)", "LL yes−no", "Shrinkage LDA", "Linear probe")
    positions, width = np.arange(len(MODELS)), 0.18
    for j, (method, label) in enumerate(zip(methods, labels)):
        metrics = [report[model]["query"]["methods"][method] for model in MODELS]
        values = np.array([metric["mean_fold_auroc"] for metric in metrics])
        intervals = np.array([metric["scene_bootstrap_auroc_ci95"] for metric in metrics])
        error = np.maximum(0, np.vstack([values-intervals[:, 0], intervals[:, 1]-values]))
        axis.bar(positions+(j-1.5)*width, values, width, yerr=error, capsize=3, label=label)
    axis.axhline(0.5, linestyle="--", color="gray", linewidth=1)
    axis.set(xticks=positions, xticklabels=[names[model] for model in MODELS], ylim=(0, 1.05),
             ylabel="Mean held-out-fold AUROC", title="Robot semantic safety: likelihood vs. representation readouts")
    axis.legend(ncol=4, loc="upper center", fontsize=9)
    for j, model in enumerate(MODELS):
        axis = fig.add_subplot(grid[1, j])
        data = np.load(run / "analysis" / model / "query" / "oof_predictions.npz", allow_pickle=False)
        score, y = data["lda_projection"], data["label"]
        edges = np.linspace(float(score.min())-0.05, float(score.max())+0.05, 17)
        for label, color, name in ((0, "#26845b", "Source non-undesirable"), (1, "#c95353", "Source undesirable")):
            axis.hist(score[y == label], bins=edges, density=True, alpha=0.55, label=name, color=color)
        axis.set(title=names[model], xlabel="Held-out LDA score (train-normalized)", ylabel="Density")
        axis.legend(fontsize=7)
    fig.savefig(run / "figure3_robot_semantic_safety.png", dpi=200)
    fig.savefig(run / "figure3_robot_semantic_safety.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Probe fitting must run in a Slurm allocation")
    destination = args.run / "analysis"
    destination.mkdir(parents=True, exist_ok=False)
    reference = np.load(args.run / "features" / MODELS[0] / "query.npz", allow_pickle=False)
    folds = folds_for(reference["group_id"])
    (args.run / "folds.json").write_text(json.dumps(folds, indent=2))
    report = {"protocol": "Five outer image-group folds; inner group dev; frozen models; unsafe positive",
              "positive_label": "Original ASIMOV undesirable_groundtruth_answer",
              "metric": "Mean within-fold AUROC; scene bootstrap; pooled cross-fold AUROC is not the headline metric"}
    report["text_baseline"] = evaluate_text(reference, folds, destination)
    for model in MODELS:
        report[model] = {}
        for condition in CONDITIONS:
            with np.load(args.run / "features" / model / f"{condition}.npz", allow_pickle=False) as packed:
                data = {key: packed[key] for key in packed.files}
            for key in ("item_id", "label", "group_id"):
                if not np.array_equal(data[key], reference[key]):
                    raise ValueError("Models and conditions must share exact item IDs, labels and scene groups")
            report[model][condition] = evaluate_one(data, folds, destination / model / condition)
            print(model, condition, {key: round(value["mean_fold_auroc"], 4)
                                    for key, value in report[model][condition]["methods"].items()}, flush=True)
    with (args.run / "results.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    make_figure(args.run, report)
    lines = ["# 로봇 환경의 의미적 안전성 표현 분석", "",
             "ASIMOV RoboVQA 기반 로봇 장면의 사전 고정 연구용 부분집합. Source undesirable label을 그대로 사용했다.",
             "일반 VLM chat 데이터·낙하/충돌 중심 사례·과제 무관 방해 사례는 주 분석에서 제외했다.", "",
             "5개 outer scene-group fold와 각 fold 내부의 독립 dev scene을 사용했다. 모든 표현과 그림은 held-out 예측이다.",
             "정답·해설·constitution은 입력하지 않았다. 모델·기존 adapter·head는 고정했다.", "",
             "| 모델 | 조건 | −NLL(yes) AUROC | LL margin AUROC | LDA AUROC | Linear AUROC | 출력 통계 probe AUROC |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        for condition in CONDITIONS:
            methods = report[model][condition]["methods"]
            values = [f"{methods[method]['mean_fold_auroc']:.3f}" for method in METHODS]
            lines.append("| " + " | ".join([model, condition] + values) + " |")
    lines += ["", f"텍스트 TF-IDF baseline AUROC: {report['text_baseline']['mean_fold_auroc']:.3f}.", "",
              "query는 안전성 질문, instruction은 추가 안전성 질문 없는 context+지시, neutral_query는 동일 크기 중립 이미지 통제다.",
              "instruction도 기존 model-specific VLM wrapper를 사용하며 native action expert 입력이나 실제 행동 실행을 뜻하지 않는다.",
              "지표는 fold별 AUROC 평균이다. CI와 장면별 macro AUROC는 results.json에 있다.", "",
              "![Figure 3 형태 비교](figure3_robot_semantic_safety.png)", "",
              "## 해석의 범위", "",
              "- LDA/linear 우위는 선택한 위치에 지도 선형 판독 가능한 source label 신호가 있다는 뜻이다.",
              "- 텍스트 또는 중립 이미지 조건도 높으면 시각적 의미 통합의 증거로 확대하지 않는다.",
              "- Source desirability는 인간 전문가가 새로 검증한 harm label이 아니다. 개별 source label에는 맥락 의존성과 오류 가능성이 있다.",
              "- 19개 독립 장면의 탐색적 결과다. 완전한 native-policy 표현·거절·안전 행동 또는 일반 배치 안전성을 입증하지 않는다.",
              "- LDA의 두 class 판별 축은 하나다. 그림은 test LDA 1차원 분포이며 supervised 2차원 투영을 만들지 않았다."]
    (args.run / "report.md").write_text("\n".join(lines) + "\n")
    (args.run / "completion.json").write_text(json.dumps({"status": "complete", "job_id": os.environ["SLURM_JOB_ID"]}))
    print(f"ANALYSIS_COMPLETE {args.run}", flush=True)


if __name__ == "__main__":
    main()
