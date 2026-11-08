"""Render saved held-out results; no model inference or probe fitting."""

from __future__ import annotations

import argparse
import json
import os
import pathlib

os.environ.setdefault("MPLCONFIGDIR", str(pathlib.Path(__file__).resolve().parents[2] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = ("pi05", "pi05_libero", "openvla_oft")
NAMES = ("π0.5 base", "π0.5 LIBERO", "OpenVLA-OFT Spatial")


def save_new(fig, run, stem):
    for extension in ("png", "pdf"):
        path = run / f"{stem}.{extension}"
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=pathlib.Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / "results.json").read_text())
    fig = plt.figure(figsize=(15, 7.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)
    axis = fig.add_subplot(grid[0, :])
    methods = ("unsafe_token_logp", "likelihood_margin", "lda", "linear_probe")
    labels = ("−NLL(yes), unprompted", "LL yes−no, unprompted", "Shrinkage LDA", "Linear probe")
    positions, width = np.arange(3), 0.18
    for j, (method, label) in enumerate(zip(methods, labels)):
        metrics = [report[model]["instruction"]["methods"][method] for model in MODELS]
        means = np.array([metric["mean_fold_auroc"] for metric in metrics])
        intervals = np.array([metric["scene_bootstrap_auroc_ci95"] for metric in metrics])
        error = np.maximum(0, np.vstack([means - intervals[:, 0], intervals[:, 1] - means]))
        axis.bar(positions + (j-1.5)*width, means, width, yerr=error, capsize=3, label=label)
    axis.axhline(0.5, linestyle="--", color="gray", linewidth=1)
    axis.set(xticks=positions, xticklabels=NAMES, ylabel="Mean held-out-fold AUROC", ylim=(0, 1.05),
             title="Robot instructions without an added safety question: frozen VLM-prefix representations")
    axis.legend(ncol=4, fontsize=9, loc="upper center")
    for j, (model, name) in enumerate(zip(MODELS, NAMES)):
        axis = fig.add_subplot(grid[1, j])
        with np.load(args.run / "analysis" / model / "instruction" / "oof_predictions.npz", allow_pickle=False) as data:
            score, y = data["lda_projection"], data["label"]
        edges = np.linspace(float(score.min()) - 0.05, float(score.max()) + 0.05, 17)
        for label, color, text in ((0, "#26845b", "Source non-undesirable"), (1, "#c95353", "Source undesirable")):
            axis.hist(score[y == label], bins=edges, density=True, alpha=0.55, color=color, label=text)
        axis.set(title=name, xlabel="Held-out LDA score (train-normalized)", ylabel="Density")
        axis.legend(fontsize=7)
    save_new(fig, args.run, "figure3_instruction_prefix")

    fig, axis = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    for j, (condition, label) in enumerate((("query", "Safety query + observed image"),
                                          ("instruction", "Instruction + observed image"),
                                          ("neutral_query", "Safety query + neutral image"))):
        metrics = [report[model][condition]["methods"]["linear_probe"] for model in MODELS]
        means = np.array([metric["mean_fold_auroc"] for metric in metrics])
        intervals = np.array([metric["scene_bootstrap_auroc_ci95"] for metric in metrics])
        axis.bar(positions + (j-1)*0.23, means, 0.23,
                 yerr=np.maximum(0, np.vstack([means-intervals[:, 0], intervals[:, 1]-means])), capsize=3, label=label)
    axis.axhline(report["text_baseline"]["mean_fold_auroc"], color="#663399", linestyle="--", label="Text-only TF-IDF baseline")
    axis.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    axis.set(xticks=positions, xticklabels=NAMES, ylim=(0, 1.15), ylabel="Mean held-out-fold AUROC",
             title="Controls: label readout is not evidence of visual safety integration")
    axis.legend(ncol=2, loc="upper center", fontsize=9)
    save_new(fig, args.run, "controls_robot_semantic_safety")
    print("ADDITIONAL_FIGURES_COMPLETE", args.run)


if __name__ == "__main__":
    main()
