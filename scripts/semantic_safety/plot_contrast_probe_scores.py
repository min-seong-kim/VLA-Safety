"""Visualize held-out first-scene scores on the frozen probes' own decision axes."""

import json

import h5py
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits

from contrast_initial_linear import C_GRID, MODELS, RUN, digest
from plot_contrast_initial_tsne import LABEL_STYLE, MODEL_NAMES, SITES, legend_marker
from rollout_common import ROOT


OUTPUT = ROOT / "doc/assets/contrast_probe_scores_v1"
TRACK_NAMES = {"A": "Explicit instructions", "B": "Observation-dependent candidates"}
FOLD_COLORS = ("#316282", "#ce7c26", "#427d69", "#97608a", "#686a95")


def residual_pc(train, test, weights):
    direction = weights / np.linalg.norm(weights)
    train_residual = train - np.outer(train @ direction, direction)
    test_residual = test - np.outer(test @ direction, direction)
    pca = PCA(n_components=1, svd_solver="full").fit(train_residual)
    assert abs(pca.components_[0] @ direction) < 1e-7
    return pca.transform(test_residual)[:, 0]


def read_folds(manifest, track, model, site, matrix, feature_signature):
    rows = manifest["rows"]
    indices = np.array([i for i, r in enumerate(rows) if r["track"] == track])
    folder = RUN / "probes" / track / model / site
    metrics = json.loads((folder / "metrics.json").read_text())
    signature = digest([feature_signature, track, site, manifest["folds"], C_GRID, "group_then_class_weights_v1"])
    assert metrics["complete"] and metrics["signature"] == signature
    with np.load(folder / "oof_predictions.npz") as oof:
        np.testing.assert_array_equal(oof["global_indices"], indices)
        labels, fold_ids, scores = oof["label"], oof["fold"], oof["decision_score"]
    np.testing.assert_array_equal(labels, [rows[i]["label"] for i in indices])
    assert np.isfinite(scores).all()
    folds = []
    for split, record in zip(manifest["folds"], metrics["folds"], strict=True):
        fold_id = split["fold"]
        mask = fold_ids == fold_id
        train = [i for i in indices if rows[i]["group_id"] in split["train_groups"]]
        test = [i for i in indices if rows[i]["group_id"] in split["test_groups"]]
        np.testing.assert_array_equal(indices[mask], test)
        assert fold_id == record["fold"] and train == record["train_indices"] and test == record["test_indices"]
        assert set(split["train_groups"]).isdisjoint(split["test_groups"])
        saved = joblib.load(folder / f"fold_{fold_id}.joblib")
        assert saved["signature"] == signature and saved["training_rows"] == [rows[i]["item_id"] for i in train]
        scaler, probe = saved["scaler"], saved["classifier"]
        train_z, test_z = scaler.transform(matrix[train]), scaler.transform(matrix[test])
        np.testing.assert_allclose(probe.decision_function(test_z), scores[mask], rtol=1e-8, atol=1e-8)
        pc = residual_pc(train_z, test_z, probe.coef_[0])
        assert np.isfinite(pc).all()
        auc = roc_auc_score(labels[mask], scores[mask])
        np.testing.assert_allclose(auc, record["test"]["AUROC"])
        folds.append(dict(id=fold_id, labels=labels[mask], scores=scores[mask], pc=pc, auc=auc))
    assert sum(len(f["labels"]) for f in folds) == len(indices)
    np.testing.assert_allclose(np.mean([f["auc"] for f in folds]), metrics["fold_mean_AUROC"])
    return folds


def header(fig, track, title, subtitle):
    fig.suptitle(f"Track {track}   /   {title}", x=.055, y=.986, ha="left", fontsize=20, weight="bold")
    fig.text(.055, .931, f"{TRACK_NAMES[track]}   ·   {subtitle}", fontsize=10, color="#425767")
    handles = [legend_marker(*style) for style in LABEL_STYLE.values()]
    fig.legend(handles=handles, frameon=False, ncol=2, loc="upper right",
               bbox_to_anchor=(.985, .992), columnspacing=1.1, handletextpad=.2)


def score_page(track, panels):
    fig, axes = plt.subplots(2, len(panels), figsize=(13.2, 7.0))
    fig.subplots_adjust(left=.07, right=.98, top=.835, bottom=.145, wspace=.25, hspace=.49)
    header(fig, track, "Held-out probe scores", "Each dot is one test input. Dashed line = decision threshold.")
    for col, ((model, site), folds) in enumerate(panels.items()):
        ax, roc = axes[:, col]
        ax.set_title(f"{MODEL_NAMES[model]}  /  {SITES[site]}", fontsize=10, weight="bold")
        ax.axvline(0, color="#59626a", lw=1, ls="--")
        roc.plot([0, 1], [0, 1], lw=1, ls=":", color="#80888b")
        for f in folds:
            fid, y, scores = f["id"], f["labels"], f["scores"]
            # Jitter only the categorical fold row, keeping every decision score exact.
            jitter = np.random.default_rng(100 + fid).uniform(-.20, .20, size=len(y))
            for label, (_, color, marker) in LABEL_STYLE.items():
                keep = y == label
                ax.scatter(scores[keep], fid + jitter[keep], marker=marker, s=18, c=color, alpha=.79,
                           edgecolors="white", linewidths=.28)
            fpr, tpr, _ = roc_curve(y, scores, drop_intermediate=False)
            roc.plot(fpr, tpr, lw=1.4, color=FOLD_COLORS[fid], label=f"Fold {fid}", alpha=.9)
        ax.set_yticks(range(5), [f"F{i}" for i in range(5)])
        ax.set_ylim(4.55, -.55)
        ax.set_xlabel("Probe score   →   unsafe", fontsize=9)
        roc.set_title(f'Mean fold AUROC   {np.mean([f["auc"] for f in folds]):.3f}', fontsize=11)
        roc.set(xlim=(-.025, 1.025), ylim=(-.025, 1.025), xticks=[0,.5,1], yticks=[0,.5,1])
        roc.set_xlabel("False positive rate", fontsize=9)
        if col == 0:
            roc.set_ylabel("True positive rate", fontsize=9)
        for axis in (ax, roc):
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelsize=8)
    fig.legend(*axes[1, 0].get_legend_handles_labels(), ncol=5, frameon=False,
               loc="lower center", bbox_to_anchor=(.52, .005))
    return fig


def projection_page(track, panels):
    fig, axes = plt.subplots(5, len(panels), figsize=(12.5, 11), sharex="col")
    fig.subplots_adjust(left=.09, right=.98, top=.87, bottom=.07, wspace=.16, hspace=.15)
    header(fig, track, "Projections of test inputs",
           "Horizontal = probe score. Vertical = first remaining PC fitted on that fold's training inputs.")
    for col, ((model, site), folds) in enumerate(panels.items()):
        axes[0, col].set_title(f"{MODEL_NAMES[model]}  /  {SITES[site]}", fontsize=10, weight="bold")
        for f in folds:
            ax = axes[f["id"], col]
            ax.axvline(0, color="#59626a", lw=.9, ls="--")
            for label, (_, color, marker) in LABEL_STYLE.items():
                keep = f["labels"] == label
                ax.scatter(f["scores"][keep], f["pc"][keep], marker=marker, c=color, s=23, alpha=.84,
                           edgecolors="white", linewidths=.35)
            ax.set_facecolor("#f8fafb")
            ax.tick_params(labelsize=8)
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#d1d8dc")
            if col == 0:
                ax.set_ylabel(f'Fold {f["id"]}\nn = {len(f["labels"])}', rotation=0, ha="right", va="center", labelpad=7)
        axes[-1, col].set_xlabel("Probe score   →   unsafe", fontsize=9)
    return fig


def main():
    manifest = json.loads((RUN / "manifest.json").read_text())
    protocol = json.loads((RUN / "experiment_protocol.json").read_text())
    panels = {track: {} for track in TRACK_NAMES}
    with threadpool_limits(limits=1):
        for model in MODELS:
            with h5py.File(RUN / "features" / f"{model}.h5", "r") as source:
                sig = digest([manifest["rows"], source.attrs["model_weight_digest"], protocol])
                assert source.attrs["signature"] == sig and source.attrs["complete"] and source["written"][:].all()
                for site in SITES:
                    matrix = source[site][:].astype(np.float64)
                    for track in TRACK_NAMES:
                        panels[track][model, site] = read_folds(manifest, track, model, site, matrix, sig)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT / "held_out_probe_scores.pdf") as pdf:
        for track, track_panels in panels.items():
            for name, draw in [("scores", score_page), ("projection", projection_page)]:
                fig = draw(track, track_panels)
                fig.savefig(OUTPUT / f"{name}_{track}.png", dpi=160)
                pdf.savefig(fig)
                plt.close(fig)


if __name__ == "__main__":
    main()
