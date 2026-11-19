"""Plot label-blind t-SNE maps of the original first-scene probe features."""

import csv
import json

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

from contrast_initial_linear import MODELS, RUN, SEED, digest
from rollout_common import ROOT


OUTPUT = ROOT / "doc/assets/contrast_initial_tsne_v1"
SITES = {"vlm/layer_17": "VLM · layer 17", "ae/flow_10/layer_17": "AE · layer 17 · flow 10"}
MODEL_NAMES = {"base": "Base", "libero": "LIBERO FT"}
PRIMARY_PERPLEXITY = 30
NEIGHBORS = 10
LABEL_STYLE = {0: ("Safe", "#237a9d", "o"), 1: ("Unsafe", "#d15b28", "^")}
CATEGORY_STYLE = {
    "body": ("Body contact", "#0072B2"), "infant": ("Infant", "#E69F00"),
    "food": ("Food", "#009E73"), "foreign": ("Non-food items", "#CC79A7"),
    "chemical": ("Chemicals", "#7B4AB1"), "ecology": ("Habitat", "#634730"),
    "hygiene": ("Hygiene", "#659822"), "heat": ("Heat", "#D55E00"),
    "electrical": ("Electronics", "#5AB4DC"), "energy": ("Stored energy", "#B52C54"),
    "property": ("Property", "#606976"), "function": ("Function", "#D0B129"),
}


def matching_neighbors(points, values):
    """Mean matching fraction among k neighbors, averaging ties at the boundary."""
    distances = squareform(pdist(points, metric="sqeuclidean"))
    np.fill_diagonal(distances, np.inf)
    boundary = np.partition(distances, NEIGHBORS - 1, axis=1)[:, [NEIGHBORS - 1]]
    closer = distances < boundary
    tied = distances == boundary
    tie_share = (NEIGHBORS - closer.sum(axis=1)) / tied.sum(axis=1)
    weights = closer + tied * tie_share[:, None]
    matches = values[:, None] == values[None, :]
    return float((weights * matches).sum() / (len(points) * NEIGHBORS))


def random_match(values):
    counts = np.unique(values, return_counts=True)[1]
    return float(counts @ (counts - 1) / (len(values) * (len(values) - 1)))


def read_features(rows, protocol):
    matrices = {}
    fingerprints = None
    for model in MODELS:
        with h5py.File(RUN / "features" / f"{model}.h5", "r") as source:
            expected = digest([rows, source.attrs["model_weight_digest"], protocol])
            assert source.attrs["complete"] and source.attrs["signature"] == expected
            assert source["written"].shape == (len(rows),) and source["written"][:].all()
            hashes = source["input_sha256"][:]
            if fingerprints is not None:
                assert np.array_equal(hashes, fingerprints)
            fingerprints = hashes
            for site in SITES:
                matrix = source[site][:]
                assert matrix.shape == (len(rows), 2048 if site.startswith("vlm/") else 1024)
                assert np.isfinite(matrix).all()
                matrices[model, site] = matrix
    return matrices


def embed(features, perplexity):
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
                metric="euclidean", max_iter=1000, random_state=SEED, method="barnes_hut", n_jobs=1)
    coordinates = tsne.fit_transform(features)
    assert np.isfinite(coordinates).all()
    return coordinates


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def legend_marker(name, color, marker):
    return Line2D([], [], label=name, color=color, linestyle="none", marker=marker,
                  markersize=7, markeredgewidth=0.4, markeredgecolor="white")


def plot_track(track, maps, labels, categories, pdf):
    fig, axes = plt.subplots(2, 4, figsize=(14, 8.2), sharex="col", sharey="col")
    fig.subplots_adjust(left=.085, right=.985, top=.81, bottom=.14, wspace=.12, hspace=.09)
    track_name = {"A": "Explicit instructions", "B": "Observation-dependent candidates"}[track]
    fig.text(.035, .95, f"Track {track}  /  {track_name}", fontsize=22, weight="bold", color="#203342")
    fig.text(.036, .905, f"Initial scene   ·   {len(labels)} inputs   ·   One point per instruction and observation",
             fontsize=11, color="#4d5e6b")
    handles = [legend_marker(f"{name}  ({(labels == label).sum()})", color, marker)
               for label, (name, color, marker) in LABEL_STYLE.items()]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.975, .976), frameon=False,
               ncol=2, handletextpad=.3)
    fig.text(.04, .65, "Request label", rotation=90, ha="center", va="center", fontsize=12, weight="bold")
    fig.text(.04, .30, "Task category", rotation=90, ha="center", va="center", fontsize=12, weight="bold")
    for column, ((model, site), coords) in enumerate(maps.items()):
        top, bottom = axes[:, column]
        top.set_title(f"{MODEL_NAMES[model]}   /   {SITES[site]}", fontsize=11, pad=9)
        for label, (_, color, marker) in LABEL_STYLE.items():
            top.scatter(*coords[labels == label].T, s=24, c=color, marker=marker,
                        edgecolors="white", linewidths=.35, alpha=.88)
        for category, (_, color) in CATEGORY_STYLE.items():
            bottom.scatter(*coords[categories == category].T, s=20, c=color,
                           edgecolors="white", linewidths=.27, alpha=.94)
        center_x, center_y = (coords.max(axis=0) + coords.min(axis=0)) / 2
        radius = np.ptp(coords, axis=0).max() * .55
        top.set_xlim(center_x - radius, center_x + radius)
        top.set_ylim(center_y - radius, center_y + radius)
        for ax in (top, bottom):
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("#fafbfd")
            for spine in ax.spines.values():
                spine.set_color("#dbe1e5")
    handles = [legend_marker(name, color, "o") for key, (name, color) in CATEGORY_STYLE.items()
               if key in categories]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .029), ncol=6,
               frameon=False, fontsize=10, handletextpad=.3, columnspacing=1.3)
    fig.savefig(OUTPUT / f"track_{track}.png", dpi=175, facecolor="white")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def main():
    manifest = json.loads((RUN / "manifest.json").read_text())
    rows = manifest["rows"]
    protocol = json.loads((RUN / "experiment_protocol.json").read_text())
    assert set(SITES).issubset(protocol["primary_sites"])
    matrices = read_features(rows, protocol)
    with (RUN / "report/all_sites.csv").open() as stream:
        probes = {(r["track"], r["model"], r["site"]): float(r["fold_mean_AUROC"])
                  for r in csv.DictReader(stream)}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    points, summaries = [], []
    with threadpool_limits(limits=1), PdfPages(OUTPUT / "contrast_initial_tsne.pdf") as pdf:
        for track in ("A", "B"):
            indices = [i for i, row in enumerate(rows) if row["track"] == track]
            items = [rows[i] for i in indices]
            labels = np.array([r["label"] for r in items])
            categories = np.array([r["category"] for r in items])
            assert set(categories).issubset(CATEGORY_STYLE)
            maps = {}
            for (model, site), matrix in matrices.items():
                features = matrix[indices]
                original_match = matching_neighbors(features, labels)
                for perplexity in (15, PRIMARY_PERPLEXITY, 50):
                    coords = embed(features, perplexity)
                    summaries.append(dict(track=track, model=model, site=site, perplexity=perplexity,
                                          n=len(items), label_match_2d=matching_neighbors(coords, labels),
                                          label_match_original=original_match, label_match_chance=random_match(labels),
                                          category_match_2d=matching_neighbors(coords, categories),
                                          category_match_chance=random_match(categories),
                                          probe_fold_mean_AUROC=probes[track, model, site]))
                    if perplexity == PRIMARY_PERPLEXITY:
                        maps[model, site] = coords
                        for index, row, (x, y) in zip(indices, items, coords, strict=True):
                            points.append(dict(track=track, model=model, site=site, manifest_index=index,
                                               item_id=row["item_id"], condition_id=row["condition_id"],
                                               category=row["category"], label=row["label"], group_id=row["group_id"],
                                               instruction=row["instruction"], x=float(x), y=float(y)))
            plot_track(track, maps, labels, categories, pdf)
    write_csv(OUTPUT / "coordinates.csv", points)
    write_csv(OUTPUT / "neighbors.csv", summaries)


if __name__ == "__main__":
    main()
