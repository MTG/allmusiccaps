"""Line plot of MRR vs complexity score for three models on both retrieval datasets."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_BASE = REPO_ROOT / "downstream_results"
OUT_PATH = REPO_ROOT / "plots" / "complexity_split_mrr_line.pdf"

MODELS = {
    "R01": "Baseline",
    "R04": "Baseline + Quotes",
    "R05": "Baseline + Struct",
}
DATASETS = {
    "song_describer": ("songd_complexity.json", "Song Describer"),
    "music_caps": ("mucaps_complexity.json", "MusicCaps"),
}
SCORES = (2, 3, 4)
SCORE_LABELS = {1: "1", 2: "2\n(simple)", 3: "3", 4: "4\n(complex)", 5: "5"}


def load_ranks(model: str, dataset: str) -> dict[int, dict]:
    path = RESULTS_BASE / model / dataset / "caption2rank.json"
    with open(path) as f:
        return {r["index"]: r for r in json.load(f)}


def build_frame() -> pd.DataFrame:
    rows = []
    for ds_key, (comp_file, ds_pretty) in DATASETS.items():
        with open(Path(__file__).parent / comp_file) as f:
            comp = {r["index"]: r for r in json.load(f)}
        per_model = {m: load_ranks(m, ds_key) for m in MODELS}
        shared = set.intersection(set(comp), *(set(r) for r in per_model.values()))
        for m, pretty in MODELS.items():
            ranks = per_model[m]
            for s in SCORES:
                idxs = [i for i in shared if comp[i].get("score") == s]
                if not idxs:
                    continue
                mrr = sum(1.0 / (ranks[i]["min_rank"] + 1) for i in idxs) / len(idxs)
                rows.append(
                    {
                        "dataset": ds_pretty,
                        "score": s,
                        "model": pretty,
                        "mrr": mrr * 100,
                        "n": len(idxs),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    df = build_frame()
    sns.set_theme(
        context="paper",
        style="ticks",
        font_scale=1.6,
        rc={
            "font.family": "serif",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": ":",
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "legend.frameon": False,
        },
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=False)
    model_names = list(MODELS.values())
    markers = ["o", "s", "D", "^", "v", "P", "X"][: len(model_names)]
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))][
        : len(model_names)
    ]
    colors = sns.color_palette("deep", n_colors=len(model_names))

    for i, (ax, (_, ds_pretty)) in enumerate(zip(axes, DATASETS.values())):
        sub = pd.DataFrame(df[df["dataset"] == ds_pretty])
        for name, marker, ls, color in zip(model_names, markers, linestyles, colors):
            msub = sub[sub["model"] == name].sort_values(by="score")
            ax.plot(
                msub["score"],
                msub["mrr"],
                marker=marker,
                linestyle=ls,
                linewidth=1.4,
                markersize=5.5,
                color=color,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.3,
                label=name,
                clip_on=False,
            )
        ax.set_title(ds_pretty, fontsize=13, pad=6)
        ax.set_xlabel("")
        if i == 0:
            ax.set_ylabel("MRR (\\%)" if plt.rcParams["text.usetex"] else "MRR (%)")
        else:
            ax.set_ylabel("")
        ax.set_xticks(list(SCORES))
        ax.set_xticklabels([SCORE_LABELS[s] for s in SCORES])
        ax.margins(x=0.02)
        ax.tick_params(axis="both", which="major", length=3, pad=2)
        if ax.get_legend():
            ax.legend_.remove()
        sns.despine(ax=ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=len(MODELS),
        frameon=False,
        handlelength=2.4,
        columnspacing=1.6,
        handletextpad=0.6,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.0)
    fig.subplots_adjust(left=0.08, right=0.995, wspace=0.18)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
