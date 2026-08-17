"""Line plot of MRR vs complexity score for three models, datasets stacked vertically.

Top: Song Describer, bottom: MusicCaps. Colors encode the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_BASE = REPO_ROOT / "downstream_results"
OUT_PATH = REPO_ROOT / "plots" / "complexity_split_mrr_stacked.pdf"

MODELS = {
    "R01": "Baseline",
    "R04": "Baseline + Quotes",
    "R05": "Baseline + Struct",
}
DATASETS = {
    "song_describer": ("songd_complexity.json", "Song Describer"),
    "music_caps": ("mucaps_complexity.json", "MusicCaps"),
}
SCORES = (1, 2, 3, 4)


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
        font_scale=1.936,
        rc={
            "font.family": "serif",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "legend.frameon": False,
        },
    )

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.566))
    model_names = list(MODELS.values())
    colors = sns.color_palette("deep", n_colors=len(model_names))
    color_by_model = dict(zip(model_names, colors))
    markers = ["o", "s", "^", "D", "v", "P", "X"][: len(model_names)]
    marker_by_model = dict(zip(model_names, markers))

    for ax_idx, (ax, (_ds_key, (_cf, ds_pretty))) in enumerate(
        zip(axes, DATASETS.items())
    ):
        sub = df[df["dataset"] == ds_pretty]
        # support per score, taken from the first model's row (n is the same across models)
        n_by_score: dict[int, int] = {}
        for s in SCORES:
            row = sub[sub["score"] == s]
            n_by_score[s] = int(row["n"].iloc[0]) if len(row) else 0
        scores_present = [s for s in SCORES if n_by_score[s] > 0]
        for name in model_names:
            msub = sub[sub["model"] == name].sort_values(by="score")
            ax.plot(
                msub["score"],
                msub["mrr"],
                linestyle="-",
                linewidth=2.0,
                marker=marker_by_model[name],
                markersize=7,
                markerfacecolor=color_by_model[name],
                markeredgewidth=0.8,
                markeredgecolor=color_by_model[name],
                color=color_by_model[name],
                clip_on=False,
            )
        ax.set_title(ds_pretty, fontsize=15.73, pad=4)
        ax.set_ylabel("MRR (\\%)" if plt.rcParams["text.usetex"] else "MRR (%)")
        ax.set_xlabel(
            "Caption complexity" if ax_idx == len(axes) - 1 else "", labelpad=2
        )
        ax.set_xticks(scores_present)
        ax.set_xticklabels([f"{s}\n({n_by_score[s]})" for s in scores_present])
        ax.margins(x=0.04)
        ax.tick_params(axis="both", which="major", length=3, pad=2)
        ax.grid(visible=True, which="major", linewidth=0.6, alpha=0.4)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.9)

    model_handles = [
        Line2D(
            [0],
            [0],
            color=color_by_model[name],
            linewidth=2.4,
            linestyle="-",
            marker=marker_by_model[name],
            markersize=7,
            markerfacecolor=color_by_model[name],
            markeredgecolor=color_by_model[name],
            markeredgewidth=0.8,
            label=name,
        )
        for name in model_names
    ]
    legend_model = axes[1].legend(
        handles=model_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        handlelength=2.4,
        handletextpad=0.6,
    )
    legend_model.get_frame().set_linewidth(0.7)

    fig.tight_layout()
    fig.subplots_adjust(left=0.10, right=0.995, hspace=0.55)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
