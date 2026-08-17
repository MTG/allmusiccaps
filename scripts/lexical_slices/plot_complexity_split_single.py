"""Line plot of MRR vs complexity score for three models, both datasets on one axis.

Solid lines = Song Describer, dotted lines = MusicCaps. Colors encode the model.
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
OUT_PATH = REPO_ROOT / "plots" / "complexity_split_mrr_single.pdf"

MODELS = {
    "R01": "Baseline",
    "R04": "Baseline + Quotes",
    "R05": "Baseline + Struct",
}
DATASETS = {
    "song_describer": ("songd_complexity.json", "Song Describer", "-"),
    "music_caps": ("mucaps_complexity.json", "MusicCaps", "--"),
}
SCORES = (2, 3, 4)
SCORE_LABELS = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}


def load_ranks(model: str, dataset: str) -> dict[int, dict]:
    path = RESULTS_BASE / model / dataset / "caption2rank.json"
    with open(path) as f:
        return {r["index"]: r for r in json.load(f)}


def build_frame() -> pd.DataFrame:
    rows = []
    for ds_key, (comp_file, ds_pretty, _ls) in DATASETS.items():
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
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "legend.frameon": False,
        },
    )

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.6))
    model_names = list(MODELS.values())
    colors = sns.color_palette("deep", n_colors=len(model_names))
    color_by_model = dict(zip(model_names, colors))

    for _ds_key, (_cf, ds_pretty, ls) in DATASETS.items():
        sub = df[df["dataset"] == ds_pretty]
        for name in model_names:
            msub = sub[sub["model"] == name].sort_values(by="score")
            ax.plot(
                msub["score"],
                msub["mrr"],
                linestyle=ls,
                linewidth=2.4,
                color=color_by_model[name],
                clip_on=False,
            )

    ax.set_xlabel("Caption complexity", labelpad=2)
    ax.set_ylabel("MRR (\\%)" if plt.rcParams["text.usetex"] else "MRR (%)")
    ax.set_xticks(list(SCORES))
    ax.set_xticklabels([SCORE_LABELS[s] for s in SCORES])
    ax.margins(x=0.02)
    ax.tick_params(axis="both", which="major", length=3, pad=2)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", visible=True)
    sns.despine(ax=ax)

    model_handles = [
        Line2D(
            [0],
            [0],
            color=color_by_model[name],
            linewidth=2.4,
            linestyle="-",
            label=name,
        )
        for name in model_names
    ]
    dataset_handles = [
        Line2D([0], [0], color="black", linewidth=2.4, linestyle=ls, label=ds_pretty)
        for _, ds_pretty, ls in DATASETS.values()
    ]

    legend_model = ax.legend(
        handles=model_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        handlelength=2.4,
        handletextpad=0.6,
    )
    legend_model.get_frame().set_linewidth(0.7)
    ax.add_artist(legend_model)
    legend_ds = ax.legend(
        handles=dataset_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.0),
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        handlelength=2.4,
        handletextpad=0.6,
    )
    legend_ds.get_frame().set_linewidth(0.7)

    fig.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.995)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
