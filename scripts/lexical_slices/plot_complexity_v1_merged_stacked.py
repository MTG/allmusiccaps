"""Stacked line plot of MRR vs v1 complexity, with levels 1+2 and 4+5 merged.

Top: Song Describer, bottom: MusicCaps. Colors encode the model.
The v1 5-level scale is collapsed to three effective bins:
  bin 1 = scores {1, 2}  (tag-expressible)
  bin 2 = score 3        (mixed)
  bin 3 = scores {4, 5}  (narrative / non-tag)
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
OUT_PATH = REPO_ROOT / "plots" / "complexity_v1_merged_split_mrr_stacked.pdf"

MODELS = {
    "R01": "Baseline",
    "R04": "Baseline + Quotes",
    "R05": "Baseline + Struct",
}
DATASETS = {
    "song_describer": ("songd_complexity.json", "Song Describer"),
    "music_caps": ("mucaps_complexity.json", "MusicCaps"),
}

BINS = [
    (1, {1, 2}, "1-2\n(simple)"),
    (2, {3}, "3"),
    (3, {4, 5}, "4-5\n(complex)"),
]


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
            for bin_idx, score_set, _label in BINS:
                idxs = [i for i in shared if comp[i].get("score") in score_set]
                if not idxs:
                    continue
                mrr = sum(1.0 / (ranks[i]["min_rank"] + 1) for i in idxs) / len(idxs)
                rows.append(
                    {
                        "dataset": ds_pretty,
                        "bin": bin_idx,
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

    fig, axes = plt.subplots(2, 1, figsize=(5.5, 4.05), sharex=True)
    model_names = list(MODELS.values())
    colors = sns.color_palette("deep", n_colors=len(model_names))
    color_by_model = dict(zip(model_names, colors))

    for ax, (_ds_key, (_cf, ds_pretty)) in zip(axes, DATASETS.items()):
        sub = df[df["dataset"] == ds_pretty]
        for name in model_names:
            msub = sub[sub["model"] == name].sort_values(by="bin")
            ax.plot(
                msub["bin"],
                msub["mrr"],
                linestyle="-",
                linewidth=2.4,
                color=color_by_model[name],
                clip_on=False,
            )
        ax.set_title(ds_pretty, fontsize=13, pad=4)
        ax.set_ylabel("MRR (\\%)" if plt.rcParams["text.usetex"] else "MRR (%)")
        ax.margins(x=0.02)
        ax.tick_params(axis="both", which="major", length=3, pad=2)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", visible=True)
        sns.despine(ax=ax)

    axes[-1].set_xlabel("Caption complexity (v1, merged)", labelpad=2)
    axes[-1].set_xticks([b[0] for b in BINS])
    axes[-1].set_xticklabels([b[2] for b in BINS])

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
    fig.subplots_adjust(left=0.12, right=0.995, hspace=0.35)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
