"""Grouped bar plot of MRR per register label for multiple models on both retrieval datasets."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_BASE = REPO_ROOT / "downstream_results"
OUT_PATH = REPO_ROOT / "plots" / "register_split_mrr.pdf"

MODELS = {
    "R01": "Baseline",
    "R02": "Quotes only",
    "R03": "Struct only",
    "R04": "Baseline + Quotes",
    "R05": "Baseline + Struct",
}
DATASETS = {
    "song_describer": ("songd_register.json", "Song Describer"),
    "music_caps": ("mucaps_register.json", "MusicCaps"),
}
LABELS = ("objective", "figurative", "hard")
LABEL_PRETTY = {
    "objective": "Objective",
    "figurative": "Figurative",
    "hard": "Hard",
}


def load_ranks(model: str, dataset: str) -> dict[int, dict]:
    path = RESULTS_BASE / model / dataset / "caption2rank.json"
    with open(path) as f:
        return {r["index"]: r for r in json.load(f)}


def build_frame() -> pd.DataFrame:
    rows = []
    for ds_key, (reg_file, ds_pretty) in DATASETS.items():
        with open(Path(__file__).parent / reg_file) as f:
            reg = {r["index"]: r for r in json.load(f)}
        per_model = {m: load_ranks(m, ds_key) for m in MODELS}
        shared = set.intersection(set(reg), *(set(r) for r in per_model.values()))
        for m, pretty in MODELS.items():
            ranks = per_model[m]
            for L in LABELS:
                idxs = [i for i in shared if reg[i]["label"] == L]
                if not idxs:
                    continue
                mrr = sum(1.0 / (ranks[i]["min_rank"] + 1) for i in idxs) / len(idxs)
                rows.append(
                    {
                        "dataset": ds_pretty,
                        "label": LABEL_PRETTY[L],
                        "model": pretty,
                        "mrr": mrr * 100,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    df = build_frame()
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    palette = sns.color_palette("deep", n_colors=len(MODELS))

    for ax, (_, ds_pretty) in zip(axes, DATASETS.values()):
        sub = pd.DataFrame(df[df["dataset"] == ds_pretty])
        sns.barplot(
            data=sub,
            x="label",
            y="mrr",
            hue="model",
            order=[LABEL_PRETTY[L] for L in LABELS],
            hue_order=list(MODELS.values()),
            palette=palette,
            ax=ax,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_title(ds_pretty)
        ax.set_xlabel("")
        ax.set_ylabel("MRR (%)")
        if ax.get_legend():
            ax.legend_.remove()
        sns.despine(ax=ax)

    fig.legend(
        list(axes[0].containers),
        list(MODELS.values()),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=len(MODELS),
        frameon=False,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
