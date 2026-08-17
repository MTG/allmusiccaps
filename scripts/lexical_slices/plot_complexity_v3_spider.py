"""Spider/radar plot of MRR per v3 2x2 bucket (density x framing) for three models.

One polar axis per dataset, horizontally displayed. Each polygon is a model.
Buckets with fewer than MIN_N captions are skipped (e.g. MusicCaps thin+none has
only n=1, so it is dropped and that dataset shows a triangle on the remaining
three buckets). The spoke ordering for the full 2x2 case puts the two density
levels and the two framing levels on antipodal pairs.
"""

from __future__ import annotations

import json
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_BASE = REPO_ROOT / "downstream_results"
OUT_PATH = REPO_ROOT / "plots" / "complexity_v3_spider.pdf"

MIN_N = 5

MODELS = {
    "R01": "Baseline",
    "R04": "Baseline + Quotes",
    "R05": "Baseline + Struct",
}
DATASETS = {
    "song_describer": ("songd_complexity_v3.json", "Song Describer"),
    "music_caps": ("mucaps_complexity_v3.json", "MusicCaps"),
}

BUCKETS_FULL = [
    ("thin", "none"),
    ("thin", "present"),
    ("dense", "present"),
    ("dense", "none"),
]
BUCKET_LABELS = {
    ("thin", "none"): "thin\nnone",
    ("thin", "present"): "thin\npresent",
    ("dense", "none"): "dense\nnone",
    ("dense", "present"): "dense\npresent",
}


def load_ranks(model: str, dataset: str) -> dict[int, dict]:
    path = RESULTS_BASE / model / dataset / "caption2rank.json"
    with open(path) as f:
        return {r["index"]: r for r in json.load(f)}


def mrr_per_bucket(
    comp: dict[int, dict], ranks: dict[int, dict], buckets: list[tuple[str, str]]
) -> dict[tuple, tuple[float, int]]:
    shared = set(comp) & set(ranks)
    out: dict[tuple, tuple[float, int]] = {}
    for d, f in buckets:
        idxs = [
            i
            for i in shared
            if comp[i].get("density") == d and comp[i].get("framing") == f
        ]
        if not idxs:
            out[(d, f)] = (float("nan"), 0)
            continue
        mrr = sum(1.0 / (ranks[i]["min_rank"] + 1) for i in idxs) / len(idxs)
        out[(d, f)] = (mrr * 100, len(idxs))
    return out


def main() -> None:
    sns.set_theme(
        context="paper",
        style="white",
        font_scale=1.4,
        rc={
            "font.family": "serif",
            "axes.linewidth": 0.9,
            "legend.frameon": False,
        },
    )

    model_names = list(MODELS.values())
    colors = sns.color_palette("deep", n_colors=len(model_names))
    color_by_model = dict(zip(model_names, colors))

    # First pass: per-dataset support, drop low-support buckets, collect MRRs.
    per_ds_buckets: dict[str, list[tuple[str, str]]] = {}
    per_ds_per_model: dict[str, dict[str, dict[tuple, tuple[float, int]]]] = {}
    per_ds_rmax: dict[str, float] = {}
    for ds_key, (comp_file, ds_pretty) in DATASETS.items():
        with open(Path(__file__).parent / comp_file) as f:
            comp = {r["index"]: r for r in json.load(f)}
        # Determine which buckets clear MIN_N using the first model's ranks.
        ref_ranks = load_ranks(next(iter(MODELS)), ds_key)
        ref = mrr_per_bucket(comp, ref_ranks, BUCKETS_FULL)
        kept = [b for b in BUCKETS_FULL if ref[b][1] >= MIN_N]
        per_ds_buckets[ds_pretty] = kept

        per_ds_per_model[ds_pretty] = {}
        ds_max = 0.0
        for m, pretty in MODELS.items():
            vals = mrr_per_bucket(comp, load_ranks(m, ds_key), kept)
            per_ds_per_model[ds_pretty][pretty] = vals
            for v, _ in vals.values():
                if not np.isnan(v):
                    ds_max = max(ds_max, v)
        per_ds_rmax[ds_pretty] = (int(ds_max // 5) + 1) * 5

    fig, axes = plt.subplots(
        1, 2, figsize=(8.0, 4.4), subplot_kw={"projection": "polar"}
    )

    for ax, (_, ds_pretty) in zip(axes, DATASETS.values()):
        kept = per_ds_buckets[ds_pretty]
        n_spokes = len(kept)
        angles = [n / float(n_spokes) * 2.0 * pi for n in range(n_spokes)]
        angles_closed = angles + [angles[0]]

        # Rotate so the layout reads cleanly: with 4 spokes, put diagonals;
        # with 3 spokes, point one spoke straight up.
        ax.set_theta_offset(pi / 4.0 if n_spokes == 4 else pi / 2.0)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles)
        ax.set_xticklabels([BUCKET_LABELS[b] for b in kept])
        rmax = per_ds_rmax[ds_pretty]
        ax.set_ylim(0, rmax)
        ticks = list(range(0, int(rmax) + 1, max(1, int(rmax // 4))))
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t}" for t in ticks], fontsize=9)
        ax.tick_params(axis="x", pad=8)
        ax.grid(True, linewidth=0.6, alpha=0.5)

        for name in model_names:
            vals = per_ds_per_model[ds_pretty][name]
            radii = [vals[b][0] for b in kept]
            radii_closed = radii + [radii[0]]
            ax.plot(
                angles_closed,
                radii_closed,
                color=color_by_model[name],
                linewidth=2.0,
                clip_on=False,
            )
            ax.fill(
                angles_closed,
                radii_closed,
                color=color_by_model[name],
                alpha=0.10,
            )

        ax.set_title(ds_pretty, fontsize=13, pad=18)

    model_handles = [
        Line2D([0], [0], color=color_by_model[name], linewidth=2.4, label=name)
        for name in model_names
    ]
    fig.legend(
        handles=model_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(model_names),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        handlelength=2.4,
        handletextpad=0.6,
        columnspacing=2.0,
    )

    fig.suptitle("MRR (%) per density x framing bucket", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12, wspace=0.45)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
