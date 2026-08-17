"""Aggregate CLAP-score vs. MusicEval MOS correlations across paper models.

Reads `downstream_results/<model_id>/musiceval/{per_sample.json,results.json}`
for each model, writes an aggregate CSV, and renders a scatter figure
(one subplot per model) of CLAP score vs. MOS alignment with Spearman rho
annotated per panel.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_BASE = REPO_ROOT / "downstream_results"
OUT_CSV = (
    REPO_ROOT / "downstream_results" / "_aggregates" / "clap_score_correlation.csv"
)
OUT_PDF = REPO_ROOT / "plots" / "clap_score_mos_correlation.pdf"

MODELS = {
    "R01": "Baseline",
    "R04": "Baseline + Quotes",
    "R05": "Baseline + Struct",
    "R07": "All-layers",
    "R14": "TE-trained + SigReg",
}


def load_per_sample(model_id: str) -> pd.DataFrame | None:
    path = RESULTS_BASE / model_id / "musiceval" / "per_sample.json"
    if not path.is_file():
        return None
    with open(path) as f:
        return pd.DataFrame(json.load(f))


def load_results(model_id: str) -> dict | None:
    path = RESULTS_BASE / model_id / "musiceval" / "results.json"
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def build_aggregate() -> pd.DataFrame:
    rows = []
    for mid, pretty in MODELS.items():
        res = load_results(mid)
        if res is None:
            print(f"skip {mid}: results.json missing")
            continue
        u = res["utterance_level"]
        s = res["system_level"]
        rows.append(
            {
                "model_id": mid,
                "model": pretty,
                "n": res["n_samples"],
                "mean_clap": res["mean_clap"],
                "ci95_lo": res["ci95_lo"],
                "ci95_hi": res["ci95_hi"],
                "utt_spearman_overall": u["spearman_overall"],
                "utt_spearman_alignment": u["spearman_alignment"],
                "utt_kendall_alignment": u["kendall_alignment"],
                "utt_pearson_alignment": u["pearson_alignment"],
                "sys_spearman_overall": s["spearman_overall"],
                "sys_spearman_alignment": s["spearman_alignment"],
                "sys_pearson_alignment": s["pearson_alignment"],
            }
        )
    return pd.DataFrame(rows)


def make_scatter(df_all: dict[str, pd.DataFrame]) -> None:
    sns.set_theme(
        context="paper",
        style="ticks",
        font_scale=1.3,
        rc={
            "font.family": "serif",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": ":",
            "axes.linewidth": 0.9,
            "legend.frameon": False,
        },
    )
    models = [m for m in MODELS if m in df_all]
    n = len(models)
    if n == 0:
        print("no per-sample data; skipping scatter")
        return
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.1 * ncols, 2.8 * nrows), sharey=True
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, mid in zip(axes, models):
        df = df_all[mid]
        rho = stats.spearmanr(df["clap_score"], df["mos_alignment"]).statistic
        ax.scatter(
            df["mos_alignment"],
            df["clap_score"],
            s=8,
            alpha=0.35,
            color="black",
            edgecolors="none",
        )
        ax.set_title(f"{MODELS[mid]}\n$\\rho_s={rho:+.2f}$", fontsize=11, pad=4)
        ax.set_xlabel("Alignment MOS")
        ax.set_ylabel("CLAP score")
        sns.despine(ax=ax)
    for ax in axes[len(models) :]:
        ax.axis("off")

    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"wrote {OUT_PDF}")


def main() -> None:
    agg = build_aggregate()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"wrote {OUT_CSV}")
    print(agg.to_string(index=False))

    per_sample = {}
    for mid in MODELS:
        df = load_per_sample(mid)
        if df is not None:
            per_sample[mid] = df
    make_scatter(per_sample)


if __name__ == "__main__":
    main()
