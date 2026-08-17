"""Training dynamics on non-probing tasks across loss/TE configurations.

One subplot per dataset; one line per model. Three colors map to the loss
family (InfoNCE, InfoNCE+SigReg, Sigmoid); two line styles encode whether
the text encoder is trainable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]

# (model_id, pretty name, loss family, trainable text encoder)
MODELS = [
    ("R07", "InfoNCE", "InfoNCE", False),
    ("R13", "InfoNCE + TE", "InfoNCE", True),
    ("o26r43l0", "InfoNCE + SigReg", "InfoNCE+SigReg", False),
    ("R14", "InfoNCE + SigReg + TE", "InfoNCE+SigReg", True),
    ("R08", "Sigmoid", "Sigmoid", False),
]

# (subpath, json_key, scale, dataset pretty title)
DATASETS = [
    ("music_caps/caption.json", "mean_reciprocal_rank", 100, "MusicCaps"),
    ("song_describer/caption.json", "mean_reciprocal_rank", 100, "Song Describer"),
    ("gtzan_zsl/results.json", "accuracy", 100, "GTZAN"),
    ("fma_small_zsl/results.json", "accuracy", 100, "FMA-Small"),
    ("dimsim/results.json", "accuracy", 100, "DimSim"),
]

METRIC_LABEL = {
    "MusicCaps": "MRR (\\%)",
    "Song Describer": "MRR (\\%)",
    "GTZAN": "Acc.\\ (\\%)",
    "FMA-Small": "Acc.\\ (\\%)",
    "DimSim": "Acc.\\ (\\%)",
}

STEP_DIR_RE = re.compile(r"step=(\d+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-base", default=str(REPO_ROOT / "downstream_results"))
    p.add_argument(
        "--out-path",
        default=str(REPO_ROOT / "plots" / "training_dynamics_objectives.pdf"),
    )
    return p.parse_args()


def discover_steps(model_dir: str) -> list[tuple[int, str]]:
    out = []
    for entry in glob(os.path.join(model_dir, "step=*")):
        if not os.path.isdir(entry):
            continue
        m = STEP_DIR_RE.search(entry)
        if m:
            out.append((int(m.group(1)), entry))
    out.sort()
    return out


def load_metric(directory: str, subpath: str, key: str) -> float | None:
    path = os.path.join(directory, subpath)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get(key)


def main() -> None:
    args = parse_args()

    series: dict[str, dict[str, tuple[list[int], list[float]]]] = {}
    for model_id, pretty, _, _ in MODELS:
        model_dir = os.path.join(args.results_base, model_id)
        steps = discover_steps(model_dir)
        if not steps:
            print(
                f"[warn] {pretty} ({model_id}): no step=* dirs in {model_dir}, skipping"
            )
            continue
        per_dataset: dict[str, tuple[list[int], list[float]]] = {}
        for subpath, key, scale, ds_pretty in DATASETS:
            xs, ys = [], []
            for step, directory in steps:
                v = load_metric(directory, subpath, key)
                if v is not None:
                    xs.append(step)
                    ys.append(v * scale)
            per_dataset[ds_pretty] = (xs, ys)
        series[pretty] = per_dataset
        print(f"{pretty} ({model_id}): {len(steps)} checkpoints")

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

    loss_families = ["InfoNCE", "InfoNCE+SigReg", "Sigmoid"]
    palette = sns.color_palette("deep", n_colors=3)
    color_for = dict(zip(loss_families, palette))
    ls_for = {False: "-", True: "--"}
    marker_for = {False: "o", True: "s"}

    fig, axes_grid = plt.subplots(2, 3, figsize=(7.0, 4.8), sharey=False)
    plot_axes = [
        axes_grid[0, 0],
        axes_grid[0, 1],
        axes_grid[0, 2],
        axes_grid[1, 0],
        axes_grid[1, 1],
    ]
    legend_ax = axes_grid[1, 2]
    legend_ax.axis("off")

    for ax, (_, _, _, ds_pretty) in zip(plot_axes, DATASETS):
        for model_id, pretty, family, trainable in MODELS:
            if pretty not in series:
                continue
            xs, ys = series[pretty][ds_pretty]
            if not xs:
                continue
            xs_k = [x / 1000 for x in xs]
            ax.plot(
                xs_k,
                ys,
                marker=marker_for[trainable],
                linestyle=ls_for[trainable],
                linewidth=1.4,
                markersize=5.0,
                color=color_for[family],
                markerfacecolor="white",
                markeredgecolor=color_for[family],
                markeredgewidth=1.2,
                label=pretty,
                clip_on=False,
            )
        ax.set_title(ds_pretty, fontsize=13, pad=6)
        ax.set_xlabel("Steps (k)")
        ax.set_ylabel(
            METRIC_LABEL[ds_pretty]
            if plt.rcParams["text.usetex"]
            else METRIC_LABEL[ds_pretty].replace("\\%", "%").replace("\\ ", " ")
        )
        ax.margins(x=0.02)
        ax.tick_params(axis="both", which="major", length=3, pad=2)
        sns.despine(ax=ax)

    color_handles = [
        Line2D([0], [0], color=color_for[f], linewidth=1.6, label=f)
        for f in loss_families
    ]
    style_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=ls_for[False],
            marker=marker_for[False],
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.2,
            markersize=5.0,
            linewidth=1.4,
            label="Frozen TE",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=ls_for[True],
            marker=marker_for[True],
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.2,
            markersize=5.0,
            linewidth=1.4,
            label="Trainable TE",
        ),
    ]
    legend_ax.legend(
        handles=color_handles + style_handles,
        loc="center",
        ncol=1,
        frameon=False,
        handlelength=2.4,
        handletextpad=0.6,
        borderaxespad=0.0,
    )

    fig.tight_layout(w_pad=1.0, h_pad=1.2)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
