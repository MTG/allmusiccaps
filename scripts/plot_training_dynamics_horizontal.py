"""Training dynamics on non-probing and probing tasks (horizontal, 5x2).

5 non-probing tasks (MusicCaps, SongDescriber, GTZAN, FMA-Small, DimSim) on the
left column and 5 probing tasks (MTT, J.Genre, J.Instr., J.Mood, MGPHot) on the
right column. Probing checkpoints may be sparse: we print availability per
(model, task) and plot only what exists.
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

# (subpath, json_key, scale, dataset pretty title, higher_is_better)
NON_PROBING = [
    ("music_caps/caption.json", "mean_reciprocal_rank", 100, "MusicCaps", True),
    (
        "song_describer/caption.json",
        "mean_reciprocal_rank",
        100,
        "Song Describer",
        True,
    ),
    ("gtzan_zsl/results.json", "accuracy", 100, "GTZAN", True),
    ("fma_small_zsl/results.json", "accuracy", 100, "FMA-Small", True),
    ("dimsim/results.json", "accuracy", 100, "DimSim", True),
]

PROBING = [
    ("mtt_autotagging/results.json", "test-MAP-macro", 100, "MTT", True),
    ("jamendo_genre/results.json", "test-MAP-macro", 100, "J.Genre", True),
    ("jamendo_instrument/results.json", "test-MAP-macro", 100, "J.Instr.", True),
    ("jamendo_moodtheme/results.json", "test-MAP-macro", 100, "J.Mood", True),
    ("mgphot_regression/results.json", "test-RMSE-macro", 1, "MGPHot", False),
]

DATASETS = NON_PROBING + PROBING

# Baselines (raw values, same scale as JSON outputs). From generate_full_table.py.
# Keyed by dataset pretty title.
BASELINES: dict[str, dict[str, float]] = {
    "Laion-CLAP": {
        "MusicCaps": 0.1018,
        "Song Describer": 0.1703,
        "GTZAN": 0.7150,
        "FMA-Small": 0.5549,
        "DimSim": 0.8055,
        "MTT": 0.462,
        "J.Genre": 0.183,
        "J.Instr.": 0.184,
        "J.Mood": 0.146,
        "MGPHot": 0.165,
    },
    "TTMR++": {
        "MusicCaps": 0.1229,
        "Song Describer": 0.1616,
        "GTZAN": 0.8167,
        "FMA-Small": 0.4333,
        "DimSim": 0.6974,
        "MTT": 0.465,
        "J.Genre": 0.200,
        "J.Instr.": 0.193,
        "J.Mood": 0.147,
        "MGPHot": 0.169,
    },
    "CLaMP 3": {
        "MusicCaps": 0.0569,
        "Song Describer": 0.1653,
        "GTZAN": 0.505,
        "FMA-Small": 0.3592,
        "DimSim": 0.7179,
        "MTT": 0.463,
        "J.Genre": 0.192,
        "J.Instr.": 0.204,
        "J.Mood": 0.146,
        "MGPHot": 0.175,
    },
}

# Per-task baseline override (use this baseline regardless of best). Otherwise
# pick the best baseline for the task respecting higher_is_better.
BASELINE_OVERRIDE: dict[str, str] = {
    "MusicCaps": "CLaMP 3",
}

# Per-task baseline blacklist (exclude these baselines from the per-task
# best-baseline selection, e.g. due to evaluation set contamination).
BASELINE_BLACKLIST: dict[str, set[str]] = {
    "MusicCaps": {"Laion-CLAP", "TTMR++"},
    "FMA-Small": {"Laion-CLAP"},
}

# Shared dashed style; baselines are distinguished by a [n] tag in legend/plot.
BASELINE_LINESTYLE = ":"
BASELINE_COLOR = "#b8002e"  # crimson, distinct from model palette
BASELINE_TAG = {
    "Laion-CLAP": "[1]",
    "TTMR++": "[2]",
    "CLaMP 3": "[3]",
}

METRIC_LABEL = {
    "MusicCaps": "MRR",
    "Song Describer": "MRR",
    "GTZAN": "Acc.",
    "FMA-Small": "Acc.",
    "DimSim": "Acc.",
    "MTT": "MAP",
    "J.Genre": "MAP",
    "J.Instr.": "MAP",
    "J.Mood": "MAP",
    "MGPHot": "RMSE",
}

STEP_DIR_RE = re.compile(r"step=(\d+)$")

FINAL_STEP_FALLBACK = 149796


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-base", default=str(REPO_ROOT / "downstream_results"))
    p.add_argument(
        "--out-path",
        default=str(REPO_ROOT / "plots" / "training_dynamics_horizontal.pdf"),
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
    print("\nAvailability summary (steps per task):")
    for model_id, pretty, _, _ in MODELS:
        model_dir = os.path.join(args.results_base, model_id)
        steps = discover_steps(model_dir)
        per_dataset: dict[str, tuple[list[int], list[float]]] = {}
        task_summary = []
        for subpath, key, scale, ds_pretty, _ in DATASETS:
            xs, ys = [], []
            for step, directory in steps:
                v = load_metric(directory, subpath, key)
                if v is not None:
                    xs.append(step)
                    ys.append(v * scale)
            v_root = load_metric(model_dir, subpath, key)
            if v_root is not None and FINAL_STEP_FALLBACK not in xs:
                xs.append(FINAL_STEP_FALLBACK)
                ys.append(v_root * scale)
                order = sorted(range(len(xs)), key=lambda i: xs[i])
                xs = [xs[i] for i in order]
                ys = [ys[i] for i in order]
            per_dataset[ds_pretty] = (xs, ys)
            task_summary.append(f"{ds_pretty}={len(xs)}")
        if not any(xs for xs, _ in per_dataset.values()):
            print(f"  {pretty} ({model_id}): no probing results, skipping")
            continue
        series[pretty] = per_dataset
        print(f"  {pretty} ({model_id}): " + ", ".join(task_summary))

    sns.set_theme(
        context="paper",
        style="ticks",
        font_scale=1.85,
        rc={
            "font.family": "serif",
            "axes.grid": True,
            "grid.alpha": 0.45,
            "grid.linestyle": ":",
            "grid.linewidth": 1,
            "axes.linewidth": 1,
            "xtick.major.width": 1.9,
            "ytick.major.width": 1.9,
            "legend.frameon": False,
        },
    )

    loss_families = ["InfoNCE", "InfoNCE+SigReg", "Sigmoid"]
    palette = sns.color_palette("deep", n_colors=len(loss_families))
    color_for = dict(zip(loss_families, palette))
    ls_for = {False: "-", True: "--"}
    marker_for = {False: "o", True: "s"}

    n_cols = 5
    fig, axes_grid = plt.subplots(2, n_cols, figsize=(15.0, 4.83), sharey=False)
    # Top row: non-probing tasks. Bottom row: probing tasks.
    plot_axes = []
    ordered_datasets = []
    for c in range(n_cols):
        plot_axes.append(axes_grid[0, c])
        ordered_datasets.append(NON_PROBING[c])
    for c in range(n_cols):
        plot_axes.append(axes_grid[1, c])
        ordered_datasets.append(PROBING[c])

    used_baselines: set[str] = set()
    for ax, (_, _, _, ds_pretty, higher_is_better) in zip(plot_axes, ordered_datasets):
        # Pick baseline for this task.
        blacklist = BASELINE_BLACKLIST.get(ds_pretty, set())
        if ds_pretty in BASELINE_OVERRIDE:
            chosen = BASELINE_OVERRIDE[ds_pretty]
        else:
            candidates = [
                (name, vals[ds_pretty])
                for name, vals in BASELINES.items()
                if ds_pretty in vals and name not in blacklist
            ]
            chosen = (
                max(candidates, key=lambda kv: kv[1])[0]
                if higher_is_better
                else min(candidates, key=lambda kv: kv[1])[0]
            )
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
                linewidth=1.8,
                markersize=4.0,
                color=color_for[family],
                markerfacecolor=color_for[family],
                markeredgecolor=color_for[family],
                markeredgewidth=2.2,
                label=pretty,
                clip_on=True,
            )
        # Find scale for this dataset (same scale as model series).
        scale = next(s for _, _, s, ds, _ in DATASETS if ds == ds_pretty)
        baseline_y = BASELINES[chosen][ds_pretty] * scale
        ax.axhline(
            y=baseline_y,
            color=BASELINE_COLOR,
            linewidth=1.8,
            linestyle=BASELINE_LINESTYLE,
            alpha=0.9,
            zorder=0.5,
        )
        tag = BASELINE_TAG[chosen]
        if BASELINE_OVERRIDE.get(ds_pretty) == chosen or blacklist:
            tag = f"{tag}*"
        # Per-task vertical nudge for baseline tag to avoid overlap with curves,
        # expressed as a fraction of the axis y-range.
        tag_dy_frac = {"J.Mood": 0.15}.get(ds_pretty, 0.0)
        y_lo, y_hi = ax.get_ylim()
        tag_y = baseline_y + tag_dy_frac * (y_hi - y_lo)
        ax.annotate(
            tag,
            xy=(1.0, tag_y),
            xycoords=("axes fraction", "data"),
            xytext=(2, 0),
            textcoords="offset points",
            color=BASELINE_COLOR,
            fontsize=12,
            ha="left",
            va="center",
            annotation_clip=False,
        )
        used_baselines.add(chosen)
        arrow = "$\\uparrow$" if higher_is_better else "$\\downarrow$"
        ax.set_title(
            f"{ds_pretty} ({METRIC_LABEL[ds_pretty]}{arrow})", fontsize=15, pad=4
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.margins(x=0.02)
        # Per-task y-axis lower bound to keep visualization readable when an
        # outlier model would otherwise compress the visible range.
        y_lower_bound = {
            "MusicCaps": 5.0,
            "Song Describer": 15.0,
            "GTZAN": 77.0,
            "FMA-Small": 41.0,
        }.get(ds_pretty)
        if y_lower_bound is not None:
            _, y_hi = ax.get_ylim()
            ax.set_ylim(bottom=y_lower_bound, top=y_hi)
        ax.tick_params(axis="both", which="major", length=3, pad=2)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontsize(lbl.get_fontsize() * 0.9)
        for spine in ax.spines.values():
            spine.set_visible(True)

    color_handles = [
        Line2D([0], [0], color=color_for[f], linewidth=2.6, label=f)
        for f in loss_families
    ]
    style_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=ls_for[False],
            marker=marker_for[False],
            markerfacecolor="black",
            markeredgecolor="black",
            markeredgewidth=2.2,
            markersize=4.0,
            linewidth=1.8,
            label="Frozen TE",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=ls_for[True],
            marker=marker_for[True],
            markerfacecolor="black",
            markeredgecolor="black",
            markeredgewidth=2.2,
            markersize=4.0,
            linewidth=1.8,
            label="Trainable TE",
        ),
    ]
    baseline_handles = [
        Line2D(
            [0],
            [0],
            color=BASELINE_COLOR,
            linestyle=BASELINE_LINESTYLE,
            linewidth=1.8,
            label=f"{BASELINE_TAG[name]} {name}",
        )
        for name in BASELINES
        if name in used_baselines
    ]

    # Two-row legend in a single framed box.
    # Row 1: loss families (our models) + Frozen TE + Trainable TE.
    # Row 2: baselines.
    top_row = color_handles + style_handles
    bottom_row = baseline_handles
    ncol = max(len(top_row), len(bottom_row))
    blank = Line2D([0], [0], color="none", label=" ")
    pad_top = ncol - len(top_row)
    left_pad_top = pad_top // 2
    right_pad_top = pad_top - left_pad_top
    row1 = [blank] * left_pad_top + top_row + [blank] * right_pad_top
    pad = ncol - len(bottom_row)
    left_pad = pad // 2
    right_pad = pad - left_pad
    row2 = [blank] * left_pad + bottom_row + [blank] * right_pad
    # matplotlib fills the legend column-major, so interleave per column.
    legend_handles = []
    for c in range(ncol):
        legend_handles.append(row1[c])
        legend_handles.append(row2[c])
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=ncol,
        frameon=True,
        fancybox=False,
        edgecolor="0.4",
        handlelength=2.4,
        columnspacing=1.6,
        handletextpad=0.5,
        borderpad=0.6,
    )

    fig.tight_layout(rect=(0, 0.12, 1, 1.0), w_pad=0.05, h_pad=0.05)
    fig.subplots_adjust(wspace=0.32, hspace=0.55)
    fig.supxlabel("Steps (k)", fontsize=15, y=0.11)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
