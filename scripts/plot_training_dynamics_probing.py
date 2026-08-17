"""Training dynamics on non-probing and probing tasks (combined, 3x3).

5 non-probing tasks (MusicCaps, SongDescriber, GTZAN, FMA-Small, DimSim) and
4 probing tasks (MTT, J.Genre, J.Instr., J.Mood) on a 3-row x 3-col grid.
Probing checkpoints may be sparse: we print availability per (model, task)
and plot only what exists.
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
DATASETS = [
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
    ("mtt_autotagging/results.json", "test-MAP-macro", 100, "MTT", True),
    ("jamendo_genre/results.json", "test-MAP-macro", 100, "J.Genre", True),
    ("jamendo_instrument/results.json", "test-MAP-macro", 100, "J.Instr.", True),
    ("jamendo_moodtheme/results.json", "test-MAP-macro", 100, "J.Mood", True),
]

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
}

STEP_DIR_RE = re.compile(r"step=(\d+)$")

FINAL_STEP_FALLBACK = 149796


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-base", default=str(REPO_ROOT / "downstream_results"))
    p.add_argument(
        "--out-path",
        default=str(REPO_ROOT / "plots" / "training_dynamics_combined.pdf"),
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

    fig, axes_grid = plt.subplots(3, 3, figsize=(7.0, 7.2), sharey=False)
    plot_axes = [axes_grid[r, c] for r in range(3) for c in range(3)]

    for ax, (_, _, _, ds_pretty, _) in zip(plot_axes, DATASETS):
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
        ax.set_title(f"{ds_pretty} ({METRIC_LABEL[ds_pretty]})", fontsize=13, pad=6)
        ax.set_xlabel("")
        ax.set_ylabel("")
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
    fig.legend(
        handles=color_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(color_handles),
        frameon=False,
        handlelength=2.4,
        columnspacing=1.6,
        handletextpad=0.5,
    )
    fig.legend(
        handles=style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=len(style_handles),
        frameon=False,
        handlelength=2.4,
        columnspacing=1.6,
        handletextpad=0.5,
    )

    fig.tight_layout(rect=(0, 0.03, 1, 0.94), w_pad=0.2, h_pad=0.3)
    fig.supxlabel("Steps (k)", fontsize=13, y=0.0)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
