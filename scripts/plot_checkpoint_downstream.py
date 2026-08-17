"""Plot downstream performance vs checkpoint step for multiple models.

Usage:
    python scripts/plot_checkpoint_downstream.py \
        --results-base /scratch/<group>/downstream_results \
        --models R07 R06 R05 \
        --out-dir plots/checkpoint_downstream
"""

import argparse
import json
import os
import re
from glob import glob

import matplotlib.pyplot as plt

# Mapping from model ID to descriptive name (from generate_full_table.py)
MODEL_NAMES = {
    "R02": "InfoNCE quotes",
    "R03": "InfoNCE struct",
    "R04": "InfoNCE quotes+mu+so",
    "R05": "InfoNCE struct+mu+so",
    "R06": "L6 quotes+mu+so",
    "R07": "LA quotes+mu+so",
    "ny6g2bzr": "LA TT quotes+mu+so",
    "R13": "LA TT quotes+mu+so (fixed)",
    "R14": "LA TT+SigReg quotes+mu+so",
    "R08": "sigmoid quotes+mu+so",
    "z0r7jh5a": "LeJEPA quotes+mu+so",
    "2cvx96fi": "SLAP quotes+mu+so",
    "f9l9z22m": "LpJEPA quotes+mu+so",
    "8jlpuoge": "LpJEPA J quotes+mu+so",
    "mgl3nnr3": "InfoNCE short+att",
    "fze0cez1": "LeJEPA 2 views",
    "7edqcl5n": "LeJEPA 4 views",
}

# (label, subpath, json_key, scale, higher_is_better)
METRICS = [
    ("MuCaps MRR", "music_caps/caption.json", "mean_reciprocal_rank", 100, True),
    ("SongD. MRR", "song_describer/caption.json", "mean_reciprocal_rank", 100, True),
    ("GTZAN Acc.", "gtzan_zsl/results.json", "accuracy", 100, True),
    ("FMA-S Acc.", "fma_small_zsl/results.json", "accuracy", 100, True),
    ("DimSim Acc.", "dimsim/results.json", "accuracy", 100, True),
]

STEP_DIR_RE = re.compile(r"step=(\d+)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot downstream metrics vs checkpoint step for multiple models."
    )
    parser.add_argument(
        "--results-base",
        required=True,
        help="Base directory containing per-model result subdirectories",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model IDs to plot (e.g. R07 R06 R05)",
    )
    parser.add_argument(
        "--out-dir",
        default="plots/checkpoint_downstream",
        help="Directory to save output plots",
    )
    return parser.parse_args()


def discover_steps(results_dir):
    """Return sorted list of (step, directory_path) tuples."""
    steps = []
    for entry in glob(os.path.join(results_dir, "step=*")):
        if os.path.isdir(entry):
            m = STEP_DIR_RE.search(entry)
            if m:
                steps.append((int(m.group(1)), entry))
    steps.sort()
    return steps


def load_metric(directory, subpath, key):
    """Load a single metric value from a JSON file."""
    path = os.path.join(directory, subpath)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get(key)


def main():
    args = parse_args()

    # Collect data for all models
    model_data = {}
    for model_id in args.models:
        results_dir = os.path.join(args.results_base, model_id)
        steps = discover_steps(results_dir)
        if not steps:
            print(f"No step=* directories found for {model_id} in {results_dir}")
            continue
        name = MODEL_NAMES.get(model_id, model_id)
        print(f"{name} ({model_id}): {len(steps)} checkpoints")
        model_data[model_id] = (name, steps)

    if not model_data:
        print("No data found for any model.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
        }
    )

    for label, subpath, key, scale, higher_is_better in METRICS:
        fig, ax = plt.subplots(figsize=(8, 4))
        has_data = False

        for model_id, (name, steps) in model_data.items():
            xs, ys = [], []
            for step, directory in steps:
                val = load_metric(directory, subpath, key)
                if val is not None:
                    xs.append(step)
                    ys.append(val * scale)

            if not xs:
                print(f"  {label} / {name}: no data, skipping")
                continue

            has_data = True
            ax.plot(xs, ys, marker="o", label=name)

            # Mark best checkpoint
            best_idx = (
                max(range(len(ys)), key=lambda i: ys[i])
                if higher_is_better
                else min(range(len(ys)), key=lambda i: ys[i])
            )
            ax.annotate(
                f"{ys[best_idx]:.1f}",
                xy=(xs[best_idx], ys[best_idx]),
                xytext=(5, 8),
                textcoords="offset points",
                fontsize=8,
            )

        if not has_data:
            plt.close(fig)
            print(f"  {label}: no data for any model, skipping")
            continue

        ax.set_xlabel("Training Step")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=9)
        fig.tight_layout()

        fname = label.lower().replace(" ", "_").replace(".", "") + ".png"
        out_path = os.path.join(args.out_dir, fname)
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
