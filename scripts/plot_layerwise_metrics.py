"""Plot layerwise metrics for a trained model via the wandb API.

Each metric gets one figure showing its value per layer, with one line per
training step (n_steps evenly sampled from beginning to end of training).

Usage:
    python scripts/plot_layerwise_metrics.py <run_id> [--n-steps N] [--out-dir DIR]
                                             [--entity ENTITY] [--project PROJECT]

Example:
    python scripts/plot_layerwise_metrics.py R07
    python scripts/plot_layerwise_metrics.py R07 --n-steps 7
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import wandb

METRICS = [
    "mlid",
    "frechet_var",
    "eff_rank",
    "norm_entropy",
    "anisotropy",
    "gaussianity",
    "ii_audio_to_text",
    "ii_text_to_audio",
]

METRIC_LABELS = {
    "mlid": "mLID",
    "frechet_var": "Fréchet Variance",
    "eff_rank": "Effective Rank",
    "norm_entropy": "Normalized Entropy",
    "anisotropy": "Anisotropy",
    "gaussianity": "Gaussianity",
    "ii_audio_to_text": "Information Imbalance (audio → text)",
    "ii_text_to_audio": "Information Imbalance (text → audio)",
}

LAYER_METRIC_RE = re.compile(r"layer_metrics/(\w+)_layer_(\d+)")
II_A2T_RE = re.compile(r"info_imbalance/audio(\d+)_to_(\w+)")
II_T2A_RE = re.compile(r"info_imbalance/(\w+)_to_audio(\d+)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", help="Wandb run ID (e.g. R07)")
    parser.add_argument("--entity", default="<anon-entity>")
    parser.add_argument("--project", default="text-audio")
    parser.add_argument(
        "--n-steps", type=int, default=5, help="Number of steps to plot"
    )
    parser.add_argument("--out-dir", default="plots/layerwise_metrics")
    return parser.parse_args()


def fetch_history(run, n_steps: int) -> dict[int, dict[int, dict]]:
    """Return {step: {layer: {metric: value}}} for n_steps evenly sampled steps.

    Fetches layer_metrics/* and info_imbalance/* columns. Info imbalance values
    are averaged over text columns for each audio layer.
    """
    # Fetch all columns so both layer_metrics/* and info_imbalance/* are included
    df = run.history(samples=500, pandas=True)
    df = df.dropna(subset=["_step"])

    all_steps = sorted(df["_step"].astype(int).tolist())
    if not all_steps:
        return {}

    indices = np.linspace(
        0, len(all_steps) - 1, min(n_steps, len(all_steps)), dtype=int
    )
    selected_steps = [all_steps[i] for i in indices]

    tables: dict[int, dict[int, dict]] = {}
    for step in selected_steps:
        row = df[df["_step"] == step].iloc[0]
        by_layer: dict[int, dict] = {}

        # Geometric metrics
        for col in df.columns:
            m = LAYER_METRIC_RE.match(col)
            if not m:
                continue
            metric, layer = m.group(1), int(m.group(2))
            val = row[col]
            if not np.isnan(val):
                by_layer.setdefault(layer, {})[metric] = val

        # Information imbalance — average over text columns per audio layer
        ii_a2t: dict[int, list] = {}
        ii_t2a: dict[int, list] = {}
        for col in df.columns:
            m = II_A2T_RE.match(col)
            if m:
                layer = int(m.group(1))
                val = row[col]
                if not np.isnan(val):
                    ii_a2t.setdefault(layer, []).append(val)
                continue
            m = II_T2A_RE.match(col)
            if m:
                layer = int(m.group(2))
                val = row[col]
                if not np.isnan(val):
                    ii_t2a.setdefault(layer, []).append(val)

        for layer, vals in ii_a2t.items():
            by_layer.setdefault(layer, {})["ii_audio_to_text"] = sum(vals) / len(vals)
        for layer, vals in ii_t2a.items():
            by_layer.setdefault(layer, {})["ii_text_to_audio"] = sum(vals) / len(vals)

        if by_layer:
            tables[step] = by_layer

    return tables


def make_plots(
    tables: dict[int, dict[int, dict]], run_name: str, run_id: str, out_dir: str
):
    os.makedirs(out_dir, exist_ok=True)
    steps = sorted(tables.keys())
    colors = plt.colormaps["viridis"](np.linspace(0, 1, max(len(steps), 2)))
    all_layers = sorted(set(l for by_layer in tables.values() for l in by_layer))

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8, 4))

        for step, color in zip(steps, colors):
            by_layer = tables[step]
            layers = sorted(by_layer.keys())
            values = [by_layer[l].get(metric, float("nan")) for l in layers]
            ax.plot(
                layers,
                values,
                marker="o",
                markersize=4,
                color=color,
                label=f"step {step:,}",
            )

        ax.set_xlabel("Layer")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"{METRIC_LABELS[metric]} — {run_name}")
        ax.legend(fontsize=8, loc="best")
        tick_labels = ["proj" if l == 12 else str(l) for l in all_layers]
        ax.set_xticks(all_layers)
        ax.set_xticklabels(tick_labels)
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"{run_id}_{metric}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


def main():
    args = parse_args()
    api = wandb.Api()
    run = api.run(f"{args.entity}/{args.project}/{args.run_id}")
    print(f"Run: {run.name} ({args.run_id})")

    tables = fetch_history(run, args.n_steps)
    if not tables:
        print("No layer_metrics data found in run history.")
        return

    print(f"Plotting {len(tables)} steps: {sorted(tables.keys())}")
    make_plots(tables, run.name, args.run_id, args.out_dir)


if __name__ == "__main__":
    main()
