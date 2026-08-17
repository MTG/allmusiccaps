"""Compare layerwise metrics across multiple runs at end of training.

Each metric gets one figure with one line per run (at its last logged step).
Legend labels are the wandb run names. Data is fetched via the wandb API
from the run summary (which holds the last logged value for each metric).

Usage:
    python scripts/plot_layerwise_comparison.py <run_id> [<run_id> ...] \
        [--entity ENTITY] [--project PROJECT] [--out-dir DIR]

Example:
    python scripts/plot_layerwise_comparison.py R07 R06 5grt2e1a
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
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
# info_imbalance/audio{layer}_to_{text_col} and info_imbalance/{text_col}_to_audio{layer}
II_A2T_RE = re.compile(r"info_imbalance/audio(\d+)_to_(\w+)")
II_T2A_RE = re.compile(r"info_imbalance/(\w+)_to_audio(\d+)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_ids", nargs="+", help="Wandb run IDs to compare")
    parser.add_argument("--entity", default="<anon-entity>")
    parser.add_argument("--project", default="text-audio")
    parser.add_argument("--out-dir", default="plots/layerwise_comparison")
    return parser.parse_args()


def fetch_run_data(api: wandb.Api, entity: str, project: str, run_id: str):
    """Return (run_name, rows) where rows is [{layer, mlid, ..., ii_audio_to_text, ...}, ...]."""
    run = api.run(f"{entity}/{project}/{run_id}")

    by_layer: dict[int, dict] = {}

    # Parse layer_metrics/* keys
    for key, value in run.summary.items():
        m = LAYER_METRIC_RE.match(key)
        if m:
            metric, layer = m.group(1), int(m.group(2))
            by_layer.setdefault(layer, {"layer": layer})[metric] = value

    if not by_layer:
        return run.name, None

    # Parse info_imbalance/* keys and average over text columns per audio layer
    ii_a2t: dict[int, list] = {}
    ii_t2a: dict[int, list] = {}
    for key, value in run.summary.items():
        m = II_A2T_RE.match(key)
        if m:
            layer = int(m.group(1))
            ii_a2t.setdefault(layer, []).append(value)
            continue
        m = II_T2A_RE.match(key)
        if m:
            layer = int(m.group(2))
            ii_t2a.setdefault(layer, []).append(value)

    for layer, vals in ii_a2t.items():
        by_layer.setdefault(layer, {"layer": layer})["ii_audio_to_text"] = sum(
            vals
        ) / len(vals)
    for layer, vals in ii_t2a.items():
        by_layer.setdefault(layer, {"layer": layer})["ii_text_to_audio"] = sum(
            vals
        ) / len(vals)

    rows = [by_layer[l] for l in sorted(by_layer)]
    return run.name, rows


def make_comparison_plots(run_data: dict[str, list[dict]], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    all_layers = max(
        (list(r["layer"] for r in rows) for rows in run_data.values()), key=len
    )

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8, 4))

        for i, (label, rows) in enumerate(run_data.items()):
            layers = [r["layer"] for r in rows]
            values = [r.get(metric, float("nan")) for r in rows]
            ax.plot(
                layers,
                values,
                marker="o",
                markersize=5,
                color=colors[i % len(colors)],
                label=label,
            )

        ax.set_xlabel("Layer")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(METRIC_LABELS[metric])
        ax.legend(fontsize=8, loc="best")
        tick_labels = ["proj" if l == 12 else str(l) for l in all_layers]
        ax.set_xticks(all_layers)
        ax.set_xticklabels(tick_labels)
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"comparison_{metric}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


def main():
    args = parse_args()
    api = wandb.Api()

    run_data: dict[str, list[dict]] = {}
    skipped = []

    for run_id in args.run_ids:
        print(f"Fetching {run_id} ...")
        name, rows = fetch_run_data(api, args.entity, args.project, run_id)
        if rows is None:
            print(
                f"  WARNING: no layer_metrics in summary for {run_id} ({name}) — skipping"
            )
            skipped.append((run_id, name))
            continue
        print(f"  {name}: {len(rows)} layers")
        run_data[name] = rows

    if skipped:
        print(f"\nSkipped (no data): {[f'{r} ({n})' for r, n in skipped]}")

    if not run_data:
        print("No data to plot.")
        return

    make_comparison_plots(run_data, args.out_dir)


if __name__ == "__main__":
    main()
