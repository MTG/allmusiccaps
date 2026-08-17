"""Generate two families of geometrical-property plots across multiple runs.

Family 1 — Layer-wise comparison at end of training:
    x = layer index, y = metric value. One line per model at its last logged step.

Family 2 — Training dynamics at the audio projector layer:
    x = training step, y = metric value. One line per model.

Usage:
    python scripts/plot_geom_properties.py \
        --runs R04:"InfoNCE (AO, L11)" R07:"InfoNCE (AO, LA)" \
        [--entity ENTITY] [--project PROJECT] [--out-dir DIR]
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
    "ii_audio_to_text": "II (audio → text)",
    "ii_text_to_audio": "II (text → audio)",
}

LAYER_METRIC_RE = re.compile(r"layer_metrics/(\w+)_layer_(\d+)")
II_A2T_RE = re.compile(r"info_imbalance/audio(\d+)_to_(\w+)")
II_T2A_RE = re.compile(r"info_imbalance/(\w+)_to_audio(\d+)")

PROJ_LAYER = 12  # proj_a output is stored as layer 12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot geometrical properties across multiple runs."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        metavar='ID:"LABEL"',
        help='Run ID and label pairs, e.g. R07:"InfoNCE (AO, LA)"',
    )
    parser.add_argument("--entity", default="<anon-entity>")
    parser.add_argument("--project", default="text-audio")
    parser.add_argument("--out-dir", default="plots/geom_properties")
    parser.add_argument(
        "--proj-layer",
        type=int,
        default=PROJ_LAYER,
        help="Layer index used as the audio projector layer (default: 12)",
    )
    return parser.parse_args()


def parse_run_spec(spec: str) -> tuple[str, str]:
    """Parse 'run_id:label' or 'run_id:"label with spaces"'."""
    if ":" in spec:
        run_id, label = spec.split(":", 1)
        label = label.strip("\"'")
        return run_id, label
    return spec, spec


def _parse_layer_data_from_row(row, columns) -> dict[int, dict]:
    """Extract per-layer metric dict from a single history row."""
    by_layer: dict[int, dict] = {}

    for col in columns:
        m = LAYER_METRIC_RE.match(col)
        if m:
            metric, layer = m.group(1), int(m.group(2))
            val = row.get(col)
            if val is not None and np.isfinite(val):
                by_layer.setdefault(layer, {})[metric] = val
            continue

    # Info imbalance — average over text columns per audio layer
    ii_a2t: dict[int, list] = {}
    ii_t2a: dict[int, list] = {}
    for col in columns:
        m_a = II_A2T_RE.match(col)
        if m_a:
            layer = int(m_a.group(1))
            val = row.get(col)
            if val is not None and np.isfinite(val):
                ii_a2t.setdefault(layer, []).append(val)
            continue
        m_t = II_T2A_RE.match(col)
        if m_t:
            layer = int(m_t.group(2))
            val = row.get(col)
            if val is not None and np.isfinite(val):
                ii_t2a.setdefault(layer, []).append(val)

    for layer, vals in ii_a2t.items():
        by_layer.setdefault(layer, {})["ii_audio_to_text"] = sum(vals) / len(vals)
    for layer, vals in ii_t2a.items():
        by_layer.setdefault(layer, {})["ii_text_to_audio"] = sum(vals) / len(vals)

    return by_layer


def fetch_run_data(api: wandb.Api, entity: str, project: str, run_id: str):
    """Return (last_step_layers, history_for_proj_layer).

    last_step_layers: {layer: {metric: value}} at the last logged step.
    proj_history:     {step: {metric: value}} for the projector layer across training.
    """
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
    except (wandb.errors.CommError, ValueError):
        return None, None

    df = run.history(samples=1500, pandas=True)
    df = df.dropna(subset=["_step"])

    if df.empty:
        return None, None

    columns = df.columns.tolist()

    # Find steps that actually have layer_metrics data (not every row does)
    layer_cols = [c for c in columns if LAYER_METRIC_RE.match(c)]
    if not layer_cols:
        return None, None

    # Keep only rows that have at least one non-NaN layer metric
    metric_mask = df[layer_cols].notna().any(axis=1)
    metric_df = df[metric_mask]

    if metric_df.empty:
        return None, None

    metric_steps = sorted(metric_df["_step"].astype(int).unique().tolist())

    # Last step — layer-wise data
    last_step = metric_steps[-1]
    last_row = metric_df[metric_df["_step"] == last_step].iloc[0]
    last_step_layers = _parse_layer_data_from_row(last_row, columns)

    # All steps — projector layer only
    proj_history: dict[int, dict] = {}
    for step in metric_steps:
        row = metric_df[metric_df["_step"] == step].iloc[0]
        by_layer = _parse_layer_data_from_row(row, columns)
        if PROJ_LAYER in by_layer:
            proj_history[step] = by_layer[PROJ_LAYER]

    return last_step_layers, proj_history


# ---- Plotting ----


def _setup_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
        }
    )


def plot_layerwise_comparison(
    run_data: dict[str, dict[int, dict]],
    out_dir: str,
):
    """Family 1: one figure per metric, x = layer, one line per model."""
    os.makedirs(out_dir, exist_ok=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    all_layers = sorted(
        set(l for layers in run_data.values() for l in layers if layers)
    )

    for metric in METRICS:
        # Skip metrics that no run has
        if not any(
            metric in layers.get(l, {})
            for layers in run_data.values()
            for l in all_layers
        ):
            continue

        fig, ax = plt.subplots(figsize=(8, 4))

        for i, (label, layers) in enumerate(run_data.items()):
            xs = [l for l in all_layers if metric in layers.get(l, {})]
            ys = [layers[l][metric] for l in xs]
            ax.plot(
                xs,
                ys,
                marker="o",
                color=colors[i % len(colors)],
                label=label,
            )

        ax.set_xlabel("Layer")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"{METRIC_LABELS[metric]} (end of training)")
        ax.legend(loc="best")
        tick_labels = ["proj" if l == PROJ_LAYER else str(l) for l in all_layers]
        ax.set_xticks(all_layers)
        ax.set_xticklabels(tick_labels)
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"layerwise_{metric}.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  Saved {out_path}")


def plot_training_dynamics(
    run_data: dict[str, dict[int, dict]],
    out_dir: str,
):
    """Family 2: one figure per metric, x = step, one line per model at proj layer."""
    os.makedirs(out_dir, exist_ok=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for metric in METRICS:
        # Skip metrics that no run has
        if not any(
            metric in step_data
            for history in run_data.values()
            for step_data in history.values()
        ):
            continue

        fig, ax = plt.subplots(figsize=(8, 4))

        for i, (label, history) in enumerate(run_data.items()):
            steps = sorted(history.keys())
            xs = [s for s in steps if metric in history[s]]
            ys = [history[s][metric] for s in xs]
            ax.plot(
                xs,
                ys,
                marker="o",
                color=colors[i % len(colors)],
                label=label,
            )

        ax.set_xlabel("Training Step")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"{METRIC_LABELS[metric]} (projector layer)")
        ax.legend(loc="best")
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"dynamics_{metric}.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  Saved {out_path}")


def main():
    args = parse_args()
    _setup_style()
    api = wandb.Api()

    layerwise_data: dict[str, dict[int, dict]] = {}
    dynamics_data: dict[str, dict[int, dict]] = {}
    skipped = []

    for spec in args.runs:
        run_id, label = parse_run_spec(spec)
        print(f"Fetching {run_id} ({label}) ...")
        last_layers, proj_history = fetch_run_data(
            api, args.entity, args.project, run_id
        )
        if last_layers is None:
            print(f"  WARNING: no data for {run_id} — skipping")
            skipped.append(run_id)
            continue
        n_layers = len(last_layers)
        n_steps = len(proj_history) if proj_history else 0
        print(f"  {n_layers} layers at last step, {n_steps} steps for projector")
        layerwise_data[label] = last_layers
        if proj_history:
            dynamics_data[label] = proj_history

    if skipped:
        print(f"\nSkipped (no data): {skipped}")

    if not layerwise_data:
        print("No data to plot.")
        return

    print("\n--- Family 1: Layer-wise comparison ---")
    plot_layerwise_comparison(
        layerwise_data,
        os.path.join(args.out_dir, "layerwise"),
    )

    if dynamics_data:
        print("\n--- Family 2: Training dynamics (projector layer) ---")
        plot_training_dynamics(
            dynamics_data,
            os.path.join(args.out_dir, "dynamics"),
        )


if __name__ == "__main__":
    main()
