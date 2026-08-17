"""Plot geometrical properties alongside downstream MRR to check trend alignment.

Produces a single grid figure:
    rows = models, columns = geometrical metrics.
    Each cell overlays the geometrical metric (left y-axis) with
    SongDescriber MRR and MusicCaps MRR (right y-axis, dual axis).

Usage:
    python scripts/plot_tracking.py --results-base downstream_results

    python scripts/plot_tracking.py \
        --runs R07:"LA" z0r7jh5a:"LeJEPA" \
        --results-base downstream_results
"""

import argparse
import json
import os
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb

# -- Constants ----------------------------------------------------------------

GEOM_METRICS = [
    "mlid",
    "frechet_var",
    "eff_rank",
    "norm_entropy",
    "anisotropy",
    "gaussianity",
    "ii_audio_to_text",
    "ii_text_to_audio",
]

GEOM_METRIC_LABELS = {
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

PROJ_LAYER = 12

STEP_DIR_RE = re.compile(r"step=(\d+)$")

DOWNSTREAM_METRICS = [
    ("SongD. MRR", "song_describer/caption.json", "mean_reciprocal_rank", 100),
    ("MuCaps MRR", "music_caps/caption.json", "mean_reciprocal_rank", 100),
]

MODEL_NAMES = {
    "R02": "InfoNCE quotes",
    "R03": "InfoNCE struct",
    "R04": "InfoNCE quotes+mu+so",
    "R05": "InfoNCE struct+mu+so",
    "R06": "L6 quotes+mu+so",
    "R07": "LA quotes+mu+so",
    "ny6g2bzr": "LA TT quotes+mu+so",
    "R08": "sigmoid quotes+mu+so",
    "z0r7jh5a": "LeJEPA quotes+mu+so",
    # "2cvx96fi": "SLAP quotes+mu+so",
    "f9l9z22m": "LpJEPA quotes+mu+so",
    "8jlpuoge": "LpJEPA J quotes+mu+so",
    "mgl3nnr3": "InfoNCE short+att",
    "fze0cez1": "LeJEPA 2 views",
    "7edqcl5n": "LeJEPA 4 views",
}

# -- Helpers (adapted from plot_geom_properties.py) ---------------------------


def _parse_layer_data_from_row(row, columns) -> dict[int, dict]:
    """Extract per-layer metric dict from a single W&B history row."""
    by_layer: dict[int, dict] = {}

    for col in columns:
        m = LAYER_METRIC_RE.match(col)
        if m:
            metric, layer = m.group(1), int(m.group(2))
            val = row.get(col)
            if val is not None and np.isfinite(val):
                by_layer.setdefault(layer, {})[metric] = val
            continue

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


# -- Data loading -------------------------------------------------------------


def fetch_geom_data(
    api: wandb.Api, entity: str, project: str, run_id: str
) -> dict[int, dict[str, float]] | None:
    """Fetch projector-layer geometrical metrics over training from W&B.

    Returns {step: {metric: value}} or None on failure.
    """
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
    except (wandb.errors.CommError, ValueError):
        return None

    df = run.history(samples=5000, pandas=True)
    df = df.dropna(subset=["_step"])
    if df.empty:
        return None

    columns = df.columns.tolist()
    layer_cols = [c for c in columns if LAYER_METRIC_RE.match(c)]
    if not layer_cols:
        return None

    metric_mask = df[layer_cols].notna().any(axis=1)
    metric_df = df[metric_mask]
    if metric_df.empty:
        return None

    metric_steps = sorted(metric_df["_step"].astype(int).unique().tolist())

    proj_history: dict[int, dict] = {}
    for step in metric_steps:
        row = metric_df[metric_df["_step"] == step].iloc[0]
        by_layer = _parse_layer_data_from_row(row, columns)
        if PROJ_LAYER in by_layer:
            proj_history[step] = by_layer[PROJ_LAYER]

    return proj_history if proj_history else None


def discover_steps(results_dir: str) -> list[tuple[int, str]]:
    """Return sorted list of (step, directory_path) tuples."""
    steps = []
    for entry in glob(os.path.join(results_dir, "step=*")):
        if os.path.isdir(entry):
            m = STEP_DIR_RE.search(entry)
            if m:
                steps.append((int(m.group(1)), entry))
    steps.sort()
    return steps


def load_metric(directory: str, subpath: str, key: str):
    """Load a single metric value from a JSON file."""
    path = os.path.join(directory, subpath)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get(key)


def load_downstream_data(
    results_base: str, model_id: str
) -> dict[int, dict[str, float]] | None:
    """Load downstream MRR values per checkpoint step.

    Returns {step: {"SongD. MRR": val, "MuCaps MRR": val}} or None.
    """
    results_dir = os.path.join(results_base, model_id)
    steps = discover_steps(results_dir)
    if not steps:
        return None

    data: dict[int, dict[str, float]] = {}
    for step, directory in steps:
        row = {}
        for label, subpath, key, scale in DOWNSTREAM_METRICS:
            val = load_metric(directory, subpath, key)
            if val is not None:
                row[label] = val * scale
        if row:
            data[step] = row

    return data if data else None


# -- CLI ----------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot geometrical properties vs downstream MRR tracking grid."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        metavar='ID:"LABEL"',
        help=(
            "W&B run specs (run_id:label). "
            "If omitted, auto-discovers models from MODEL_NAMES with downstream results."
        ),
    )
    parser.add_argument(
        "--results-base",
        default="downstream_results",
        help="Base directory for downstream result files",
    )
    parser.add_argument("--out-dir", default="plots/tracking")
    parser.add_argument("--entity", default="<anon-entity>")
    parser.add_argument("--project", default="text-audio")
    return parser.parse_args()


def parse_run_spec(spec: str) -> tuple[str, str]:
    """Parse 'run_id:label' or 'run_id:"label with spaces"'."""
    if ":" in spec:
        run_id, label = spec.split(":", 1)
        return run_id, label.strip("\"'")
    return spec, spec


# -- Plotting -----------------------------------------------------------------


def _format_step(x, _pos):
    """Tick formatter: 50000 → '50k'."""
    return f"{int(x / 1000)}k" if x >= 1000 else str(int(x))


def plot_tracking_grid(
    geom_data: dict[str, dict[int, dict]],
    downstream_data: dict[str, dict[int, dict]],
    ds_label: str,
    ds_key: str,
    ds_color,
    ds_marker: str,
    out_dir: str,
):
    """Create and save a tracking grid for one downstream metric.

    Rows = geom metrics, columns = models.
    Each cell overlays one geometrical metric (left y-axis) with the
    downstream metric (right y-axis). Geom data is clipped to start from
    the first downstream checkpoint step.
    """
    sns.set_theme(style="whitegrid", font_scale=0.9)

    model_labels = list(geom_data.keys())
    n_models = len(model_labels)
    n_rows = len(GEOM_METRICS)

    # Pre-pass: compute global y-range per geom metric and for downstream MRR
    geom_ranges: dict[str, tuple[float, float]] = {}
    all_ds_vals: list[float] = []
    for metric in GEOM_METRICS:
        all_vals = []
        for label in model_labels:
            geom = geom_data[label]
            ds = downstream_data[label]
            ds_steps = sorted(ds.keys())
            ds_xs = [s for s in ds_steps if ds_key in ds[s]]
            min_ds_step = ds_xs[0] if ds_xs else float("inf")
            geom_steps = sorted(geom.keys())
            vals = [
                geom[s][metric]
                for s in geom_steps
                if metric in geom[s] and s >= min_ds_step
            ]
            all_vals.extend(vals)
        if all_vals:
            margin = (max(all_vals) - min(all_vals)) * 0.05
            geom_ranges[metric] = (min(all_vals) - margin, max(all_vals) + margin)

    for label in model_labels:
        ds = downstream_data[label]
        for step_data in ds.values():
            if ds_key in step_data:
                all_ds_vals.append(step_data[ds_key])

    if all_ds_vals:
        ds_margin = (max(all_ds_vals) - min(all_ds_vals)) * 0.05
        ds_range = (min(all_ds_vals) - ds_margin, max(all_ds_vals) + ds_margin)
    else:
        ds_range = None

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_models,
        figsize=(3.5 * n_models, 2.8 * n_rows),
        squeeze=False,
    )

    geom_color = sns.color_palette("tab10")[0]  # blue

    for col_idx, label in enumerate(model_labels):
        geom = geom_data[label]
        ds = downstream_data[label]

        # Downstream x/y for this metric
        ds_steps = sorted(ds.keys())
        ds_xs = [s for s in ds_steps if ds_key in ds[s]]
        ds_ys = [ds[s][ds_key] for s in ds_xs]

        # Earliest downstream step — used to clip geom data
        min_ds_step = ds_xs[0] if ds_xs else float("inf")

        for row_idx, metric in enumerate(GEOM_METRICS):
            ax = axes[row_idx, col_idx]

            # Left y-axis: geometrical metric (clipped to >= first ds step)
            geom_steps = sorted(geom.keys())
            xs = [s for s in geom_steps if metric in geom[s] and s >= min_ds_step]
            ys = [geom[s][metric] for s in xs]

            if xs:
                ax.plot(
                    xs,
                    ys,
                    color=geom_color,
                    linewidth=1.5,
                    marker="o",
                    markersize=3,
                )
                ax.tick_params(axis="y", labelcolor=geom_color)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="gray",
                )

            # Apply shared y-range for this geom metric across all models
            if metric in geom_ranges:
                ax.set_ylim(geom_ranges[metric])

            # Right y-axis: downstream metric
            ax2 = ax.twinx()
            if ds_xs:
                ax2.plot(
                    ds_xs,
                    ds_ys,
                    color=ds_color,
                    linewidth=1.5,
                    marker=ds_marker,
                    markersize=3,
                    linestyle="--",
                )
            ax2.tick_params(axis="y", labelcolor=ds_color)

            # Apply shared y-range for downstream MRR
            if ds_range is not None:
                ax2.set_ylim(ds_range)

            # Column title (first row only) — model name
            if row_idx == 0:
                ax.set_title(label, fontsize=10, fontweight="bold")

            # Row label (first column only) — metric name
            if col_idx == 0:
                ax.set_ylabel(GEOM_METRIC_LABELS[metric], fontsize=9, fontweight="bold")

            # X-axis: show tick labels only on bottom row
            ax.xaxis.set_major_formatter(plt.FuncFormatter(_format_step))
            if row_idx < n_rows - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Step", fontsize=8)
                ax.tick_params(axis="x", rotation=45)

            # Hide right y-axis labels for inner columns
            if col_idx < n_models - 1:
                ax2.set_yticklabels([])

    # Figure-level legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            color=geom_color,
            linewidth=1.5,
            marker="o",
            markersize=4,
            label="Geom. metric",
        ),
        plt.Line2D(
            [0],
            [0],
            color=ds_color,
            linewidth=1.5,
            marker=ds_marker,
            markersize=4,
            linestyle="--",
            label=ds_label,
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=2,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.98])

    os.makedirs(out_dir, exist_ok=True)
    fname = ds_label.lower().replace(" ", "_").replace(".", "") + ".png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# -- Main ---------------------------------------------------------------------


def main():
    args = parse_args()
    api = wandb.Api()

    # Determine which models to process
    if args.runs:
        run_specs = [parse_run_spec(s) for s in args.runs]
    else:
        # Auto-discover: MODEL_NAMES keys that have downstream results on disk
        run_specs = []
        for model_id, name in MODEL_NAMES.items():
            model_dir = os.path.join(args.results_base, model_id)
            if os.path.isdir(model_dir):
                run_specs.append((model_id, name))
        print(
            f"Auto-discovered {len(run_specs)} models with downstream results: "
            f"{[s[0] for s in run_specs]}"
        )

    geom_all: dict[str, dict[int, dict]] = {}
    ds_all: dict[str, dict[int, dict]] = {}

    for run_id, label in run_specs:
        print(f"Fetching {run_id} ({label}) ...")

        geom = fetch_geom_data(api, args.entity, args.project, run_id)
        if geom is None:
            print(f"  WARNING: no W&B geom data for {run_id}, skipping")
            continue

        ds = load_downstream_data(args.results_base, run_id)
        if ds is None:
            print(f"  WARNING: no downstream data for {run_id}, skipping")
            continue

        geom_all[label] = geom
        ds_all[label] = ds
        print(f"  {len(geom)} geom steps, {len(ds)} downstream steps")

    if not geom_all:
        print("No models with both data sources found.")
        return

    palette = sns.color_palette("tab10")
    ds_styles = [
        ("SongD. MRR", palette[3], "s"),  # red
        ("MuCaps MRR", palette[1], "^"),  # orange
    ]

    for ds_label, ds_color, ds_marker in ds_styles:
        print(f"\n--- {ds_label} ---")
        plot_tracking_grid(
            geom_all,
            ds_all,
            ds_label=ds_label,
            ds_key=ds_label,
            ds_color=ds_color,
            ds_marker=ds_marker,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
