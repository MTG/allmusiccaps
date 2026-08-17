"""Correlation heatmaps: downstream performance vs geometrical metrics.

Generates Pearson and Spearman correlation heatmaps between checkpoint-wise
downstream task performance and geometrical metrics (from WandB) for each model.

Usage:
    python scripts/plot_correlation_heatmap.py \
        --results-dir downstream_results \
        [--entity ENTITY] [--project PROJECT] [--out-dir DIR]
"""

import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import wandb
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, spearmanr

# ---- Models ----

MODELS = [
    ("R08", "Sigmoid"),
    ("z0r7jh5a", "LeJEPA"),
    # ("2cvx96fi", "SLAP"),
    ("f9l9z22m", "LpJPEA"),
    ("8jlpuoge", "LpJPEA J"),
]

# ---- Downstream tasks: (label, subpath, json_key, scale) ----

DOWNSTREAM_TASKS = [
    ("MuCaps MRR", "music_caps/caption.json", "mean_reciprocal_rank", 100),
    ("SongD. MRR", "song_describer/caption.json", "mean_reciprocal_rank", 100),
    ("GTZAN Acc", "gtzan_zsl/results.json", "accuracy", 100),
    ("FMA-S Acc", "fma_small_zsl/results.json", "accuracy", 100),
    ("DimSim Acc", "dimsim/results.json", "accuracy", 100),
    # ("MTT MAP", "mtt_autotagging/results.json", "test-MAP-macro", 100),
    # ("J.Genre MAP", "jamendo_genre/results.json", "test-MAP-macro", 100),
    # ("J.Instr MAP", "jamendo_instrument/results.json", "test-MAP-macro", 100),
    # ("J.Mood MAP", "jamendo_moodtheme/results.json", "test-MAP-macro", 100),
    # ("MGPHot RMSE", "mgphot_regression/results.json", "test-RMSE-macro", 1),
]

# ---- Geometrical metrics (projector layer 12 only) ----

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

GEOM_LABELS = {
    "mlid": "mLID",
    "frechet_var": "Fréchet Var.",
    "eff_rank": "Eff. Rank",
    "norm_entropy": "Norm. Entropy",
    "anisotropy": "Anisotropy",
    "gaussianity": "Gaussianity",
    "ii_audio_to_text": "II (A→T)",
    "ii_text_to_audio": "II (T→A)",
}

# ---- Regex patterns (from plot_geom_properties.py) ----

STEP_DIR_RE = re.compile(r"step=(\d+)$")
LAYER_METRIC_RE = re.compile(r"layer_metrics/(\w+)_layer_(\d+)")
II_A2T_RE = re.compile(r"info_imbalance/audio(\d+)_to_(\w+)")
II_T2A_RE = re.compile(r"info_imbalance/(\w+)_to_audio(\d+)")

PROJ_LAYER = 12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Correlation heatmaps: downstream vs geometrical metrics."
    )
    parser.add_argument(
        "--results-dir",
        default="downstream_results",
        help="Local downstream results directory",
    )
    parser.add_argument("--entity", default="<anon-entity>")
    parser.add_argument("--project", default="text-audio")
    parser.add_argument("--out-dir", default="plots/correlation_heatmap")
    parser.add_argument(
        "--min-points",
        type=int,
        default=3,
        help="Min common steps for valid correlation (default: 3)",
    )
    return parser.parse_args()


# ---- Downstream data loading (from plot_checkpoint_downstream.py) ----


def discover_steps(results_dir):
    """Return sorted list of (step, directory_path) tuples."""
    from glob import glob

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


def load_downstream_data(results_dir, run_id):
    """Return {step: {task_label: value}} for a model."""
    model_dir = os.path.join(results_dir, run_id)
    steps = discover_steps(model_dir)
    if not steps:
        print(f"  No step=* directories found in {model_dir}")
        return {}

    data = {}
    for step, directory in steps:
        step_data = {}
        for label, subpath, key, scale in DOWNSTREAM_TASKS:
            val = load_metric(directory, subpath, key)
            if val is not None:
                step_data[label] = val * scale
        if step_data:
            data[step] = step_data
    return data


# ---- WandB geometrical metrics (from plot_geom_properties.py) ----


def _parse_layer_data_from_row(row, columns):
    """Extract per-layer metric dict from a single history row."""
    by_layer = {}

    for col in columns:
        m = LAYER_METRIC_RE.match(col)
        if m:
            metric, layer = m.group(1), int(m.group(2))
            val = row.get(col)
            if val is not None and np.isfinite(val):
                by_layer.setdefault(layer, {})[metric] = val
            continue

    # Info imbalance — average over text columns per audio layer
    ii_a2t = {}
    ii_t2a = {}
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


def fetch_geom_data(api, entity, project, run_id):
    """Return {step: {metric: value}} for projector layer across training."""
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
    except (wandb.errors.CommError, ValueError):
        print(f"  WARNING: could not fetch WandB run {run_id}")
        return {}

    df = run.history(samples=1500, pandas=True)
    df = df.dropna(subset=["_step"])

    if df.empty:
        return {}

    columns = df.columns.tolist()

    layer_cols = [c for c in columns if LAYER_METRIC_RE.match(c)]
    if not layer_cols:
        return {}

    metric_mask = df[layer_cols].notna().any(axis=1)
    metric_df = df[metric_mask]

    if metric_df.empty:
        return {}

    metric_steps = sorted(metric_df["_step"].astype(int).unique().tolist())

    proj_history = {}
    for step in metric_steps:
        row = metric_df[metric_df["_step"] == step].iloc[0]
        by_layer = _parse_layer_data_from_row(row, columns)
        if PROJ_LAYER in by_layer:
            proj_history[step] = by_layer[PROJ_LAYER]

    return proj_history


# ---- Correlation computation ----


def _interpolate_to_common_steps(downstream, geom):
    """Interpolate geom metrics onto downstream steps using linear interpolation.

    Downstream steps define the evaluation grid. Geom metric values are
    interpolated at those steps (only within the geom step range — no
    extrapolation). Returns list of common steps and interpolated geom dict.
    """
    down_steps = sorted(downstream.keys())
    geom_steps = sorted(geom.keys())

    if len(geom_steps) < 2 or not down_steps:
        return [], {}

    geom_min, geom_max = geom_steps[0], geom_steps[-1]

    # Keep only downstream steps within the geom range
    eval_steps = [s for s in down_steps if geom_min <= s <= geom_max]
    if not eval_steps:
        return [], {}

    # Build interpolators per geom metric
    geom_interp = {}
    for gm in GEOM_METRICS:
        xs, ys = [], []
        for s in geom_steps:
            val = geom[s].get(gm)
            if val is not None:
                xs.append(s)
                ys.append(val)
        if len(xs) >= 2:
            geom_interp[gm] = interp1d(xs, ys, kind="linear", fill_value="extrapolate")

    # Evaluate at downstream steps
    interp_geom = {}
    for s in eval_steps:
        interp_geom[s] = {gm: float(fn(s)) for gm, fn in geom_interp.items()}

    return eval_steps, interp_geom


def compute_correlations(downstream, geom, min_points):
    """Compute correlation matrix between geom metrics and downstream tasks.

    Geom metrics are linearly interpolated onto downstream steps.

    Returns (pearson_matrix, spearman_matrix) as 2D numpy arrays
    with shape (len(GEOM_METRICS), len(DOWNSTREAM_TASKS)).
    """
    n_geom = len(GEOM_METRICS)
    n_down = len(DOWNSTREAM_TASKS)
    pearson = np.full((n_geom, n_down), np.nan)
    spearman = np.full((n_geom, n_down), np.nan)

    eval_steps, interp_geom = _interpolate_to_common_steps(downstream, geom)

    if len(eval_steps) < min_points:
        print(f"    Only {len(eval_steps)} common steps (need {min_points}), skipping")
        return pearson, spearman

    print(f"    {len(eval_steps)} eval steps (geom interpolated): {eval_steps}")

    for gi, gm in enumerate(GEOM_METRICS):
        for di, (dl, *_) in enumerate(DOWNSTREAM_TASKS):
            g_vals = []
            d_vals = []
            for s in eval_steps:
                gv = interp_geom[s].get(gm)
                dv = downstream[s].get(dl)
                if gv is not None and dv is not None:
                    g_vals.append(gv)
                    d_vals.append(dv)

            if len(g_vals) < min_points:
                continue

            # Check for constant arrays
            if np.std(g_vals) == 0 or np.std(d_vals) == 0:
                continue

            pearson[gi, di] = pearsonr(g_vals, d_vals)[0]
            spearman[gi, di] = spearmanr(g_vals, d_vals)[0]

    return pearson, spearman


# ---- Plotting ----


def plot_heatmap(ax, matrix, title, annotate=True):
    """Plot a single correlation heatmap on the given axes."""
    task_labels = [t[0] for t in DOWNSTREAM_TASKS]
    geom_labels = [GEOM_LABELS[m] for m in GEOM_METRICS]

    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(task_labels)))
    ax.set_xticklabels(task_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(geom_labels)))
    ax.set_yticklabels(geom_labels, fontsize=8)
    ax.set_title(title, fontsize=10)

    if annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if np.isnan(val):
                    text = "—"
                    color = "gray"
                else:
                    text = f"{val:.2f}"
                    color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=6, color=color)

    return im


def plot_per_model(all_matrices, model_labels, corr_type, out_dir):
    """Plot 1×N subplot grid, one heatmap per model."""
    n = len(model_labels)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(4.5 * n + 1, 5),
        gridspec_kw={"right": 0.92},
    )
    if n == 1:
        axes = [axes]

    im = None
    for ax, label, matrix in zip(axes, model_labels, all_matrices):
        im = plot_heatmap(ax, matrix, label)

    fig.suptitle(f"{corr_type} Correlation: Downstream vs Geometrical Metrics", y=1.02)

    # Place colorbar in its own axes to avoid overlap
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=f"{corr_type} r")
    fig.subplots_adjust(wspace=0.4)

    fname = f"correlation_{corr_type.lower()}.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_averaged(all_matrices, corr_type, out_dir):
    """Plot single heatmap averaging correlations across models."""
    stacked = np.stack(all_matrices)
    avg = np.nanmean(stacked, axis=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = plot_heatmap(
        ax, avg, f"Average {corr_type} Correlation (N={len(all_matrices)} models)"
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label=f"{corr_type} r")
    fig.tight_layout()

    fname = f"correlation_{corr_type.lower()}_avg.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---- Main ----


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    api = wandb.Api()

    pearson_matrices = []
    spearman_matrices = []
    model_labels = []

    for run_id, label in MODELS:
        print(f"\n--- {label} ({run_id}) ---")

        # Load downstream results
        downstream = load_downstream_data(args.results_dir, run_id)
        if not downstream:
            print(f"  No downstream data, skipping")
            continue
        print(f"  Downstream: {len(downstream)} steps")

        # Fetch geometrical metrics from WandB
        print(f"  Fetching WandB data...")
        geom = fetch_geom_data(api, args.entity, args.project, run_id)
        if not geom:
            print(f"  No geom data, skipping")
            continue
        print(f"  Geom: {len(geom)} steps")

        # Compute correlations
        pearson, spearman = compute_correlations(downstream, geom, args.min_points)
        pearson_matrices.append(pearson)
        spearman_matrices.append(spearman)
        model_labels.append(label)

    if not model_labels:
        print("\nNo models with both downstream and geom data. Nothing to plot.")
        return

    print(f"\n--- Plotting ({len(model_labels)} models) ---")

    # Per-model plots
    plot_per_model(pearson_matrices, model_labels, "Pearson", args.out_dir)
    plot_per_model(spearman_matrices, model_labels, "Spearman", args.out_dir)

    # Averaged plots
    plot_averaged(pearson_matrices, "Pearson", args.out_dir)
    plot_averaged(spearman_matrices, "Spearman", args.out_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
