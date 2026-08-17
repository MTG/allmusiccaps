"""Generate a combined LaTeX table covering all downstream evaluation tasks."""

import json
import os
import argparse


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# (id, loss, data, audio_type_token)
# audio_type_token=True → id shown with $^*$ superscript
# loss is shown as a suffix on the data string when not InfoNCE
MODELS = [
    # data
    ("R01", "InfoNCE", "mu+so (no reviews)", False),
    ("R02", "InfoNCE", "quotes", False),
    ("R03", "InfoNCE", "struct", False),
    ("R04", "InfoNCE", "quotes+mu+so", False),
    ("R05", "InfoNCE", "struct+mu+so", False),
    #  encoder
    ("R06", "InfoNCE", "Layer 6", False),
    ("R07", "InfoNCE", "All layer", False),
    ("ny6g2bzr", "InfoNCE", "All layers/TT", False),
    ("R14", "InfoNCE", "All layers/TT + SigReg", False),
    # ("5grt2e1a", "SigReg", "quotes+mu+so", False),
    # loss
    ("R08", "Sigmoid", "--", False),
    ("z0r7jh5a", "LeJEPA", "1 view", False),
    ("fze0cez1", "LeJEPA", "2 views", False),
    ("7edqcl5n", "LeJEPA", "4 views", False),
    # Did not work
    ("2cvx96fi", "SLAP", "--", False),
    ("f9l9z22m", "LpJEPA", "--", False),
    ("8jlpuoge", "LpJEPA", "mixed", False),
    #
    # ("R09", "InfoNCE", "DTV1+MSD+FS+PSE", False),
    # ("1zpfl121", "InfoNCE", "DTV2+MSD+FS+PSE", False),
    # ("R04", "InfoNCE", "DTV1+MSD+R4+FS+PSE", True),
    # ("rgpxysf5", "InfoNCE", "DTV1+MSD+R4", False),
    # ("f5v6nb3l", "Sig.", "DTV1+MSD+R4", False),
    # ("6si461i0", "Sig. bal", "DTV1+MSD+R4", False),
    # ("kqbtt80w", "Sig. dyn", "DTV1+MSD+R4", False),
    # ("vcu8j2r3", "Sig. dyn", "DTV1+MSD+R4+FS+PSE", False),
    # ("5pnv42wy", "Sig. dyn", "DTV2+MSD+R4+FS+PSE", False),
]

# Baseline rows: (display_name, {col_index: raw_value})
# col_index matches position in COLUMNS list (0-based).
# Values are in the same raw scale as the JSON outputs (i.e. [0,1] for
# metrics that use scale_factor=100).  The scale is applied at render time.
#
# COLUMNS index reference for tagging MAP metrics:
#
#   0 → MuCaps MRR
#   1 → Song Describer MRR
#   2 -> GTZAN ZS Acc.
#   3 → FMA-small ZS Acc.
#   4 → DimSim Acc.
#   5 → MTT MAP
#   6 → J.Genre MAP
#   7 → J.Instr. MAP
#   8 → J.Mood MAP
#   9 → MGPHot RMSE

BASELINES = [
    (
        "Laion-CLAP",
        {
            0: 0.1018,
            1: 0.1703,
            2: 0.7150,
            3: 0.5549,
            4: 0.8055,
            5: 0.462,
            6: 0.183,
            7: 0.184,
            8: 0.146,
            9: 0.165,
        },
    ),
    (
        "TTMR++",
        {
            0: 0.1229,
            1: 0.1616,
            2: 0.8167,
            3: 0.4333,
            4: 0.6974,
            5: 0.465,
            6: 0.200,
            7: 0.193,
            8: 0.147,
            9: 0.169,
        },
    ),
    (
        r"$\text{CLAMP3}_{saas}$",
        {
            0: 0.0569,
            1: 0.1653,
            2: 0.505,
            3: 0.3592,
            4: 0.7179,
            5: 0.463,
            6: 0.192,
            7: 0.204,
            8: 0.146,
            9: 0.175,
        },
    ),
]


# Columns: (dataset_label, metric_label, result_subdir, json_key, scale_factor, task, higher_is_better, decimals)
# task → one of: retrieval | zero_shot | similarity | probing
# dataset_label    → header row 1 (short dataset name)
# metric_label     → header row 2 (metric + arrow)
# scale_factor     → multiply raw value (100 to convert [0,1] → %)
# higher_is_better → True for ↑ metrics, False for ↓ metrics
# decimals         → number of decimal places in the table cell
COLUMNS = [
    # --- Retrieval (MRR only) ---
    (
        "MuCaps",
        "MRR$\\uparrow$",
        "music_caps/caption.json",
        "mean_reciprocal_rank",
        100,
        "retrieval",
        True,
        1,
    ),
    (
        "SongD.",
        "MRR$\\uparrow$",
        "song_describer/caption.json",
        "mean_reciprocal_rank",
        100,
        "retrieval",
        True,
        1,
    ),
    # --- Zero-shot classification ---
    (
        "GTZAN",
        "Acc.$\\uparrow$",
        "gtzan_zsl/results.json",
        "accuracy",
        100,
        "zero_shot",
        True,
        1,
    ),
    (
        "FMA-S",
        "Acc.$\\uparrow$",
        "fma_small_zsl/results.json",
        "accuracy",
        100,
        "zero_shot",
        True,
        1,
    ),
    # --- Music similarity ---
    (
        "DimSim",
        "Acc.$\\uparrow$",
        "dimsim/results.json",
        "accuracy",
        100,
        "similarity",
        True,
        1,
    ),
    # --- Autotagging: MTT ---
    (
        "MTT",
        "MAP$\\uparrow$",
        "mtt_autotagging/results.json",
        "test-MAP-macro",
        100,
        "probing",
        True,
        1,
    ),
    # --- Autotagging: Jamendo ---
    (
        "J.Gen",
        "MAP$\\uparrow$",
        "jamendo_genre/results.json",
        "test-MAP-macro",
        100,
        "probing",
        True,
        1,
    ),
    (
        "J.Ins",
        "MAP$\\uparrow$",
        "jamendo_instrument/results.json",
        "test-MAP-macro",
        100,
        "probing",
        True,
        1,
    ),
    (
        "J.Mood",
        "MAP$\\uparrow$",
        "jamendo_moodtheme/results.json",
        "test-MAP-macro",
        100,
        "probing",
        True,
        1,
    ),
    # --- MGPHot regression ---
    (
        "MGPHot",
        "RMSE$\\downarrow$",
        "mgphot_regression/results.json",
        "test-RMSE-macro",
        1,
        "probing",
        False,
        3,
    ),
]

# Top-level group spans: (group_label, n_cols, task)
# Empty string → no label (meta columns), task=None → always included
COLUMN_GROUPS = [
    ("", 2, None),  # ID / Data
    ("Retrieval", 2, "retrieval"),
    ("ZS Class.", 2, "zero_shot"),
    ("Sim.", 1, "similarity"),
    ("MTT", 1, "probing"),
    ("J.Genre", 1, "probing"),
    ("J.Instr.", 1, "probing"),
    ("J.Mood", 1, "probing"),
    ("MGPHot", 1, "probing"),
]

TASK_CHOICES = {"all", "retrieval", "zero_shot", "similarity", "probing"}

META_COLS = 2  # ID, Data (loss encoded as suffix; ATT encoded as superscript)

# Named subsets of model IDs for --groups
MODEL_GROUPS = {
    "data": ["R02", "R03", "R04", "R05"],
    "encoder": ["R04", "R06", "R07", "ny6g2bzr", "R14"],
    "jepa_views": ["z0r7jh5a", "fze0cez1", "7edqcl5n"],
}


def fmt(val, scale):
    if val is None:
        return "--"
    return f"{val * scale:.1f}"


def col_key(subpath, key):
    return f"{subpath}/{key}"


def model_id_cell(model_id, step=None):
    text = f"{model_id}@{step}" if step is not None else model_id
    return rf"\texttt{{{text}}}"


def model_data_cell(loss, data, use_audio_type_token):
    """Compact data+loss cell: append loss in parens when not plain InfoNCE,
    and add * superscript when audio type token is used."""
    base = data if loss == "InfoNCE" else rf"{data} ({loss})"
    if use_audio_type_token:
        return rf"{base}$^*$"
    return base


def filter_columns(tasks):
    """Return filtered (columns, column_groups) based on selected task set."""
    if tasks == {"all"}:
        return COLUMNS, COLUMN_GROUPS
    cols = [c for c in COLUMNS if c[5] in tasks]
    # Keep only groups whose task is selected (meta group always included).
    groups = [
        (label, n, task)
        for label, n, task in COLUMN_GROUPS
        if task is None or task in tasks
    ]
    return cols, groups


def collect_row(
    model_id, loss, data, use_audio_type_token, output_dir, columns, step=None
):
    row = {
        "id": model_id,
        "loss": loss,
        "data": data,
        "use_att": use_audio_type_token,
        "step": step,
    }
    cache = {}
    step_dir = f"step={step}" if step is not None else None
    for _, _, subpath, key, _, _, _, _ in columns:
        base = os.path.join(output_dir, model_id)
        path = (
            os.path.join(base, step_dir, subpath)
            if step_dir
            else os.path.join(base, subpath)
        )
        if path not in cache:
            cache[path] = load_json(path)
        data_json = cache[path]
        row[col_key(subpath, key)] = data_json.get(key) if data_json else None
    return row


def compute_best(rows, columns):
    """For each column, find the best value among our models. Returns list of best values."""
    best = []
    for _, _, subpath, key, _, _, higher_is_better, _ in columns:
        vals = [
            row[col_key(subpath, key)]
            for row in rows
            if row[col_key(subpath, key)] is not None
        ]
        if not vals:
            best.append(None)
        elif higher_is_better:
            best.append(max(vals))
        else:
            best.append(min(vals))
    return best


def fmt_cell(val, scale, best_val, decimals=1):
    """Format value, bolding it if it matches the best."""
    if val is None:
        return "--"
    formatted = f"{val * scale:.{decimals}f}"
    if best_val is not None and abs(val - best_val) < 1e-9:
        return rf"\textbf{{{formatted}}}"
    return formatted


def main():
    parser = argparse.ArgumentParser(
        description="Generate combined LaTeX table for all downstream tasks"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base output directory containing per-model result subdirectories",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Evaluate results at a specific checkpoint step (e.g. 40000). "
        "Looks under {output_dir}/{model_id}/step={step}/",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        choices=sorted(MODEL_GROUPS.keys()),
        metavar="GROUP",
        help=(
            "Model groups to include. Options: "
            + ", ".join(sorted(MODEL_GROUPS.keys()))
            + ". Default: all models."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        choices=sorted(TASK_CHOICES),
        metavar="TASK",
        help=(
            "Tasks to include in the table. "
            "Options: all retrieval zero_shot similarity probing. "
            "Default: all"
        ),
    )
    args = parser.parse_args()

    tasks = set(args.tasks)
    columns, column_groups = filter_columns(tasks)

    if args.groups:
        allowed_ids = {mid for g in args.groups for mid in MODEL_GROUPS[g]}
        # Preserve order from MODELS, deduplicate while keeping first occurrence
        seen = set()
        filtered_models = []
        for entry in MODELS:
            mid = entry[0]
            if mid in allowed_ids and mid not in seen:
                filtered_models.append(entry)
                seen.add(mid)
        models = filtered_models
    else:
        models = MODELS

    rows = [
        collect_row(mid, loss, data, att, args.output_dir, columns, step=args.step)
        for mid, loss, data, att in models
    ]
    best = compute_best(rows, columns)

    n_data_cols = len(columns)

    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\setlength{\tabcolsep}{3pt}")
    # p-column for Data so long strings wrap; fixed width
    print(r"\begin{tabular}{lp{3.2cm}" + "c" * n_data_cols + "}")
    print(r"\toprule")

    # ---- Header row 1: group spans ----
    parts = []
    for group_label, n, _ in column_groups:
        align = "l" if not group_label else "c"
        if not group_label:
            parts.append(rf"\multicolumn{{{n}}}{{{align}}}{{}}")
        else:
            parts.append(rf"\multicolumn{{{n}}}{{{align}}}{{\textbf{{{group_label}}}}}")
    print(" & ".join(parts) + r" \\")

    # ---- Cmidrules for labeled groups only ----
    rules = []
    cursor = 1
    for group_label, n, _ in column_groups:
        if group_label:
            rules.append(rf"\cmidrule(lr){{{cursor}-{cursor + n - 1}}}")
        cursor += n
    print(" ".join(rules))

    # ---- Header row 2: short dataset names ----
    meta_row2 = [r"\textbf{ID}", r"\textbf{Data}"]
    dataset_labels = [ds for ds, _, _, _, _, _, _, _ in columns]
    print(" & ".join(meta_row2 + dataset_labels) + r" \\")

    # ---- Header row 3: metric + arrow ----
    meta_row3 = ["", ""]
    metric_labels = [m for _, m, _, _, _, _, _, _ in columns]
    print(" & ".join(meta_row3 + metric_labels) + r" \\")
    print(r"\midrule")

    # ---- Baselines section ----
    # Map original COLUMNS index to active column index for baseline lookup
    active_orig_indices = [
        i for i, c in enumerate(COLUMNS) if tasks == {"all"} or c[5] in tasks
    ]
    print(rf"\multicolumn{{{META_COLS + n_data_cols}}}{{l}}{{\textbf{{Baselines}}}} \\")
    for name, values in BASELINES:
        cells = [rf"\multicolumn{{2}}{{l}}{{{name}}}"]
        for orig_i in active_orig_indices:
            scale = COLUMNS[orig_i][4]
            decimals = COLUMNS[orig_i][7]
            raw = values.get(orig_i)
            cells.append("--" if raw is None else f"{raw * scale:.{decimals}f}")
        print(" & ".join(cells) + r" \\")

    # ---- Our models section ----
    print(r"\midrule")
    print(
        rf"\multicolumn{{{META_COLS + n_data_cols}}}{{l}}{{\textbf{{Our Models}}}} \\"
    )
    for row in rows:
        id_cell = model_id_cell(row["id"], row["step"])
        data_cell = model_data_cell(row["loss"], row["data"], row["use_att"])
        cells = [id_cell, data_cell]
        for i, (_, _, subpath, key, scale, _, _, decimals) in enumerate(columns):
            cells.append(fmt_cell(row[col_key(subpath, key)], scale, best[i], decimals))
        print(" & ".join(cells) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(
        r"\caption{Combined downstream evaluation. "
        r"Retrieval: MRR (\%); ZS Class.\ and Similarity: Accuracy (\%); "
        r"Autotagging: AUROC and MAP (\%). "
        r"$^*$~audio type token enabled. "
        r"\textbf{Bold} = best among our models per column.}"
    )
    print(r"\label{tab:full_downstream}")
    print(r"\end{table*}")


if __name__ == "__main__":
    main()
