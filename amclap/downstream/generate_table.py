import json
import os
import argparse


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


METRICS = ["mean_reciprocal_rank", "recall@1", "recall@5", "recall@10"]
METRIC_LABELS = ["MRR", "R@1", "R@5", "R@10"]
DATASETS = [
    ("music_caps", "MusicCaps"),
    ("song_describer", "Song Describer"),
]

# Models to evaluate: (id, loss, data, audio_token_type)
MODELS = [
    ("R09", "InfoNCE", "DTV1+MSD+FS+PSE", False),
    ("1zpfl121", "InfoNCE", "DTV2+MSD+FS+PSE", False),
    ("R04", "InfoNCE", "DTV1+MSD+R4+FS+PSE", True),
    ("rgpxysf5", "InfoNCE", "DTV1+MSD+R4", False),
    ("f5v6nb3l", "InfoNCE/Sigreg", "DTV1+MSD+R4", False),
    ("vcu8j2r3", "InfoNCE/Sigreg", "DTV1+MSD+R4+FS+PSE", False),
    ("5pnv42wy", "InfoNCE/Sigreg", "DTV2+MSD+R4+FS+PSE", False),
]


def fmt(val):
    if val is None:
        return "--"
    return f"{val * 100:.1f}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX table from evaluation results"
    )
    parser.add_argument(
        "--models", nargs="+", help="List of model IDs (overrides built-in list)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base output directory containing model results",
    )
    args = parser.parse_args()

    if args.models:
        # Fallback: no metadata, just IDs
        models = [(m, "--", "--", False) for m in args.models]
    else:
        models = MODELS

    # Collect results
    rows = []
    for model_id, loss, data, use_audio_type_token in models:
        model_dir = os.path.join(args.output_dir, model_id)
        row = {
            "id": model_id,
            "loss": loss,
            "data": data,
            "audio_token_type": "yes" if use_audio_type_token else "no",
        }
        for ds_key, _ in DATASETS:
            path = os.path.join(model_dir, ds_key, "caption.json")
            data_json = load_json(path)
            for metric in METRICS:
                key = f"{ds_key}_{metric}"
                row[key] = data_json[metric] if data_json else None
        rows.append(row)

    # Generate LaTeX
    n_ds = len(DATASETS)
    n_metrics = len(METRICS)
    meta_cols = 4  # id, loss, data, audio_token_type

    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{llll" + "c" * (n_ds * n_metrics) + "}")
    print(r"\toprule")

    # Header row 1: dataset group spans
    header1 = (
        r"\multicolumn{1}{c}{}"
        + r" & \multicolumn{1}{c}{}"
        + r" & \multicolumn{1}{c}{}"
        + r" & \multicolumn{1}{c}{}"
    )
    for _, ds_label in DATASETS:
        header1 += rf" & \multicolumn{{{n_metrics}}}{{c}}{{\textbf{{{ds_label}}}}}"
    header1 += r" \\"
    print(header1)

    # Cmidrules for dataset columns
    rules = []
    for i in range(n_ds):
        start = meta_cols + 1 + i * n_metrics
        end = start + n_metrics - 1
        rules.append(rf"\cmidrule(lr){{{start}-{end}}}")
    print(" ".join(rules))

    # Header row 2: column names
    header2 = r"\textbf{ID} & \textbf{Loss} & \textbf{Data} & \textbf{AudTokType}"
    for _ in DATASETS:
        for label in METRIC_LABELS:
            header2 += f" & {label}"
    header2 += r" \\"
    print(header2)
    print(r"\midrule")

    # Data rows
    for row in rows:
        line = (
            f"{row['id']} & {row['loss']} & {row['data']} & {row['audio_token_type']}"
        )
        for ds_key, _ in DATASETS:
            for metric in METRICS:
                key = f"{ds_key}_{metric}"
                line += f" & {fmt(row[key])}"
        line += r" \\"
        print(line)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Text--audio retrieval results on MusicCaps and Song Describer.}")
    print(r"\label{tab:text_audio_retrieval}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
