import json
import os
import argparse


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


DATASETS = [
    ("gtzan_zsl", "GTZAN"),
    ("fma_small_zsl", "FMA (Small)"),
]

# (id, loss, data, audio_type_token)
MODELS = [
    ("R09", "InfoNCE", "DTV1+MSD+FS+PSE", False),
    ("1zpfl121", "InfoNCE", "DTV2+MSD+FS+PSE", True),
    ("R04", "InfoNCE", "DTV1+MSD+R4+FS+PSE", True),
    ("rgpxysf5", "InfoNCE", "DTV1+MSD+R4", False),
    ("f5v6nb3l", "InfoNCE/Sigreg", "DTV1+MSD+R4", False),
]

BASELINES = [
    # (name, gtzan_acc, fma_acc)  — values already in % or None
    ("Laion-CLAP", 0.72, None),
    ("TTMR++", None, None),
    (r"$CLAMP3_{saas}$", None, None),
]


def fmt_acc(val):
    """Format accuracy: input is a float in [0,1], output is percent with 1 decimal."""
    if val is None:
        return "--"
    return f"{val * 100:.1f}"


def fmt_baseline(val):
    """Format a baseline value that is already in [0,1] scale."""
    if val is None:
        return "--"
    return f"{val * 100:.1f}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX zero-shot classification table from evaluation results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base output directory containing model results",
    )
    args = parser.parse_args()

    # Collect our model results
    rows = []
    for model_id, loss, data, use_audio_type_token in MODELS:
        row = {
            "id": model_id,
            "loss": loss,
            "data": data,
            "audio_token_type": "yes" if use_audio_type_token else "no",
        }
        for ds_key, _ in DATASETS:
            path = os.path.join(args.output_dir, model_id, ds_key, "results.json")
            data_json = load_json(path)
            row[ds_key] = data_json["accuracy"] if data_json else None
        rows.append(row)

    n_ds = len(DATASETS)
    meta_cols = 4  # id, loss, data, audio_token_type

    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\begin{tabular}{llll" + "c" * n_ds + "}")
    print(r"\toprule")

    # Header
    header = r"\textbf{ID} & \textbf{Loss} & \textbf{Data} & \textbf{AudTokType}"
    for _, ds_label in DATASETS:
        header += rf" & \textbf{{{ds_label}}}"
    header += r" \\"
    print(header)

    sub_header = r" & & & "
    for _ in DATASETS:
        sub_header += r" & Acc."
    sub_header += r" \\"
    print(sub_header)

    # Baselines section
    print(r"\midrule")
    print(rf"\multicolumn{{{meta_cols + n_ds}}}{{l}}{{\textbf{{Baselines}}}} \\")
    for name, gtzan_acc, fma_acc in BASELINES:
        accs = [gtzan_acc, fma_acc]
        line = rf"\multicolumn{{4}}{{l}}{{{name}}}"
        for acc in accs:
            line += f" & {fmt_baseline(acc)}"
        line += r" \\"
        print(line)

    # Our models section
    print(r"\midrule")
    print(rf"\multicolumn{{{meta_cols + n_ds}}}{{l}}{{\textbf{{Our Models}}}} \\")
    for row in rows:
        line = (
            f"{row['id']} & {row['loss']} & {row['data']} & {row['audio_token_type']}"
        )
        for ds_key, _ in DATASETS:
            line += f" & {fmt_acc(row[ds_key])}"
        line += r" \\"
        print(line)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Zero-shot classification accuracy on GTZAN and FMA (Small).}")
    print(r"\label{tab:zero_shot_classification}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
