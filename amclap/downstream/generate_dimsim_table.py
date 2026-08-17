import json
import os
import argparse


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# (id, loss, data, audio_type_token)
MODELS = [
    ("R09", "InfoNCE", "DTV1+MSD+FS+PSE", False),
    ("1zpfl121", "InfoNCE", "DTV2+MSD+FS+PSE", True),
    ("R04", "InfoNCE", "DTV1+MSD+R4+FS+PSE", True),
    ("rgpxysf5", "InfoNCE", "DTV1+MSD+R4", False),
    ("f5v6nb3l", "InfoNCE/Sigreg", "DTV1+MSD+R4", False),
]

BASELINES = [
    # (name, accuracy) — None means not available
    ("Laion-CLAP", None),
    ("TTMR++", None),
    (r"$CLAMP3_{saas}$", None),
]


def fmt(val):
    if val is None:
        return "--"
    return f"{val * 100:.1f}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX DimSim similarity table from evaluation results"
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
        path = os.path.join(args.output_dir, model_id, "dimsim", "results.json")
        data_json = load_json(path)
        rows.append(
            {
                "id": model_id,
                "loss": loss,
                "data": data,
                "audio_token_type": "yes" if use_audio_type_token else "no",
                "accuracy": data_json["accuracy"] if data_json else None,
            }
        )

    meta_cols = 4  # id, loss, data, audio_token_type

    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\begin{tabular}{llllc}")
    print(r"\toprule")
    print(
        r"\textbf{ID} & \textbf{Loss} & \textbf{Data} & \textbf{AudTokType} & \textbf{DimSim} \\"
    )
    print(r" & & & & Acc. \\")

    # Baselines
    print(r"\midrule")
    print(rf"\multicolumn{{{meta_cols + 1}}}{{l}}{{\textbf{{Baselines}}}} \\")
    for name, acc in BASELINES:
        print(rf"\multicolumn{{4}}{{l}}{{{name}}} & {fmt(acc)} \\")

    # Our models
    print(r"\midrule")
    print(rf"\multicolumn{{{meta_cols + 1}}}{{l}}{{\textbf{{Our Models}}}} \\")
    for row in rows:
        print(
            f"{row['id']} & {row['loss']} & {row['data']} & {row['audio_token_type']} & {fmt(row['accuracy'])} \\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Music similarity accuracy on DimSim.}")
    print(r"\label{tab:dimsim}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
