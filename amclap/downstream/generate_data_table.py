"""Generate the data-composition LaTeX table for the ISMIR 2026 paper.

Single-column, retrieval + zero-shot only. The baseline is R01
(tags+sounds, no album reviews); the other rows isolate the contribution
of each review-derived text source on top of that baseline.

Usage:
    python generate_data_table.py --output_dir downstream_results
"""

from __future__ import annotations

import argparse
import json
import os

# (id, data_label) — order = paper presentation order
MODELS = [
    ("R01", "tags+sounds"),  # baseline: no review supervision
    ("R02", "quotes"),
    ("R03", "struct"),
    ("R04", "quotes+tags+sounds"),
    ("R05", "struct+tags+sounds"),
]

# (display_name, metric_label, subpath, json_key, scale, higher_is_better, decimals)
COLUMNS = [
    (
        "MuCaps",
        "MRR$\\uparrow$",
        "music_caps/caption.json",
        "mean_reciprocal_rank",
        100,
        True,
        1,
    ),
    (
        "SongD.",
        "MRR$\\uparrow$",
        "song_describer/caption.json",
        "mean_reciprocal_rank",
        100,
        True,
        1,
    ),
    ("GTZAN", "Acc.$\\uparrow$", "gtzan_zsl/results.json", "accuracy", 100, True, 1),
    (
        "FMA-S",
        "Acc.$\\uparrow$",
        "fma_small_zsl/results.json",
        "accuracy",
        100,
        True,
        1,
    ),
]

# Column groups for the header
COLUMN_GROUPS = [
    ("", 2, None),
    ("Retrieval", 2, "retrieval"),
    ("ZS Class.", 2, "zero_shot"),
]


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def collect(model_id: str, output_dir: str) -> dict:
    out = {"id": model_id}
    for name, _, subpath, key, *_ in COLUMNS:
        data = load_json(os.path.join(output_dir, model_id, subpath))
        out[name] = data.get(key) if data else None
    return out


def best_per_col(rows: list[dict]) -> dict:
    best = {}
    for name, _, _, _, _, higher, _ in COLUMNS:
        vals = [r[name] for r in rows if r[name] is not None]
        if not vals:
            best[name] = None
        else:
            best[name] = max(vals) if higher else min(vals)
    return best


def fmt(val, scale, decimals, best_val):
    if val is None:
        return "--"
    s = f"{val * scale:.{decimals}f}"
    if best_val is not None and abs(val - best_val) < 1e-9:
        return rf"\textbf{{{s}}}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    rows = [collect(mid, args.output_dir) for mid, _ in MODELS]
    best = best_per_col(rows)
    n = len(COLUMNS)

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\setlength{\tabcolsep}{3pt}")
    print(r"\begin{tabular}{lp{2.5cm}" + "c" * n + "}")
    print(r"\toprule")

    # group spans
    parts = []
    for label, span, _ in COLUMN_GROUPS:
        if not label:
            parts.append(rf"\multicolumn{{{span}}}{{l}}{{}}")
        else:
            parts.append(rf"\multicolumn{{{span}}}{{c}}{{\textbf{{{label}}}}}")
    print(" & ".join(parts) + r" \\")

    rules, cur = [], 1
    for label, span, _ in COLUMN_GROUPS:
        if label:
            rules.append(rf"\cmidrule(lr){{{cur}-{cur + span - 1}}}")
        cur += span
    print(" ".join(rules))

    # dataset names + metric rows
    print(
        " & ".join([r"\textbf{ID}", r"\textbf{Data}"] + [c[0] for c in COLUMNS])
        + r" \\"
    )
    print(" & ".join(["", ""] + [c[1] for c in COLUMNS]) + r" \\")
    print(r"\midrule")

    for (mid, label), row in zip(MODELS, rows):
        cells = [rf"\texttt{{{mid}}}", label]
        for name, _, _, _, scale, _, decimals in COLUMNS:
            cells.append(fmt(row[name], scale, decimals, best[name]))
        print(" & ".join(cells) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(
        r"\caption{Contribution of each review-derived text corpus to retrieval "
        r"and zero-shot classification, on top of a tags+sounds baseline "
        r"(\texttt{R01}: MSD + r4 + Freesound + PSE, no Discogs reviews). "
        r"\textit{quotes} and \textit{struct} cover the same \num{222}\,k Discogs tracks "
        r"with two different LLM caption styles. Retrieval: MRR (\%); "
        r"ZS Class.: accuracy (\%). \textbf{Bold} = best per column.}"
    )
    print(r"\label{tab:data}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
