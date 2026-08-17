"""Top per-query rank improvements from adding review supervision.

For each (review-augmented model, dataset) pair, compute
    delta = rank(R01) - rank(model)
on the common query set (delta > 0 means review supervision helped).
Print the top-K queries by delta, and emit a LaTeX table.

Usage:
    python top_review_gains.py            # K=5, both pairs, both datasets
    python top_review_gains.py --top-k 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_BASE = REPO_ROOT / "downstream_results"

BASELINE = "R01"
PAIRS = [
    ("R04", "quotes+tags+sounds"),
    ("R05", "struct+tags+sounds"),
]
DATASETS = [
    ("music_caps", "MuCaps"),
    ("song_describer", "SongD."),
]


def load_caption2rank(model_id: str, dataset: str) -> dict[int, dict]:
    path = RESULTS_BASE / model_id / dataset / "caption2rank.json"
    items = json.loads(path.read_text())
    return {it["index"]: it for it in items}


def latex_escape(s: str) -> str:
    # Minimal escaping for caption text in LaTeX cells.
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)


def truncate(s: str, max_chars: int) -> str:
    s = " ".join(s.split())
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "\u2026"


def top_gains(model_id: str, dataset: str, k: int) -> list[dict]:
    base = load_caption2rank(BASELINE, dataset)
    mod = load_caption2rank(model_id, dataset)
    common = sorted(set(base) & set(mod))
    rows = []
    for idx in common:
        r_base = base[idx]["min_rank"]
        r_mod = mod[idx]["min_rank"]
        rows.append(
            {
                "index": idx,
                "delta": r_base - r_mod,  # >0 → review-augmented better
                "rank_baseline": r_base,
                "rank_model": r_mod,
                "query": base[idx]["query"],
            }
        )
    rows.sort(key=lambda r: r["delta"], reverse=True)
    return rows[:k]


def emit_latex(all_results: dict, max_chars: int) -> str:
    """all_results: {(model_id, dataset_label): [rows]} → LaTeX string."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{rp{6.0cm}}")
    lines.append(r"\toprule")
    lines.append(r"$\Delta$\,rank & Query \\")
    lines.append(r"\midrule")
    blocks = []
    for (_model_id, model_label, dataset_label), rows in all_results.items():
        block = []
        block.append(
            rf"\multicolumn{{2}}{{l}}{{\textit{{{model_label} vs.\ "
            rf"\texttt{{{BASELINE}}} on {dataset_label}}}}} \\"
        )
        for r in rows:
            q = latex_escape(truncate(r["query"], max_chars))
            block.append(rf"$+${r['delta']} & {q} \\")
        blocks.append("\n".join(block))
    lines.append("\n\\midrule\n".join(blocks))
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Top-5 queries with the largest rank improvement when adding "
        r"review supervision on top of the tags+sounds baseline "
        r"(\texttt{" + BASELINE + r"}). $\Delta$\,rank = rank under baseline "
        r"$-$ rank under review-augmented model; larger is better.}"
    )
    lines.append(r"\label{tab:top_review_gains}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-chars", type=int, default=220)
    args = ap.parse_args()

    all_results = {}
    for model_id, model_label in PAIRS:
        for ds, ds_label in DATASETS:
            rows = top_gains(model_id, ds, args.top_k)
            all_results[(model_id, model_label, ds_label)] = rows
            print(f"\n=== {model_label} ({model_id}) vs {BASELINE} on {ds_label} ===")
            for r in rows:
                print(
                    f"  +{r['delta']:>4}  (base {r['rank_baseline']:>4} "
                    f"→ {r['rank_model']:>4})  {r['query'][:140]}"
                )

    print()
    print("=" * 80)
    print("LaTeX")
    print("=" * 80)
    print(emit_latex(all_results, args.max_chars))


if __name__ == "__main__":
    main()
