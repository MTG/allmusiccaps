"""EX1 — Lexical-slice MRR on existing benchmarks.

Tag each MuCaps and SongDescriber query with binary labels from curated
lexical slices (derived from E8's distinctive-term analysis), then compute
MRR per (slice, model) to show that review-trained models improve most on
narrative/affect/scene/figurative queries and least on technical ones.

Inputs:
    downstream_results/<model_id>/{music_caps,song_describer}/caption2rank.json

Outputs:
    scripts/lexical_slices/slice_mrr_results.json   — raw per-slice MRR table
    scripts/lexical_slices/slice_mrr_table.txt       — formatted ASCII table
    (optionally) scripts/lexical_slices/slice_mrr_table.tex — LaTeX table

Lexicon slices (curated from rank_divergence_k300 distinctive terms + PLAN):
    narrative   — temporal/structural language (opens, transitions, builds, ...)
    affect      — emotional/mood descriptors (melancholic, brooding, haunting, ...)
    scene       — contextual/evocative phrases (live performance, perfect for, ...)
    figurative  — comparative/metaphorical language (like, reminiscent, evokes, ...)
    technical   — production/recording language (mono, mix, microphone, tuning, ...)

Usage:
    python scripts/lexical_slices/slice_mrr.py
    python scripts/lexical_slices/slice_mrr.py --models R01 R04 R05
    python scripts/lexical_slices/slice_mrr.py --pool   # pool MuCaps + SongD
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_BASE = REPO_ROOT / "downstream_results"

# ── Model configuration ─────────────────────────────────────────────────────
# Table 1 models from the PLAN
TABLE1_MODELS = [
    "R01",  # tags+sounds baseline (no reviews)
    "R02",  # quotes only
    "R03",  # struct only
    "R04",  # quotes+mu+so
    "R05",  # struct+mu+so
]

MODEL_NAMES = {
    "R01": "tags+sounds",
    "R02": "quotes",
    "R03": "struct",
    "R04": "quotes+mu+so",
    "R05": "struct+mu+so",
    "R14": "quotes+mu+so+SigReg",
}

DATASETS = ["music_caps", "song_describer"]

# ── Lexicon slices ───────────────────────────────────────────────────────────
# Curated from rank_divergence_k300 distinctive terms + PLAN § EX1.
# Each slice is a set of lowercased stems/words. Multi-word phrases are
# matched as substrings (case-insensitive) before tokenization.

# Phrases matched as substrings in the raw query text (before tokenization).
PHRASE_SLICES: dict[str, list[str]] = {
    "scene": [
        "live performance",
        "perfect for",
        "feels like",
        "sounds like",
        "in a bar",
        "in a club",
        "at a concert",
        "at a party",
        "in a church",
        "in a cafe",
        "in a restaurant",
        "playing at",
        "played at",
        "may be playing",
        "could be playing",
        "would fit",
        "suitable for",
        "background music",
        "soundtrack",
    ],
    "figurative": [
        "reminiscent of",
        "hints at",
        "suggests a",
        "as if",
        "as though",
    ],
}

# Single-word tokens matched after lowercased tokenization.
TOKEN_SLICES: dict[str, set[str]] = {
    "narrative": {
        # Temporal / structural / sequencing language
        "opens",
        "transitions",
        "transition",
        "builds",
        "building",
        "leading",
        "leads",
        "next",
        "then",
        "before",
        "after",
        "starts",
        "start",
        "begins",
        "ends",
        "drops",
        "drop",
        "breaks",
        "shifts",
        "evolves",
        "develops",
        "progresses",
        "follows",
        "introduces",
        "returns",
        "repeats",
        "stops",
        "continues",
        "fades",
        "climbs",
        "resolves",
        "gradually",
        "suddenly",
        "eventually",
        "finally",
        "initially",
        "progressively",
        "indicating",
        "serves",
    },
    "affect": {
        # Emotional / mood descriptors
        "melancholic",
        "melancholy",
        "brooding",
        "chaotic",
        "euphoric",
        "tension",
        "yearning",
        "quirky",
        "wistful",
        "haunting",
        "eerie",
        "sinister",
        "ominous",
        "mysterious",
        "dreamy",
        "nostalgic",
        "romantic",
        "passionate",
        "aggressive",
        "angry",
        "joyful",
        "playful",
        "serene",
        "peaceful",
        "somber",
        "hopeful",
        "desperate",
        "triumphant",
        "bittersweet",
        "compelling",
        "entertaining",
        "futuristic",
        "boisterous",
        "soothing",
        "uplifting",
        "contemplative",
        "intimate",
        "dramatic",
        "whimsical",
        "unsettling",
        "aura",
    },
    "scene": {
        # Single-word scene/context markers (supplement phrase matches)
        "documentary",
        "cinematic",
        "commercial",
        "concert",
        "worship",
        "meditation",
        "workout",
        "game",
    },
    "figurative": {
        # Comparative / metaphorical single-word markers
        "like",
        "reminiscent",
        "evokes",
        "evocative",
        "resembles",
        "resembling",
        "suggests",
        "suggesting",
        "echoes",
        "echoing",
        "conjures",
        "metaphor",
        "metaphorical",
        "imagery",
    },
    "technical": {
        # Production / recording / mix language (control slice)
        "mono",
        "stereo",
        "mix",
        "mixing",
        "microphone",
        "tuning",
        "tuned",
        "amateur",
        "low-quality",
        "recording",
        "mastered",
        "mastering",
        "compressed",
        "clipping",
        "distorted",
        "distortion",
        "reverb",
        "delay",
        "panning",
        "equalization",
        "bitrate",
        "sample",
        "sampling",
        "instrumentation",
        "arrangement",
        "production",
        "fidelity",
        "frequency",
        "narrow",
        "bandwidth",
        "vocoder",
    },
}

SLICE_NAMES = ["narrative", "affect", "scene", "figurative", "technical"]

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def assign_slices(query: str) -> dict[str, bool]:
    """Return multi-hot slice labels for a query."""
    labels = {s: False for s in SLICE_NAMES}
    query_lower = query.lower()

    # Phrase matching (substring in raw text)
    for slice_name, phrases in PHRASE_SLICES.items():
        for phrase in phrases:
            if phrase in query_lower:
                labels[slice_name] = True
                break

    # Token matching
    tokens = set(tokenize(query))
    for slice_name, keywords in TOKEN_SLICES.items():
        if tokens & keywords:
            labels[slice_name] = True

    return labels


def load_caption2rank(model_id: str, dataset: str) -> list[dict]:
    path = RESULTS_BASE / model_id / dataset / "caption2rank.json"
    return json.loads(path.read_text())


def compute_mrr(ranks: list[int]) -> float:
    """MRR from 0-indexed ranks."""
    if not ranks:
        return float("nan")
    return sum(1.0 / (r + 1) for r in ranks) / len(ranks)


def main():
    ap = argparse.ArgumentParser(description="EX1: Lexical-slice MRR analysis")
    ap.add_argument(
        "--models",
        nargs="+",
        default=TABLE1_MODELS,
        help="Model IDs to evaluate",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        help="Datasets to evaluate on",
    )
    ap.add_argument(
        "--pool",
        action="store_true",
        help="Pool MuCaps + SongD queries together",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Output directory",
    )
    ap.add_argument(
        "--latex",
        action="store_true",
        help="Also emit a LaTeX table",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect per-query data ───────────────────────────────────────────
    # Structure: results[dataset][model_id] = list of (query, min_rank, slice_labels)
    all_data: dict[str, dict[str, list[tuple[str, int, dict[str, bool]]]]] = (
        defaultdict(lambda: defaultdict(list))
    )

    for dataset in args.datasets:
        # Use first model to get query text + slice labels (identical across models)
        ref_data = load_caption2rank(args.models[0], dataset)
        query_slices = {}
        for item in ref_data:
            query_slices[item["index"]] = assign_slices(item["query"])

        for model_id in args.models:
            data = load_caption2rank(model_id, dataset)
            for item in data:
                idx = item["index"]
                all_data[dataset][model_id].append(
                    (item["query"], item["min_rank"], query_slices[idx])
                )

    # ── Compute MRR per slice ────────────────────────────────────────────
    # results_table[pool_key][slice_name][model_id] = (mrr, n_queries)
    results_table: dict[str, dict[str, dict[str, tuple[float, int]]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    pool_keys = ["pooled"] if args.pool else args.datasets

    for pool_key in pool_keys:
        datasets_in_pool = args.datasets if pool_key == "pooled" else [pool_key]

        for model_id in args.models:
            # Gather all items for this model across datasets in this pool
            items = []
            for ds in datasets_in_pool:
                items.extend(all_data[ds][model_id])

            # Aggregate MRR
            all_slice_names = SLICE_NAMES + ["all", "unsliced"]
            for slice_name in all_slice_names:
                if slice_name == "all":
                    ranks = [r for _, r, _ in items]
                elif slice_name == "unsliced":
                    ranks = [
                        r for _, r, sl in items if not any(sl[s] for s in SLICE_NAMES)
                    ]
                else:
                    ranks = [r for _, r, sl in items if sl[slice_name]]

                mrr = compute_mrr(ranks)
                results_table[pool_key][slice_name][model_id] = (mrr, len(ranks))

    # ── Slice coverage statistics ────────────────────────────────────────
    print("\n=== Slice coverage ===\n")
    for pool_key in pool_keys:
        datasets_in_pool = args.datasets if pool_key == "pooled" else [pool_key]
        # Use first model to count
        items = []
        for ds in datasets_in_pool:
            items.extend(all_data[ds][args.models[0]])

        total = len(items)
        print(f"  {pool_key}: {total} queries total")
        for s in SLICE_NAMES:
            n = sum(1 for _, _, sl in items if sl[s])
            print(f"    {s:12s}: {n:5d} ({100 * n / total:5.1f}%)")
        n_any = sum(1 for _, _, sl in items if any(sl[s] for s in SLICE_NAMES))
        n_none = total - n_any
        print(f"    {'any':12s}: {n_any:5d} ({100 * n_any / total:5.1f}%)")
        print(f"    {'unsliced':12s}: {n_none:5d} ({100 * n_none / total:5.1f}%)")
        print()

    # ── Check if pooling is needed ───────────────────────────────────────
    if not args.pool:
        for ds in args.datasets:
            items = all_data[ds][args.models[0]]
            for s in SLICE_NAMES:
                n = sum(1 for _, _, sl in items if sl[s])
                if n < 150:
                    print(
                        f"  WARNING: {ds}/{s} has only {n} queries (<150). "
                        f"Consider --pool to combine datasets."
                    )

    # ── Print ASCII table ────────────────────────────────────────────────
    slice_order = SLICE_NAMES + ["all", "unsliced"]

    for pool_key in pool_keys:
        print(f"\n{'=' * 100}")
        print(f"  MRR (×100) — {pool_key}")
        print(f"{'=' * 100}")

        # Header
        header = f"{'Slice':<14s} {'N':>5s}"
        for model_id in args.models:
            name = MODEL_NAMES.get(model_id, model_id)
            header += f"  {name:>18s}"
        print(header)
        print("-" * len(header))

        for slice_name in slice_order:
            if slice_name in ("all", "unsliced"):
                print("-" * len(header))
            _, n = results_table[pool_key][slice_name][args.models[0]]
            row = f"{slice_name:<14s} {n:>5d}"
            for model_id in args.models:
                mrr, _ = results_table[pool_key][slice_name][model_id]
                row += f"  {mrr * 100:>18.2f}"
            print(row)
        print()

    # ── Delta table (vs baseline) ────────────────────────────────────────
    baseline_id = args.models[0]  # R01 by default
    print(f"\n{'=' * 100}")
    print(f"  MRR delta vs {MODEL_NAMES.get(baseline_id, baseline_id)} (×100)")
    print(f"{'=' * 100}")

    for pool_key in pool_keys:
        print(f"\n  --- {pool_key} ---")
        header = f"{'Slice':<14s} {'N':>5s}"
        for model_id in args.models[1:]:
            name = MODEL_NAMES.get(model_id, model_id)
            header += f"  {name:>18s}"
        print(header)
        print("-" * len(header))

        for slice_name in slice_order:
            if slice_name in ("all", "unsliced"):
                print("-" * len(header))
            base_mrr, n = results_table[pool_key][slice_name][baseline_id]
            row = f"{slice_name:<14s} {n:>5d}"
            for model_id in args.models[1:]:
                mrr, _ = results_table[pool_key][slice_name][model_id]
                delta = (mrr - base_mrr) * 100
                sign = "+" if delta >= 0 else ""
                row += f"  {sign}{delta:>17.2f}"
            print(row)
        print()

    # ── Save JSON ────────────────────────────────────────────────────────
    json_out = {}
    for pool_key in pool_keys:
        json_out[pool_key] = {}
        for slice_name in slice_order:
            json_out[pool_key][slice_name] = {}
            for model_id in args.models:
                mrr, n = results_table[pool_key][slice_name][model_id]
                json_out[pool_key][slice_name][model_id] = {
                    "mrr": round(mrr, 6),
                    "n_queries": n,
                }

    out_path = args.out_dir / "slice_mrr_results.json"
    out_path.write_text(json.dumps(json_out, indent=2))
    print(f"\nWrote {out_path}")

    # ── Save ASCII table ─────────────────────────────────────────────────
    lines = []
    for pool_key in pool_keys:
        lines.append(f"MRR (×100) — {pool_key}")
        lines.append("")
        header = f"{'Slice':<14s} {'N':>5s}"
        for model_id in args.models:
            name = MODEL_NAMES.get(model_id, model_id)
            header += f"  {name:>18s}"
        lines.append(header)
        lines.append("-" * len(header))
        for slice_name in slice_order:
            if slice_name in ("all", "unsliced"):
                lines.append("-" * len(header))
            _, n = results_table[pool_key][slice_name][args.models[0]]
            row = f"{slice_name:<14s} {n:>5d}"
            for model_id in args.models:
                mrr, _ = results_table[pool_key][slice_name][model_id]
                row += f"  {mrr * 100:>18.2f}"
            lines.append(row)
        lines.append("")

    txt_path = args.out_dir / "slice_mrr_table.txt"
    txt_path.write_text("\n".join(lines))
    print(f"Wrote {txt_path}")

    # ── LaTeX table ──────────────────────────────────────────────────────
    if args.latex:
        latex_lines = []
        for pool_key in pool_keys:
            n_cols = len(args.models) + 1
            latex_lines.append(f"% {pool_key}")
            latex_lines.append(r"\begin{tabular}{l" + "r" * len(args.models) + "}")
            latex_lines.append(r"\toprule")

            header_cells = ["Slice"]
            for model_id in args.models:
                name = MODEL_NAMES.get(model_id, model_id)
                header_cells.append(name)
            latex_lines.append(" & ".join(header_cells) + r" \\")
            latex_lines.append(r"\midrule")

            for slice_name in slice_order:
                if slice_name in ("all", "unsliced"):
                    latex_lines.append(r"\midrule")
                _, n = results_table[pool_key][slice_name][args.models[0]]
                cells = [f"{slice_name} ({n})"]
                for model_id in args.models:
                    mrr, _ = results_table[pool_key][slice_name][model_id]
                    cells.append(f"{mrr * 100:.1f}")
                latex_lines.append(" & ".join(cells) + r" \\")

            latex_lines.append(r"\bottomrule")
            latex_lines.append(r"\end{tabular}")
            latex_lines.append("")

        tex_path = args.out_dir / "slice_mrr_table.tex"
        tex_path.write_text("\n".join(latex_lines))
        print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
