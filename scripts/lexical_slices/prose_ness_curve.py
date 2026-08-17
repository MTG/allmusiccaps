"""EX2 — Evaluative-query MusicCaps subset: prose-ness curve.

Compute a continuous "prose-ness" score for each MuCaps (and optionally
SongDescriber) query, then plot MRR in percentile bins to show that
review-trained models' advantage grows with prose-ness.

Prose-ness features (combined into a single score via sum-of-z-scores):
    1. Caption length in tokens
    2. Adjective density (spaCy POS: ADJ count / token count)
    3. Affect/narrative keyword count (from EX1 lexicon)
    4. Simile/metaphor marker count

Inputs:
    downstream_results/<model_id>/{music_caps,song_describer}/caption2rank.json

Outputs:
    scripts/lexical_slices/prose_ness_curve.png — MRR vs prose-ness percentile
    scripts/lexical_slices/prose_ness_scores.json — per-query scores + labels

Usage:
    python scripts/lexical_slices/prose_ness_curve.py
    python scripts/lexical_slices/prose_ness_curve.py --models R01 R04
    python scripts/lexical_slices/prose_ness_curve.py --n-bins 5 --datasets music_caps song_describer
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_BASE = REPO_ROOT / "downstream_results"

# ── Model configuration ─────────────────────────────────────────────────────
TABLE1_MODELS = [
    "R01",  # tags+sounds baseline
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

MODEL_STYLES = {
    "R01": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
    "R02": {"color": "#ff7f0e", "linestyle": "--", "marker": "s"},
    "R03": {"color": "#2ca02c", "linestyle": "--", "marker": "^"},
    "R04": {"color": "#d62728", "linestyle": "-", "marker": "D"},
    "R05": {"color": "#9467bd", "linestyle": "-", "marker": "v"},
    "R14": {"color": "#8c564b", "linestyle": "-.", "marker": "p"},
}

DEFAULT_STYLE = {"color": "#7f7f7f", "linestyle": ":", "marker": "x"}

# ── Keyword lexicons (from EX1) ─────────────────────────────────────────────
AFFECT_NARRATIVE_KEYWORDS = {
    # narrative
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
    "gradually",
    "suddenly",
    "eventually",
    "finally",
    "initially",
    "progressively",
    "indicating",
    "serves",
    # affect
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
}

SIMILE_METAPHOR_MARKERS = {
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
    "imagery",
    "feels",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def compute_prose_features(query: str, adj_tagger=None) -> dict[str, float]:
    """Compute prose-ness features for a single query."""
    tokens = tokenize(query)
    n_tokens = max(len(tokens), 1)
    token_set = set(tokens)

    # 1. Length in tokens
    length = len(tokens)

    # 2. Adjective density (via spaCy POS if available)
    if adj_tagger is not None:
        doc = adj_tagger(query)
        n_adj = sum(1 for tok in doc if tok.pos_ == "ADJ")
        adj_density = n_adj / max(len(doc), 1)
    else:
        adj_density = 0.0

    # 3. Affect/narrative keyword count
    affect_narrative_count = len(token_set & AFFECT_NARRATIVE_KEYWORDS)

    # 4. Simile/metaphor marker count
    simile_count = sum(1 for t in tokens if t in SIMILE_METAPHOR_MARKERS)

    return {
        "length": length,
        "adj_density": adj_density,
        "affect_narrative_count": affect_narrative_count,
        "simile_count": simile_count,
    }


def zscore(values: np.ndarray) -> np.ndarray:
    """Z-score normalization. Returns zeros if std is 0."""
    std = values.std()
    if std < 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def load_caption2rank(model_id: str, dataset: str) -> list[dict]:
    path = RESULTS_BASE / model_id / dataset / "caption2rank.json"
    return json.loads(path.read_text())


def compute_mrr(ranks: list[int]) -> float:
    if not ranks:
        return float("nan")
    return sum(1.0 / (r + 1) for r in ranks) / len(ranks)


def main():
    ap = argparse.ArgumentParser(description="EX2: Prose-ness curve analysis")
    ap.add_argument(
        "--models",
        nargs="+",
        default=TABLE1_MODELS,
        help="Model IDs to evaluate",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["music_caps"],
        help="Datasets to evaluate on",
    )
    ap.add_argument(
        "--n-bins",
        type=int,
        default=5,
        help="Number of percentile bins (default: 5 = quintiles)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Output directory",
    )
    ap.add_argument(
        "--no-spacy",
        action="store_true",
        help="Skip spaCy POS tagging (uses 0 for adj_density)",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load spaCy ───────────────────────────────────────────────────────
    adj_tagger = None
    if not args.no_spacy:
        try:
            import spacy

            adj_tagger = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            print("Using spaCy for adjective density.")
        except (ImportError, OSError):
            print("spaCy not available; adjective density will be 0.")

    # ── Compute prose-ness scores ────────────────────────────────────────
    for dataset in args.datasets:
        print(f"\n{'=' * 80}")
        print(f"  Dataset: {dataset}")
        print(f"{'=' * 80}")

        # Load queries from first model (queries are identical across models)
        ref_data = load_caption2rank(args.models[0], dataset)

        # Compute features for each query
        feature_names = [
            "length",
            "adj_density",
            "affect_narrative_count",
            "simile_count",
        ]
        features_list = []
        queries_info = []

        print(f"  Computing prose-ness features for {len(ref_data)} queries...")
        for item in ref_data:
            feats = compute_prose_features(item["query"], adj_tagger)
            features_list.append(feats)
            queries_info.append({"index": item["index"], "query": item["query"][:200]})

        # Build feature matrix and compute z-scored prose-ness
        feat_matrix = np.array([[f[fn] for fn in feature_names] for f in features_list])
        z_matrix = np.column_stack(
            [zscore(feat_matrix[:, i]) for i in range(len(feature_names))]
        )
        prose_scores = z_matrix.sum(axis=1)

        # Assign percentile bins
        percentile_edges = np.linspace(0, 100, args.n_bins + 1)
        bin_thresholds = np.percentile(prose_scores, percentile_edges)
        bin_indices = np.digitize(prose_scores, bin_thresholds[1:-1])  # 0-indexed bins

        # Print score distribution
        print(f"\n  Prose-ness score distribution:")
        print(
            f"    min={prose_scores.min():.2f}  mean={prose_scores.mean():.2f}  "
            f"max={prose_scores.max():.2f}  std={prose_scores.std():.2f}"
        )
        print(f"\n  Feature contributions:")
        for i, fn in enumerate(feature_names):
            col = feat_matrix[:, i]
            print(
                f"    {fn:25s}: mean={col.mean():.3f}  std={col.std():.3f}  "
                f"min={col.min():.3f}  max={col.max():.3f}"
            )
        print(f"\n  Bin sizes:")
        for b in range(args.n_bins):
            n = (bin_indices == b).sum()
            lo = prose_scores[bin_indices == b].min() if n > 0 else 0
            hi = prose_scores[bin_indices == b].max() if n > 0 else 0
            print(
                f"    bin {b} (P{percentile_edges[b]:.0f}-P{percentile_edges[b + 1]:.0f}): "
                f"{n} queries  score=[{lo:.2f}, {hi:.2f}]"
            )

        # ── Compute MRR per bin per model ────────────────────────────────
        mrr_per_bin: dict[str, list[float]] = {}

        for model_id in args.models:
            data = load_caption2rank(model_id, dataset)
            rank_by_idx = {item["index"]: item["min_rank"] for item in data}

            bin_mrrs = []
            for b in range(args.n_bins):
                mask = bin_indices == b
                indices_in_bin = [
                    ref_data[i]["index"] for i in range(len(ref_data)) if mask[i]
                ]
                ranks = [rank_by_idx[idx] for idx in indices_in_bin]
                bin_mrrs.append(compute_mrr(ranks))
            mrr_per_bin[model_id] = bin_mrrs

        # ── Print table ──────────────────────────────────────────────────
        print(f"\n  MRR (×100) per prose-ness bin:")
        header = f"  {'Bin':>12s}"
        for model_id in args.models:
            name = MODEL_NAMES.get(model_id, model_id)
            header += f"  {name:>18s}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for b in range(args.n_bins):
            label = f"P{percentile_edges[b]:.0f}-P{percentile_edges[b + 1]:.0f}"
            row = f"  {label:>12s}"
            for model_id in args.models:
                row += f"  {mrr_per_bin[model_id][b] * 100:>18.2f}"
            print(row)
        print()

        # ── Delta table ──────────────────────────────────────────────────
        baseline_id = args.models[0]
        if len(args.models) > 1:
            print(f"  MRR delta vs {MODEL_NAMES.get(baseline_id, baseline_id)} (×100):")
            header = f"  {'Bin':>12s}"
            for model_id in args.models[1:]:
                name = MODEL_NAMES.get(model_id, model_id)
                header += f"  {name:>18s}"
            print(header)
            print("  " + "-" * (len(header) - 2))

            for b in range(args.n_bins):
                label = f"P{percentile_edges[b]:.0f}-P{percentile_edges[b + 1]:.0f}"
                row = f"  {label:>12s}"
                base_mrr = mrr_per_bin[baseline_id][b]
                for model_id in args.models[1:]:
                    delta = (mrr_per_bin[model_id][b] - base_mrr) * 100
                    sign = "+" if delta >= 0 else ""
                    row += f"  {sign}{delta:>17.2f}"
                print(row)
            print()

        # ── Plot ─────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))

        bin_centers = [
            (percentile_edges[b] + percentile_edges[b + 1]) / 2
            for b in range(args.n_bins)
        ]

        for model_id in args.models:
            style = MODEL_STYLES.get(model_id, DEFAULT_STYLE)
            name = MODEL_NAMES.get(model_id, model_id)
            mrrs = [m * 100 for m in mrr_per_bin[model_id]]
            ax.plot(
                bin_centers,
                mrrs,
                label=name,
                **style,
                markersize=7,
                linewidth=2,
            )

        ax.set_xlabel("Prose-ness percentile", fontsize=12)
        ax.set_ylabel("MRR (×100)", fontsize=12)
        ax.set_title(f"MRR vs prose-ness — {dataset}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add bin size annotation
        bin_sizes = [(bin_indices == b).sum() for b in range(args.n_bins)]
        ax.set_xticks(bin_centers)
        ax.set_xticklabels(
            [
                f"P{percentile_edges[b]:.0f}-{percentile_edges[b + 1]:.0f}\n(n={bin_sizes[b]})"
                for b in range(args.n_bins)
            ],
            fontsize=9,
        )

        plt.tight_layout()
        fig_path = args.out_dir / f"prose_ness_curve_{dataset}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  Wrote {fig_path}")

        # ── Also plot delta curve ────────────────────────────────────────
        if len(args.models) > 1:
            fig2, ax2 = plt.subplots(figsize=(8, 5))

            for model_id in args.models[1:]:
                style = MODEL_STYLES.get(model_id, DEFAULT_STYLE)
                name = MODEL_NAMES.get(model_id, model_id)
                deltas = [
                    (mrr_per_bin[model_id][b] - mrr_per_bin[baseline_id][b]) * 100
                    for b in range(args.n_bins)
                ]
                ax2.plot(
                    bin_centers,
                    deltas,
                    label=f"{name} vs {MODEL_NAMES.get(baseline_id, baseline_id)}",
                    **style,
                    markersize=7,
                    linewidth=2,
                )

            ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax2.set_xlabel("Prose-ness percentile", fontsize=12)
            ax2.set_ylabel(
                f"MRR delta vs {MODEL_NAMES.get(baseline_id, baseline_id)} (×100)",
                fontsize=12,
            )
            ax2.set_title(f"MRR gain vs prose-ness — {dataset}", fontsize=13)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

            ax2.set_xticks(bin_centers)
            ax2.set_xticklabels(
                [
                    f"P{percentile_edges[b]:.0f}-{percentile_edges[b + 1]:.0f}\n(n={bin_sizes[b]})"
                    for b in range(args.n_bins)
                ],
                fontsize=9,
            )

            plt.tight_layout()
            fig2_path = args.out_dir / f"prose_ness_delta_{dataset}.png"
            fig2.savefig(fig2_path, dpi=150)
            plt.close(fig2)
            print(f"  Wrote {fig2_path}")

    # ── Save per-query scores ────────────────────────────────────────────
    # Save for the last dataset processed (typically music_caps)
    scores_out = []
    for i, qi in enumerate(queries_info):
        scores_out.append(
            {
                "index": qi["index"],
                "query": qi["query"],
                "prose_score": round(float(prose_scores[i]), 4),
                "bin": int(bin_indices[i]),
                "features": {
                    k: round(float(v), 4) for k, v in features_list[i].items()
                },
            }
        )
    scores_path = args.out_dir / "prose_ness_scores.json"
    scores_path.write_text(json.dumps(scores_out, indent=2))
    print(f"\nWrote {scores_path}")


if __name__ == "__main__":
    main()
