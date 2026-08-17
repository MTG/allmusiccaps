"""Rare-vocabulary retrieval split (Experiment E4).

For each MusicCaps and SongDescriber query, tag every token as *common* if
it appears in the combined non-review training vocabulary (struct ∪ r4/M4-RAG
∪ MSD ∪ Freesound ∪ PSE) and *rare* otherwise.  Compute each query's
rare-token fraction, split queries into quantiles, and recompute MRR per
quantile from the existing per-query rank data (caption2rank.json) — no
model inference needed.

Quotes is deliberately excluded from the "common" vocab so that tokens
appearing *only* in quotes surface as "rare".

Usage (on the cluster, clap env):
    source /projects/<group>/envs/clap/bin/activate
    python rare_vocab_split.py --out-path rare_vocab_split.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from text_corpus_stats import (
    DEFAULTS,
    _load_ids_freesound,
    _load_ids_stem,
    iter_freesound_texts,
    iter_m4rag_texts,
    iter_msd_texts,
    iter_pse_texts,
    iter_struct_texts,
    tokenize,
    vocab_from_texts,
)

# ---------------------------------------------------------------------------
# Model registry (Table 1 models)
# ---------------------------------------------------------------------------

MODEL_IDS = {
    "quotes+mu+so": "R04",
    "struct+mu+so": "R05",
    "struct_only": "R03",
    "quotes_only": "R02",
}

BENCHMARKS = {
    "music_caps": "music_caps",
    "song_describer": "song_describer",
}

DEFAULT_RESULTS_BASE = "/scratch/<group>/downstream_results"


# ---------------------------------------------------------------------------
# Build common (non-review) vocabulary
# ---------------------------------------------------------------------------


def build_common_vocab(args: argparse.Namespace) -> set[str]:
    """Build vocab from struct ∪ r4 ∪ msd ∪ freesound ∪ pse (no quotes)."""

    print("Loading training filelist IDs...", flush=True)
    struct_ids = _load_ids_stem(args.struct_filelist)
    print(f"  struct: {len(struct_ids):,} IDs", flush=True)
    r4_ids = _load_ids_stem(args.r4_filelist)
    print(f"  r4: {len(r4_ids):,} IDs", flush=True)
    msd_ids = _load_ids_stem(args.msd_filelist)
    print(f"  msd: {len(msd_ids):,} IDs", flush=True)
    freesound_ids = _load_ids_freesound(args.freesound_filelist)
    print(f"  freesound: {len(freesound_ids):,} IDs", flush=True)

    corpora = {
        "struct": lambda: iter_struct_texts(args.struct_text_file, struct_ids),
        "r4": lambda: iter_m4rag_texts(args.m4rag_metadata_file, r4_ids),
        "msd": lambda: iter_msd_texts(msd_ids),
        "freesound": lambda: iter_freesound_texts(
            args.freesound_text_file, freesound_ids
        ),
        "pse": lambda: iter_pse_texts(args.pse_filelist),
    }

    common_vocab: set[str] = set()
    for name, iter_fn in corpora.items():
        print(f"  Building vocab from {name}...", flush=True)
        counter, total = vocab_from_texts(iter_fn())
        corpus_vocab = set(counter.keys())
        print(f"    {name}: {total:,} tokens, {len(corpus_vocab):,} types", flush=True)
        common_vocab |= corpus_vocab

    print(
        f"\nCommon vocabulary (struct ∪ r4 ∪ msd ∪ fs ∪ pse): "
        f"{len(common_vocab):,} types",
        flush=True,
    )
    return common_vocab


# ---------------------------------------------------------------------------
# Query analysis
# ---------------------------------------------------------------------------


def analyze_queries(queries: list[dict], common_vocab: set[str]) -> list[dict]:
    """Tokenize each query and compute its rare-token fraction.

    Parameters
    ----------
    queries : list[dict]
        Each entry is a dict from caption2rank.json:
        {"index": int, "query": str, "targets": list[int], "min_rank": int}
    common_vocab : set[str]
        Tokens considered "common" (from non-review corpora).

    Returns
    -------
    list[dict]
        Same entries augmented with "tokens", "n_tokens", "n_rare",
        "rare_frac", "rare_tokens".
    """
    for q in queries:
        tokens = tokenize(q["query"])
        rare = [t for t in tokens if t not in common_vocab]
        n = len(tokens)
        q["tokens"] = tokens
        q["n_tokens"] = n
        q["n_rare"] = len(rare)
        q["rare_frac"] = len(rare) / n if n > 0 else 0.0
        q["rare_tokens"] = rare
    return queries


def bin_queries(queries: list[dict], n_bins: int = 4) -> list[tuple[str, list[dict]]]:
    """Bin queries into quantiles by rare_frac.

    Returns a list of (bin_label, queries_in_bin) tuples.
    The first bin (Q1) contains queries with rare_frac == 0 (all-common).
    The remaining bins split the queries with rare_frac > 0 into
    equal-count groups.
    """
    # Separate all-common queries from those with at least one rare token
    all_common = [q for q in queries if q["rare_frac"] == 0.0]
    has_rare = sorted(
        [q for q in queries if q["rare_frac"] > 0.0],
        key=lambda q: q["rare_frac"],
    )

    bins: list[tuple[str, list[dict]]] = []
    bins.append(("Q1_all_common", all_common))

    if has_rare:
        # Split the remaining queries into (n_bins - 1) equal-count groups
        remaining_bins = n_bins - 1
        chunk_size = len(has_rare) / remaining_bins
        for i in range(remaining_bins):
            start = int(i * chunk_size)
            end = int((i + 1) * chunk_size) if i < remaining_bins - 1 else len(has_rare)
            subset = has_rare[start:end]
            if subset:
                low = subset[0]["rare_frac"]
                high = subset[-1]["rare_frac"]
                label = f"Q{i + 2}_rare_{low:.3f}_{high:.3f}"
                bins.append((label, subset))

    return bins


def compute_mrr(queries: list[dict]) -> float:
    """Compute Mean Reciprocal Rank from queries with min_rank field.

    Note: ``min_rank`` is 0-indexed (first correct hit → 0) as produced by
    ``src/downstream/retrieval/metrics.py::median_rank``, so the reciprocal
    rank is ``1 / (min_rank + 1)``.
    """
    if not queries:
        return 0.0
    return sum(1.0 / (q["min_rank"] + 1) for q in queries) / len(queries)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Text files
    parser.add_argument(
        "--struct-text-file",
        default=DEFAULTS["struct_text_file"],
    )
    parser.add_argument(
        "--m4rag-metadata-file",
        default=DEFAULTS["m4rag_metadata_file"],
    )
    parser.add_argument(
        "--freesound-text-file",
        default=DEFAULTS["freesound_text_file"],
    )
    parser.add_argument(
        "--pse-filelist",
        default=DEFAULTS["pse_filelist"],
    )
    # Training filelists
    parser.add_argument(
        "--struct-filelist",
        default=DEFAULTS["struct_filelist"],
    )
    parser.add_argument(
        "--r4-filelist",
        default=DEFAULTS["r4_filelist"],
    )
    parser.add_argument(
        "--msd-filelist",
        default=DEFAULTS["msd_filelist"],
    )
    parser.add_argument(
        "--freesound-filelist",
        default=DEFAULTS["freesound_filelist"],
    )
    # Results
    parser.add_argument(
        "--results-base",
        default=DEFAULT_RESULTS_BASE,
        help="Base path for downstream_results/",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=4,
        help="Number of quantile bins (default: 4)",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("rare_vocab_split.json"),
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("rare_vocab_split.png"),
        help="Path for the MRR-vs-quantile plot. Pass an empty string to skip.",
    )
    args = parser.parse_args()

    # Step 1: Build common vocabulary (no quotes)
    print("=" * 70)
    print("STEP 1: Building common (non-review) vocabulary")
    print("=" * 70)
    common_vocab = build_common_vocab(args)

    # Step 2: Process each benchmark
    report: dict = {
        "common_vocab_size": len(common_vocab),
        "n_bins": args.n_bins,
        "benchmarks": {},
    }

    for bench_key, bench_dir in BENCHMARKS.items():
        print(f"\n{'=' * 70}")
        print(f"STEP 2: Processing {bench_key}")
        print(f"{'=' * 70}")

        # Load queries from any model's caption2rank.json (queries are the same)
        first_model = next(iter(MODEL_IDS.values()))
        c2r_path = (
            Path(args.results_base) / first_model / bench_dir / "caption2rank.json"
        )
        print(f"  Loading queries from {c2r_path}", flush=True)

        with open(c2r_path) as f:
            queries = json.load(f)
        print(f"  {len(queries)} queries loaded", flush=True)

        # Analyze queries
        queries = analyze_queries(queries, common_vocab)

        n_with_rare = sum(1 for q in queries if q["rare_frac"] > 0)
        avg_rare_frac = sum(q["rare_frac"] for q in queries) / len(queries)
        print(
            f"  Queries with at least 1 rare token: {n_with_rare}/{len(queries)} "
            f"({n_with_rare / len(queries):.1%})",
            flush=True,
        )
        print(f"  Average rare fraction: {avg_rare_frac:.4f}", flush=True)

        # Bin queries
        bins = bin_queries(queries, n_bins=args.n_bins)
        print(f"\n  Bins:", flush=True)
        for label, bin_queries_list in bins:
            if bin_queries_list:
                avg_rf = sum(q["rare_frac"] for q in bin_queries_list) / len(
                    bin_queries_list
                )
                print(
                    f"    {label}: {len(bin_queries_list)} queries, "
                    f"avg rare_frac={avg_rf:.4f}",
                    flush=True,
                )

        # Step 3: Compute MRR per quantile per model
        bench_results: dict = {
            "n_queries": len(queries),
            "n_with_rare": n_with_rare,
            "avg_rare_frac": round(avg_rare_frac, 4),
            "bins": {},
            "model_mrr": {},
        }

        # Store bin info
        for label, bin_queries_list in bins:
            bench_results["bins"][label] = {
                "n_queries": len(bin_queries_list),
                "avg_rare_frac": round(
                    sum(q["rare_frac"] for q in bin_queries_list)
                    / len(bin_queries_list),
                    4,
                )
                if bin_queries_list
                else 0.0,
                "query_indices": [q["index"] for q in bin_queries_list],
            }

        # Compute MRR per model per bin
        print(f"\n  MRR per model per bin:", flush=True)

        # Print header
        bin_labels = [label for label, _ in bins]
        header = (
            f"  {'Model':<20}"
            + "".join(f"{bl:>25}" for bl in bin_labels)
            + f"{'Overall':>12}"
        )
        print(header, flush=True)
        print(f"  {'-' * (len(header) - 2)}", flush=True)

        for model_name, model_id in MODEL_IDS.items():
            c2r_path = (
                Path(args.results_base) / model_id / bench_dir / "caption2rank.json"
            )
            with open(c2r_path) as f:
                model_queries = json.load(f)

            # Index by query index for lookup
            model_rank_by_index = {q["index"]: q["min_rank"] for q in model_queries}

            model_bin_mrr: dict[str, float] = {}
            row = f"  {model_name:<20}"

            for label, bin_queries_list in bins:
                # Map bin queries to this model's ranks
                bin_with_ranks = []
                for q in bin_queries_list:
                    rank = model_rank_by_index.get(q["index"])
                    if rank is not None:
                        bin_with_ranks.append({"min_rank": rank})

                mrr = compute_mrr(bin_with_ranks)
                model_bin_mrr[label] = round(mrr, 4)
                row += f"{mrr:>25.4f}"

            # Overall MRR
            overall_mrr = compute_mrr(
                [{"min_rank": model_rank_by_index[q["index"]]} for q in queries]
            )
            row += f"{overall_mrr:>12.4f}"
            print(row, flush=True)

            bench_results["model_mrr"][model_name] = {
                "model_id": model_id,
                "per_bin": model_bin_mrr,
                "overall": round(overall_mrr, 4),
            }

        # Store some example rare queries
        rare_queries = sorted(queries, key=lambda q: q["rare_frac"], reverse=True)[:10]
        bench_results["example_rare_queries"] = [
            {
                "index": q["index"],
                "query": q["query"],
                "rare_frac": round(q["rare_frac"], 4),
                "rare_tokens": q["rare_tokens"],
            }
            for q in rare_queries
        ]

        report["benchmarks"][bench_key] = bench_results

    # Write output
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {args.out_path}")

    # Plot MRR vs rare-vocab quantile
    if str(args.plot_path):
        plot_mrr_curves(report, args.plot_path)


def plot_mrr_curves(report: dict, plot_path: Path) -> None:
    """One figure per benchmark, one line per model, x=bin, y=MRR."""
    import matplotlib.pyplot as plt

    benchmarks = report["benchmarks"]
    n = len(benchmarks)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (bench_key, bench) in zip(axes, benchmarks.items()):
        # Skip empty bins so that the plot isn't anchored at (Q1, 0) when no
        # queries landed in that bin.
        bin_labels = [
            label for label, info in bench["bins"].items() if info["n_queries"] > 0
        ]
        x = list(range(len(bin_labels)))
        for model_name, mrec in bench["model_mrr"].items():
            y = [mrec["per_bin"][label] for label in bin_labels]
            ax.plot(x, y, marker="o", label=model_name)
        ax.set_xticks(x)
        # Short labels: Q1, Q2, ... — full label in the JSON
        ax.set_xticklabels([label.split("_")[0] for label in bin_labels])
        ax.set_xlabel("Rare-token quantile (low → high)")
        ax.set_ylabel("MRR")
        ax.set_title(bench_key)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot written to {plot_path}")


if __name__ == "__main__":
    main()
