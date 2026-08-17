"""Rank-divergence analysis: where do quotes-trained and struct-trained models disagree?

Reframed E4 follow-up. Instead of asking "is quotes better?", we ask:
"For which queries do the two models rank the ground-truth track very
differently, and is there a linguistic pattern?"

Inputs (already on disk locally, no inference needed):
    downstream_results/<model_id>/{music_caps,song_describer}/caption2rank.json

Each entry is `{"index": int, "query": str, "targets": [int], "min_rank": int}`
where ``min_rank`` is 0-indexed. We pair models and compute, per query,

    delta = rank(struct) - rank(quotes)            # lower rank = better
            > 0  → quotes wins
            < 0  → struct wins

We then bucket queries (top-K quotes-favors / struct-favors / agree) and run
simple linguistic descriptors on each bucket: token counts, top-distinctive
words via odds ratio against the rest of the dataset.

Usage:
    python rank_divergence.py                      # writes rank_divergence/
    python rank_divergence.py --top-k 100
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# Pairings to compare (text-source contrast, holding aux constant)
PAIRS = [
    ("aux_data", "R04", "R05"),  # quotes+mu+so vs struct+mu+so
    ("no_aux", "R02", "R03"),  # quotes only   vs struct only
]

DATASETS = ["music_caps", "song_describer"]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_BASE = REPO_ROOT / "downstream_results"

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "by",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "song",
    "track",
    "music",
    "sound",
    "sounds",
    "playing",
    "plays",
    "played",
    "audio",
    "recording",
    "features",
}


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def load_ranks(model_id: str, dataset: str) -> dict[int, dict]:
    path = RESULTS_BASE / model_id / dataset / "caption2rank.json"
    items = json.loads(path.read_text())
    return {it["index"]: it for it in items}


def odds_ratio_top_terms(
    in_texts: list[str], out_texts: list[str], top_n: int = 25
) -> list[tuple[str, float, int, int]]:
    """Return top distinctive terms in `in_texts` vs `out_texts` by odds ratio.

    Uses add-one smoothing on the contingency. Filters out stopwords and tokens
    that appear <3 times in either bucket. Returns (term, log_odds, in_count, out_count).
    """
    in_counter = Counter(
        t for txt in in_texts for t in set(tokenize(txt))
    )  # doc-frequency
    out_counter = Counter(t for txt in out_texts for t in set(tokenize(txt)))
    n_in = max(len(in_texts), 1)
    n_out = max(len(out_texts), 1)

    scored = []
    vocab = set(in_counter) | set(out_counter)
    for term in vocab:
        if term in STOPWORDS or len(term) <= 2:
            continue
        a = in_counter.get(term, 0)
        b = out_counter.get(term, 0)
        if a < 3:
            continue
        # log odds with add-1 smoothing
        p_in = (a + 1) / (n_in + 2)
        p_out = (b + 1) / (n_out + 2)
        lo = (p_in / (1 - p_in)) / (p_out / (1 - p_out))
        scored.append((term, lo, a, b))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def summarize_bucket(name: str, items: list[dict]) -> dict:
    if not items:
        return {"name": name, "n": 0}
    queries = [it["query"] for it in items]
    tokens = [tokenize(q) for q in queries]
    lengths = [len(t) for t in tokens]
    type_counts = [len(set(t)) for t in tokens]
    return {
        "name": name,
        "n": len(items),
        "mean_length": round(sum(lengths) / len(lengths), 2),
        "mean_unique_types": round(sum(type_counts) / len(type_counts), 2),
        "mean_abs_delta": round(sum(abs(it["delta"]) for it in items) / len(items), 1),
        "examples": [
            {
                "delta": it["delta"],
                "rank_quotes": it["rank_quotes"],
                "rank_struct": it["rank_struct"],
                "query": it["query"][:280],
            }
            for it in items[:5]
        ],
    }


def analyze_pair(
    pair_name: str, quotes_id: str, struct_id: str, dataset: str, top_k: int
) -> dict:
    q = load_ranks(quotes_id, dataset)
    s = load_ranks(struct_id, dataset)
    common = sorted(set(q) & set(s))

    rows = []
    for idx in common:
        rq = q[idx]["min_rank"]
        rs = s[idx]["min_rank"]
        rows.append(
            {
                "index": idx,
                "query": q[idx]["query"],
                "rank_quotes": rq,
                "rank_struct": rs,
                "delta": rs - rq,  # >0: quotes wins (lower rank), <0: struct wins
            }
        )

    # Sort
    by_quotes_wins = sorted(rows, key=lambda r: r["delta"], reverse=True)
    by_struct_wins = sorted(rows, key=lambda r: r["delta"])
    by_agree = sorted(rows, key=lambda r: (abs(r["delta"]), r["rank_quotes"]))

    quotes_bucket = by_quotes_wins[:top_k]
    struct_bucket = by_struct_wins[:top_k]
    agree_bucket = by_agree[:top_k]

    quotes_texts = [r["query"] for r in quotes_bucket]
    struct_texts = [r["query"] for r in struct_bucket]
    rest_texts = [r["query"] for r in rows]  # full corpus as background

    # Distinctive terms (each bucket vs the *full* dataset)
    quotes_terms = odds_ratio_top_terms(quotes_texts, rest_texts)
    struct_terms = odds_ratio_top_terms(struct_texts, rest_texts)
    # Also: quotes bucket vs struct bucket directly
    quotes_vs_struct = odds_ratio_top_terms(quotes_texts, struct_texts)
    struct_vs_quotes = odds_ratio_top_terms(struct_texts, quotes_texts)

    deltas = [r["delta"] for r in rows]
    abs_deltas = [abs(d) for d in deltas]
    return {
        "pair": pair_name,
        "quotes_model": quotes_id,
        "struct_model": struct_id,
        "dataset": dataset,
        "n_queries": len(rows),
        "delta_stats": {
            "mean": round(sum(deltas) / len(deltas), 2),
            "mean_abs": round(sum(abs_deltas) / len(abs_deltas), 2),
            "frac_quotes_wins": round(sum(1 for d in deltas if d > 0) / len(deltas), 4),
            "frac_struct_wins": round(sum(1 for d in deltas if d < 0) / len(deltas), 4),
            "frac_tied": round(sum(1 for d in deltas if d == 0) / len(deltas), 4),
        },
        "buckets": {
            "quotes_favors": summarize_bucket("quotes_favors", quotes_bucket),
            "struct_favors": summarize_bucket("struct_favors", struct_bucket),
            "agree": summarize_bucket("agree", agree_bucket),
        },
        "distinctive_terms": {
            "quotes_favors_vs_all": [
                {"term": t, "log_odds": round(lo, 3), "in": a, "out": b}
                for t, lo, a, b in quotes_terms
            ],
            "struct_favors_vs_all": [
                {"term": t, "log_odds": round(lo, 3), "in": a, "out": b}
                for t, lo, a, b in struct_terms
            ],
            "quotes_favors_vs_struct_favors": [
                {"term": t, "log_odds": round(lo, 3), "in": a, "out": b}
                for t, lo, a, b in quotes_vs_struct
            ],
            "struct_favors_vs_quotes_favors": [
                {"term": t, "log_odds": round(lo, 3), "in": a, "out": b}
                for t, lo, a, b in struct_vs_quotes
            ],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Bucket size for divergent / agreeing queries",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT
        / "scripts"
        / "text_corpus_vocabulary_coverage"
        / "rank_divergence",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for pair_name, quotes_id, struct_id in PAIRS:
        for dataset in DATASETS:
            res = analyze_pair(pair_name, quotes_id, struct_id, dataset, args.top_k)
            all_results.append(res)
            out_path = args.out_dir / f"{pair_name}__{dataset}.json"
            out_path.write_text(json.dumps(res, indent=2))
            print(f"wrote {out_path}")

    # Console summary
    print()
    print("=" * 100)
    print(
        f"{'Pair':<10} {'Dataset':<16} {'N':>5} {'mean Δ':>8} {'mean|Δ|':>8} {'%qwin':>6} {'%swin':>6} {'%tied':>6}"
    )
    print("=" * 100)
    for r in all_results:
        ds = r["delta_stats"]
        print(
            f"{r['pair']:<10} {r['dataset']:<16} {r['n_queries']:>5} "
            f"{ds['mean']:>8.2f} {ds['mean_abs']:>8.2f} "
            f"{ds['frac_quotes_wins'] * 100:>5.1f}% {ds['frac_struct_wins'] * 100:>5.1f}% "
            f"{ds['frac_tied'] * 100:>5.1f}%"
        )


if __name__ == "__main__":
    main()
