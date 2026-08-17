"""Check coverage of EVALUATIVE_WORDS lexicon against actual SongD queries.

Strategy: take SongD queries, POS-tag with a lightweight heuristic (suffix-based
since no spaCy on cluster), and print the most frequent adjective-like tokens
that are NOT already in EVALUATIVE_WORDS. Also print the most frequent tokens
overall so we can eyeball taggy words to exclude.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from explore_stratified import EVALUATIVE_WORDS, TAGGY_SIGNALS, tokenize

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_BASE = REPO_ROOT / "downstream_results"

BASELINE = "R01"
REVIEW = "R04"

# Suffixes common to English adjectives / reviewer register
ADJ_SUFFIXES = (
    "ous",
    "ful",
    "less",
    "ive",
    "ish",
    "al",
    "ic",
    "y",
    "ly",
    "ing",
    "ed",
    "en",
    "esque",
)

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "in",
    "on",
    "at",
    "to",
    "is",
    "it",
    "its",
    "this",
    "that",
    "with",
    "for",
    "by",
    "as",
    "be",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "but",
    "so",
    "not",
    "no",
    "from",
    "into",
    "song",
    "track",
    "music",
    "piece",
    "vibe",
    "sound",
    "sounds",
    "sounding",
    "style",
    "feel",
    "feels",
    "feeling",
    "like",
    "you",
    "your",
    "we",
    "us",
    "one",
    "some",
    "any",
    "all",
    "more",
    "very",
    "also",
    "just",
    "only",
    "very",
    "pretty",
    "quite",
    "rather",
    "really",
    "can",
    "could",
    "would",
    "should",
    "may",
    "might",
    "will",
    "there",
    "here",
    "when",
    "while",
    "during",
    "over",
    "under",
    "through",
    "about",
    "around",
    "along",
}


def load_queries(model: str, dataset: str) -> list[str]:
    path = RESULTS_BASE / model / dataset / "caption2rank.json"
    with open(path) as f:
        data = json.load(f)
    return [r["query"] for r in data]


def looks_adj(tok: str) -> bool:
    if len(tok) < 4:
        return False
    return tok.endswith(ADJ_SUFFIXES)


def main() -> None:
    for ds in ("song_describer", "music_caps"):
        queries = load_queries(REVIEW, ds)
        tokens = Counter()
        for q in queries:
            for t in tokenize(q):
                if t in STOPWORDS or t in TAGGY_SIGNALS:
                    continue
                tokens[t] += 1

        adj_like = Counter({t: c for t, c in tokens.items() if looks_adj(t)})
        in_lex = Counter({t: c for t, c in adj_like.items() if t in EVALUATIVE_WORDS})
        missing = Counter(
            {t: c for t, c in adj_like.items() if t not in EVALUATIVE_WORDS}
        )

        total_adj = sum(adj_like.values())
        covered = sum(in_lex.values())
        print(f"\n=== {ds}   queries={len(queries)} ===")
        print(
            f"adj-like tokens: total={total_adj}  covered_by_lex={covered} ({100 * covered / max(1, total_adj):.1f}%)"
        )

        print(
            f"\n  TOP 40 adj-like NOT in EVALUATIVE_WORDS (candidates to add or exclude):"
        )
        for t, c in missing.most_common(40):
            print(f"    {c:4d}  {t}")

        print(f"\n  TOP 20 adj-like ALREADY in EVALUATIVE_WORDS:")
        for t, c in in_lex.most_common(20):
            print(f"    {c:4d}  {t}")


if __name__ == "__main__":
    main()
