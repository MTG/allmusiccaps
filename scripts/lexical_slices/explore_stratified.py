"""Stratified exploration to inform a new lexical taxonomy.

Dimensions we stratify by (each computed per-query, then bucketed):
  - length (word count)
  - tag-likeness: fraction of tokens that are short nouns/adjectives from a
    tag-ish vocabulary, or comma/slash density (proxy for tag lists)
  - evaluative-word density: fraction of tokens matching a broad reviewer-
    evaluative adjective lexicon (idiosyncratic, sentimental, anguished, ...)
  - sentence count
  - presence of era/decade tokens ("70s", "90s", "early 2010s", ...)
  - presence of functional phrases ("perfect for", "soundtrack", "background
    music for", "suitable for", "fit for")
  - presence of instrument-list pattern (many comma-separated instruments)

For each stratum we print:
  - bucket edges + count
  - mean Δrank (baseline - review); positive = reviews help
  - mean baseline MRR (reciprocal of min_rank) and review MRR
so we can see which strata review supervision actually moves.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_BASE = REPO_ROOT / "downstream_results"

BASELINE = "R01"
REVIEW = "R04"
DATASETS = ["music_caps", "song_describer"]

# -- lexicons --------------------------------------------------------------

ERA_RE = re.compile(
    r"\b(?:"
    r"\d{2}s|\d{4}s|\d{4}|"
    r"(?:early|mid|late)[\s\-]+(?:\d{2}s|\d{4}s|\d{4})|"
    r"(?:19|20)\d{2}s?"
    r")\b",
    re.IGNORECASE,
)

FUNCTIONAL_PHRASES = [
    "perfect for",
    "suitable for",
    "would fit",
    "fit for",
    "background music",
    "soundtrack for",
    "soundtrack to",
    "backing track",
    "theme song",
    "used for",
    "fits a",
    "can be used",
    "used in",
    "track for",
    "music for a",
    "music for an",
]

EVALUATIVE_WORDS = {
    # reviewer-style subjective adjectives & nouns
    "idiosyncratic",
    "sentimental",
    "anguished",
    "whimsical",
    "expressive",
    "evocative",
    "haunting",
    "brooding",
    "yearning",
    "wistful",
    "dreamy",
    "nostalgic",
    "melancholic",
    "melancholy",
    "ethereal",
    "unsettling",
    "intimate",
    "contemplative",
    "triumphant",
    "hopeful",
    "desperate",
    "bittersweet",
    "cinematic",
    "atmospheric",
    "lush",
    "sparse",
    "raw",
    "polished",
    "gritty",
    "grimy",
    "cathartic",
    "euphoric",
    "dissonant",
    "jagged",
    "shimmering",
    "pulsating",
    "driving",
    "propulsive",
    "urgent",
    "meditative",
    "trance-like",
    "hypnotic",
    "tense",
    "eerie",
    "ominous",
    "sinister",
    "mysterious",
    "romantic",
    "passionate",
    "aggressive",
    "joyful",
    "playful",
    "serene",
    "peaceful",
    "somber",
    "dramatic",
    "groovy",
    "funky",
    "slick",
    "snappy",
    "boisterous",
    "soothing",
    "uplifting",
    "chaotic",
    "elegant",
    "understated",
    "opulent",
    "stark",
    "fragile",
    "volatile",
    "visceral",
    "ragged",
    "luminous",
    "heartfelt",
    "poignant",
    "tender",
    "abrasive",
    "menacing",
    "playful",
    "spirited",
    "inspired",
    "introspective",
    "buoyant",
}

# Phrases that mark era/lineage references beyond the regex
LINEAGE_PHRASES = [
    "reminiscent of",
    "in the style of",
    "style of",
    "similar to",
    "influenced by",
    "evokes",
    "like a",
]

TAGGY_SIGNALS = {
    # near-tag words often in MuCaps enumerations; not evaluative
    "male",
    "female",
    "vocal",
    "vocals",
    "tempo",
    "medium",
    "fast",
    "slow",
    "instrumental",
    "drums",
    "bass",
    "guitar",
    "piano",
    "keyboard",
    "synth",
    "acoustic",
    "electric",
    "song",
    "track",
    "audio",
    "recording",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")


def tokenize(s: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(s)]


# -- feature extraction ----------------------------------------------------


def features(q: str) -> dict:
    tokens = tokenize(q)
    n_tok = len(tokens) or 1
    n_sent = max(1, q.count(".") + q.count("!") + q.count("?"))
    n_commas = q.count(",")
    qlow = q.lower()

    eval_hits = sum(1 for t in tokens if t in EVALUATIVE_WORDS)
    taggy_hits = sum(1 for t in tokens if t in TAGGY_SIGNALS)

    era_hit = bool(ERA_RE.search(q))
    lineage_hit = any(p in qlow for p in LINEAGE_PHRASES)
    functional_hit = any(p in qlow for p in FUNCTIONAL_PHRASES)

    return {
        "len": n_tok,
        "sent": n_sent,
        "commas": n_commas,
        "eval_density": eval_hits / n_tok,
        "eval_hits": eval_hits,
        "taggy_density": taggy_hits / n_tok,
        "era": era_hit,
        "lineage": lineage_hit or era_hit,
        "functional": functional_hit,
    }


# -- I/O -------------------------------------------------------------------


def load_ranks(model: str, dataset: str) -> dict[int, dict]:
    path = RESULTS_BASE / model / dataset / "caption2rank.json"
    with open(path) as f:
        data = json.load(f)
    return {r["index"]: r for r in data}


def merged(dataset: str) -> list[dict]:
    b = load_ranks(BASELINE, dataset)
    r = load_ranks(REVIEW, dataset)
    out = []
    for idx in sorted(set(b) & set(r)):
        q = b[idx]["query"]
        out.append(
            {
                "idx": idx,
                "q": q,
                "br": b[idx]["min_rank"],
                "rr": r[idx]["min_rank"],
                "delta": b[idx]["min_rank"] - r[idx]["min_rank"],
                "mrr_base": 1.0 / (b[idx]["min_rank"] + 1),
                "mrr_rev": 1.0 / (r[idx]["min_rank"] + 1),
                "f": features(q),
            }
        )
    return out


# -- reporting -------------------------------------------------------------


def report_bucket(name: str, rows: list[dict]) -> None:
    if not rows:
        print(f"  {name:<28}  (empty)")
        return
    n = len(rows)
    md = statistics.mean(r["delta"] for r in rows)
    mb = statistics.mean(r["mrr_base"] for r in rows) * 100
    mr = statistics.mean(r["mrr_rev"] for r in rows) * 100
    print(
        f"  {name:<28}  n={n:4d}  mean Δrank={md:+7.1f}   MRR base→rev: {mb:5.2f} → {mr:5.2f}   ΔMRR={mr - mb:+5.2f}"
    )


def stratify_length(rows: list[dict]) -> None:
    print("\n[length buckets, by word count]")
    buckets = [(0, 10), (11, 20), (21, 40), (41, 80), (81, 10**6)]
    for lo, hi in buckets:
        sel = [r for r in rows if lo <= r["f"]["len"] <= hi]
        report_bucket(f"len [{lo}..{hi}]", sel)


def stratify_eval_density(rows: list[dict]) -> None:
    print("\n[evaluative-word density]")
    cuts = [
        ("eval=0", lambda r: r["f"]["eval_hits"] == 0),
        ("eval=1", lambda r: r["f"]["eval_hits"] == 1),
        ("eval=2", lambda r: r["f"]["eval_hits"] == 2),
        ("eval>=3", lambda r: r["f"]["eval_hits"] >= 3),
    ]
    for name, pred in cuts:
        report_bucket(name, [r for r in rows if pred(r)])


def stratify_era(rows: list[dict]) -> None:
    print("\n[era / decade token present]")
    report_bucket("era yes", [r for r in rows if r["f"]["era"]])
    report_bucket("era no", [r for r in rows if not r["f"]["era"]])


def stratify_lineage(rows: list[dict]) -> None:
    print("\n[lineage (era OR 'reminiscent of' / 'style of' / 'like a')]")
    report_bucket("lineage yes", [r for r in rows if r["f"]["lineage"]])
    report_bucket("lineage no", [r for r in rows if not r["f"]["lineage"]])


def stratify_functional(rows: list[dict]) -> None:
    print("\n[functional phrase ('perfect for', 'soundtrack', 'background music')]")
    report_bucket("functional yes", [r for r in rows if r["f"]["functional"]])
    report_bucket("functional no", [r for r in rows if not r["f"]["functional"]])


def stratify_taggy(rows: list[dict]) -> None:
    print("\n[taggy density (tag-like nouns/adj)]")
    cuts = [
        ("taggy 0-0.10", lambda r: r["f"]["taggy_density"] < 0.10),
        ("taggy 0.10-0.20", lambda r: 0.10 <= r["f"]["taggy_density"] < 0.20),
        ("taggy 0.20+", lambda r: r["f"]["taggy_density"] >= 0.20),
    ]
    for name, pred in cuts:
        report_bucket(name, [r for r in rows if pred(r)])


def cross_eval_and_functional(rows: list[dict]) -> None:
    print("\n[cross: lineage × evaluative-density]")
    for lin in (True, False):
        for ev_hi in (True, False):
            sel = [
                r
                for r in rows
                if r["f"]["lineage"] == lin and (r["f"]["eval_hits"] >= 1) == ev_hi
            ]
            name = f"lineage={lin!s:<5} eval>=1={ev_hi!s:<5}"
            report_bucket(name, sel)


def describe_baseline_mrr(rows: list[dict]) -> None:
    # Check whether reviews help queries that were already easy or hard.
    print("\n[baseline difficulty (baseline rank quartiles)]")
    ranks = sorted(r["br"] for r in rows)
    n = len(ranks)
    q1 = ranks[n // 4]
    q2 = ranks[n // 2]
    q3 = ranks[3 * n // 4]
    cuts = [
        (f"Q1 easy (br<={q1})", lambda r: r["br"] <= q1),
        (f"Q2 (br {q1 + 1}..{q2})", lambda r: q1 < r["br"] <= q2),
        (f"Q3 (br {q2 + 1}..{q3})", lambda r: q2 < r["br"] <= q3),
        (f"Q4 hard (br>{q3})", lambda r: r["br"] > q3),
    ]
    for name, pred in cuts:
        report_bucket(name, [r for r in rows if pred(r)])


def main() -> None:
    for ds in DATASETS:
        print(f"\n{'=' * 90}\n{ds}   baseline={BASELINE}  review={REVIEW}\n{'=' * 90}")
        rows = merged(ds)
        print(f"n_total={len(rows)}")
        # global
        report_bucket("ALL", rows)
        stratify_length(rows)
        stratify_eval_density(rows)
        stratify_era(rows)
        stratify_lineage(rows)
        stratify_functional(rows)
        stratify_taggy(rows)
        cross_eval_and_functional(rows)
        describe_baseline_mrr(rows)


if __name__ == "__main__":
    main()
