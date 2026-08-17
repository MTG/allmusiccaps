"""Analyze ΔMRR and Δrank by 5-way descriptive-register label for both datasets."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_BASE = REPO_ROOT / "downstream_results"

BASELINE = "R01"
REVIEW = "R04"
DATASETS = {
    "song_describer": "songd_taxonomy.json",
    "music_caps": "mucaps_taxonomy.json",
}
LABELS = (
    "stylistic_context",
    "structural",
    "descriptive",
    "acoustic_detail",
    "impressionistic",
)


def load_ranks(model: str, dataset: str) -> dict[int, dict]:
    path = RESULTS_BASE / model / dataset / "caption2rank.json"
    with open(path) as f:
        return {r["index"]: r for r in json.load(f)}


def report(name: str, rows: list[dict]) -> None:
    if not rows:
        print(f"  {name:<28}  (empty)")
        return
    n = len(rows)
    md = statistics.mean(r["delta"] for r in rows)
    mb = statistics.mean(r["mrr_base"] for r in rows) * 100
    mr = statistics.mean(r["mrr_rev"] for r in rows) * 100
    r1_b = sum(1 for r in rows if r["br"] == 0) / n * 100
    r1_r = sum(1 for r in rows if r["rr"] == 0) / n * 100
    r10_b = sum(1 for r in rows if r["br"] < 10) / n * 100
    r10_r = sum(1 for r in rows if r["rr"] < 10) / n * 100
    print(
        f"  {name:<28}  n={n:4d}  Δrank={md:+6.1f}   MRR {mb:5.2f}→{mr:5.2f} ({mr - mb:+5.2f})   R@1 {r1_b:4.1f}→{r1_r:4.1f}   R@10 {r10_b:4.1f}→{r10_r:4.1f}"
    )


def analyze(dataset: str) -> None:
    print(f"\n{'=' * 100}\n{dataset}\n{'=' * 100}")
    with open(Path(__file__).parent / DATASETS[dataset]) as f:
        tax = {r["index"]: r for r in json.load(f)}
    b = load_ranks(BASELINE, dataset)
    r = load_ranks(REVIEW, dataset)

    rows = []
    for idx in sorted(set(b) & set(r) & set(tax)):
        rows.append(
            {
                "idx": idx,
                "q": b[idx]["query"],
                "br": b[idx]["min_rank"],
                "rr": r[idx]["min_rank"],
                "delta": b[idx]["min_rank"] - r[idx]["min_rank"],
                "mrr_base": 1.0 / (b[idx]["min_rank"] + 1),
                "mrr_rev": 1.0 / (r[idx]["min_rank"] + 1),
                "label": tax[idx]["label"],
                "rationale": tax[idx].get("rationale", ""),
            }
        )

    hist = Counter(r["label"] for r in rows)
    print(f"n = {len(rows)}")
    for L in LABELS + ("noparse",):
        if hist.get(L, 0):
            print(f"  {L:<18}: n={hist[L]:4d}  ({100 * hist[L] / len(rows):.1f}%)")

    print("\n[ALL]")
    report("ALL", rows)

    print("\n[per label]")
    for L in LABELS:
        report(f"label={L}", [r for r in rows if r["label"] == L])

    print("\n[examples per label]")
    import random

    random.seed(0)
    for L in LABELS:
        bucket = [r for r in rows if r["label"] == L]
        if not bucket:
            continue
        sample = random.sample(bucket, min(4, len(bucket)))
        for r in sample:
            print(
                f"  [{L[:6]}] br={r['br']:4d} rr={r['rr']:4d} Δ={r['delta']:+5d}  {r['q'][:110]}"
            )


def main() -> None:
    for ds in DATASETS:
        analyze(ds)


if __name__ == "__main__":
    main()
