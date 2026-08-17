"""Split SongDescriber by LLM-judged complexity; report Δrank and ΔMRR.

Simple half = bottom 50% by score, complex half = top 50%.
Ties on the median broken toward the smaller half by stable sort order.
Also report per-score-value breakdown (1..5).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_BASE = REPO_ROOT / "downstream_results"

BASELINE = "R01"
REVIEW = "R04"
DATASETS = {
    "song_describer": "songd_complexity.json",
    "music_caps": "mucaps_complexity.json",
}


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
    print(f"\n{'=' * 90}\n{dataset}\n{'=' * 90}")
    with open(Path(__file__).parent / DATASETS[dataset]) as f:
        comp = {r["index"]: r for r in json.load(f)}
    b = load_ranks(BASELINE, dataset)
    r = load_ranks(REVIEW, dataset)

    rows = []
    for idx in sorted(set(b) & set(r) & set(comp)):
        rows.append(
            {
                "idx": idx,
                "q": b[idx]["query"],
                "br": b[idx]["min_rank"],
                "rr": r[idx]["min_rank"],
                "delta": b[idx]["min_rank"] - r[idx]["min_rank"],
                "mrr_base": 1.0 / (b[idx]["min_rank"] + 1),
                "mrr_rev": 1.0 / (r[idx]["min_rank"] + 1),
                "score": comp[idx]["score"],
                "rationale": comp[idx].get("rationale", ""),
            }
        )

    print(f"n = {len(rows)}  (complexity judged + ranked)\n")

    # score histogram
    from collections import Counter

    hist = Counter(r["score"] for r in rows)
    print("score histogram (1=tag-expressible, 5=deeply narrative):")
    for s in sorted(hist):
        print(f"  score {s}: n={hist[s]:4d}  ({100 * hist[s] / len(rows):.1f}%)")

    print("\n[ALL]")
    report("ALL", rows)

    # per-score
    print("\n[per complexity score]")
    for s in sorted(hist):
        report(f"score={s}", [r for r in rows if r["score"] == s])

    # 50/50 split
    print("\n[median split (simple vs complex)]")
    sorted_rows = sorted(rows, key=lambda r: (r["score"], r["idx"]))
    half = len(sorted_rows) // 2
    simple = sorted_rows[:half]
    complex_ = sorted_rows[half:]
    s_max = simple[-1]["score"]
    c_min = complex_[0]["score"]
    print(f"  simple: scores up to {s_max}  (n={len(simple)})")
    print(f"  complex: scores {c_min}+     (n={len(complex_)})")
    report("SIMPLE half", simple)
    report("COMPLEX half", complex_)

    # natural split: scores {1,2} vs {3,4,5}
    low = [r for r in rows if r["score"] <= 2]
    high = [r for r in rows if r["score"] >= 3]
    print(f"\n[natural split score<=2 vs score>=3]")
    report("simple (score 1-2)", low)
    report("complex (score 3-5)", high)

    # examples
    print("\n[3 example queries per score]")
    import random

    random.seed(0)
    for s in sorted(hist):
        bucket = [r for r in rows if r["score"] == s]
        for r in random.sample(bucket, min(3, len(bucket))):
            print(
                f"  [{s}] br={r['br']:4d} rr={r['rr']:4d} Δ={r['delta']:+5d}  {r['q'][:110]}"
            )
            if r["rationale"]:
                print(f"        reason: {r['rationale'][:100]}")


def main() -> None:
    for ds in DATASETS:
        analyze(ds)


if __name__ == "__main__":
    main()
