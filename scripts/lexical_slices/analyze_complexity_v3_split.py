"""Split captions by v3 (2x2: density x framing) judgements; report Δrank and ΔMRR.

Reports per-bucket and per-axis breakdowns plus the legacy 1..4 score path.
"""

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
    "song_describer": "songd_complexity_v3.json",
    "music_caps": "mucaps_complexity_v3.json",
}

BUCKETS = [
    ("thin", "none"),
    ("thin", "present"),
    ("dense", "none"),
    ("dense", "present"),
]


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
    print(f"\n{'=' * 95}\n{dataset}\n{'=' * 95}")
    path = Path(__file__).parent / DATASETS[dataset]
    if not path.exists():
        print(
            f"  (missing: {path}, run judge_complexity_v3_vllm.py on the cluster first)"
        )
        return
    with open(path) as f:
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
                "density": comp[idx].get("density", ""),
                "framing": comp[idx].get("framing", ""),
                "rationale": comp[idx].get("rationale", ""),
            }
        )

    print(f"n = {len(rows)}  (judged + ranked)")

    bucket_hist = Counter((r["density"], r["framing"]) for r in rows)
    print("\n2x2 histogram (density, framing):")
    for d, f in BUCKETS:
        n = bucket_hist.get((d, f), 0)
        pct = 100 * n / len(rows) if rows else 0
        print(f"  {d:>5} + {f:<7}  n={n:4d}  ({pct:5.1f}%)")

    print("\n[ALL]")
    report("ALL", rows)

    print("\n[per 2x2 bucket]")
    for d, f in BUCKETS:
        sub = [r for r in rows if r["density"] == d and r["framing"] == f]
        report(f"{d}+{f}", sub)

    print("\n[per axis -- density]")
    for d in ("thin", "dense"):
        report(f"density={d}", [r for r in rows if r["density"] == d])

    print("\n[per axis -- framing]")
    for f in ("none", "present"):
        report(f"framing={f}", [r for r in rows if r["framing"] == f])

    print("\n[legacy 1..4 score]")
    for s in (1, 2, 3, 4):
        report(f"score={s}", [r for r in rows if r["score"] == s])

    print("\n[3 example queries per bucket]")
    import random

    random.seed(0)
    for d, f in BUCKETS:
        bucket = [r for r in rows if r["density"] == d and r["framing"] == f]
        for r in random.sample(bucket, min(3, len(bucket))):
            print(
                f"  [{d:>5}+{f:<7}] br={r['br']:4d} rr={r['rr']:4d} Δ={r['delta']:+5d}  {r['q'][:100]}"
            )
            if r["rationale"]:
                print(f"          reason: {r['rationale'][:90]}")


def main() -> None:
    for ds in DATASETS:
        analyze(ds)


if __name__ == "__main__":
    main()
