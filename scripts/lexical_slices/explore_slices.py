"""Explore current lexical slices to inform a redesigned taxonomy.

For each dataset (MuCaps, SongD) and each slice, print:
  - slice size
  - top-K queries by review-driven rank gain (baseline_rank - review_rank)
  - top-K queries by review-driven rank loss
  - random sample of "unsliced" queries (no slice hits)

Baseline: R01 (tags+sounds, no reviews)
Review model: R04 (quotes+mu+so)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from slice_mrr import SLICE_NAMES, assign_slices

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_BASE = REPO_ROOT / "downstream_results"

BASELINE = "R01"
REVIEW = "R04"
DATASETS = ["music_caps", "song_describer"]


def load_ranks(model: str, dataset: str) -> dict[int, dict]:
    path = RESULTS_BASE / model / dataset / "caption2rank.json"
    with open(path) as f:
        data = json.load(f)
    return {r["index"]: r for r in data}


def explore(dataset: str, top_k: int, n_unsliced: int, seed: int) -> None:
    print(f"\n{'=' * 80}\n{dataset}\n{'=' * 80}")
    base = load_ranks(BASELINE, dataset)
    rev = load_ranks(REVIEW, dataset)
    shared = sorted(set(base) & set(rev))

    rows = []
    for idx in shared:
        q = base[idx]["query"]
        br = base[idx]["min_rank"]
        rr = rev[idx]["min_rank"]
        labels = assign_slices(q)
        rows.append(
            {
                "idx": idx,
                "query": q,
                "br": br,
                "rr": rr,
                "delta": br - rr,
                "labels": labels,
            }
        )

    n_total = len(rows)
    n_unsliced_total = sum(1 for r in rows if not any(r["labels"].values()))
    print(
        f"total={n_total}   unsliced={n_unsliced_total} ({100 * n_unsliced_total / n_total:.1f}%)"
    )

    for slice_name in SLICE_NAMES:
        slice_rows = [r for r in rows if r["labels"][slice_name]]
        if not slice_rows:
            continue
        gains = sorted(slice_rows, key=lambda r: -r["delta"])
        losses = sorted(slice_rows, key=lambda r: r["delta"])
        mean_delta = sum(r["delta"] for r in slice_rows) / len(slice_rows)
        print(
            f"\n-- slice: {slice_name}   n={len(slice_rows)}   mean Δrank (base-rev)={mean_delta:+.1f}"
        )
        print(f"   TOP GAINS (review much better):")
        for r in gains[:top_k]:
            print(
                f"     [{r['br']:4d} -> {r['rr']:4d}   Δ={r['delta']:+5d}]  {r['query'][:140]}"
            )
        print(f"   TOP LOSSES (review worse):")
        for r in losses[:top_k]:
            print(
                f"     [{r['br']:4d} -> {r['rr']:4d}   Δ={r['delta']:+5d}]  {r['query'][:140]}"
            )

    # Unsliced
    unsliced = [r for r in rows if not any(r["labels"].values())]
    random.seed(seed)
    print(f"\n-- UNSLICED (no slice hit)   n={len(unsliced)}")
    # Split unsliced by delta sign to see where review helps on "plain" queries.
    u_gains = sorted(unsliced, key=lambda r: -r["delta"])
    u_losses = sorted(unsliced, key=lambda r: r["delta"])
    mean_delta_u = sum(r["delta"] for r in unsliced) / max(1, len(unsliced))
    print(f"   mean Δrank (base-rev)={mean_delta_u:+.1f}")
    print(f"   TOP GAINS among unsliced:")
    for r in u_gains[:top_k]:
        print(
            f"     [{r['br']:4d} -> {r['rr']:4d}   Δ={r['delta']:+5d}]  {r['query'][:140]}"
        )
    print(f"   TOP LOSSES among unsliced:")
    for r in u_losses[:top_k]:
        print(
            f"     [{r['br']:4d} -> {r['rr']:4d}   Δ={r['delta']:+5d}]  {r['query'][:140]}"
        )
    print(f"   RANDOM unsliced sample:")
    for r in random.sample(unsliced, min(n_unsliced, len(unsliced))):
        print(
            f"     [{r['br']:4d} -> {r['rr']:4d}   Δ={r['delta']:+5d}]  {r['query'][:140]}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--n-unsliced", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", choices=DATASETS + ["both"], default="both")
    args = ap.parse_args()

    if args.dataset == "both":
        for ds in DATASETS:
            explore(ds, args.top_k, args.n_unsliced, args.seed)
    else:
        explore(args.dataset, args.top_k, args.n_unsliced, args.seed)


if __name__ == "__main__":
    main()
