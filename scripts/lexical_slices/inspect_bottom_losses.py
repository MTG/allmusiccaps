"""Inspect queries where quotes model *loses* most on SongD. Mirror of
inspect_top_gains.py but sorted ascending (largest losses first). Goal: spot
patterns in the queries the review model hurts, to refine the taxonomy.

Prints for SongD:
- bottom-60 queries by rank gain (most negative delta), any register
- bottom-30 losses within each register label
- rationales from the register judge
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_BASE = REPO_ROOT / "downstream_results"
BASELINE = "R01"
REVIEW = "R04"
DATASET = "song_describer"
REG_PATH = Path(__file__).parent / "songd_register.json"


def load_ranks(model: str) -> dict[int, dict]:
    with open(RESULTS_BASE / model / DATASET / "caption2rank.json") as f:
        return {r["index"]: r for r in json.load(f)}


def main() -> None:
    b = load_ranks(BASELINE)
    r = load_ranks(REVIEW)
    with open(REG_PATH) as f:
        reg = {r["index"]: r for r in json.load(f)}

    rows = []
    for idx in sorted(set(b) & set(r) & set(reg)):
        rows.append(
            {
                "idx": idx,
                "q": b[idx]["query"],
                "br": b[idx]["min_rank"],
                "rr": r[idx]["min_rank"],
                "delta": b[idx]["min_rank"] - r[idx]["min_rank"],
                "label": reg[idx]["label"],
                "rationale": reg[idx].get("rationale", ""),
            }
        )

    rows.sort(key=lambda x: x["delta"])

    print("=" * 100)
    print("BOTTOM 60 (largest losses) on SongD (any register)")
    print("=" * 100)
    for r in rows[:60]:
        print(
            f"  [{r['label'][:4]}] br={r['br']:4d} rr={r['rr']:4d} Δ={r['delta']:+5d}  {r['q']}"
        )

    for label in ("objective", "figurative", "hard"):
        print()
        print("=" * 100)
        print(f"BOTTOM 30 losses in register={label}")
        print("=" * 100)
        sub = [x for x in rows if x["label"] == label][:30]
        for r in sub:
            print(f"  br={r['br']:4d} rr={r['rr']:4d} Δ={r['delta']:+5d}  {r['q']}")
            print(f"        reason: {r['rationale'][:140]}")

    # near-zero / no-effect band for contrast
    print()
    print("=" * 100)
    print("NEAR-ZERO delta (|Δ| <= 2), sample of 40")
    print("=" * 100)
    neutral = [x for x in rows if abs(x["delta"]) <= 2]
    for r in neutral[:40]:
        print(
            f"  [{r['label'][:4]}] br={r['br']:4d} rr={r['rr']:4d} Δ={r['delta']:+5d}  {r['q']}"
        )


if __name__ == "__main__":
    main()
