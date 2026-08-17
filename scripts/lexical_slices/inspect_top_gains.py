"""Inspect queries where quotes model improves most on SongD, focusing on the
'hard' register bucket (mixed objective+figurative). Goal: spot shared patterns
to design a better taxonomy.

Prints for SongD:
- top-50 queries by rank gain (review - baseline), any register
- top-30 gains within each register label (objective / figurative / hard)
- rationales from the register judge (for hard bucket)
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

    rows.sort(key=lambda x: -x["delta"])

    print("=" * 100)
    print("TOP 60 gains on SongD (any register)")
    print("=" * 100)
    for r in rows[:60]:
        print(
            f"  [{r['label'][:4]}] br={r['br']:4d} rr={r['rr']:4d} Δ={r['delta']:+5d}  {r['q']}"
        )

    for label in ("hard", "figurative", "objective"):
        print()
        print("=" * 100)
        print(f"TOP 30 gains in register={label}")
        print("=" * 100)
        sub = [x for x in rows if x["label"] == label][:30]
        for r in sub:
            print(f"  br={r['br']:4d} rr={r['rr']:4d} Δ={r['delta']:+5d}  {r['q']}")
            print(f"        reason: {r['rationale'][:140]}")


if __name__ == "__main__":
    main()
