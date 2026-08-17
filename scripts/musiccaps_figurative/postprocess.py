#!/usr/bin/env python
"""Validate, filter and compact the raw LLM rewrites into a clean levels file.

Reads the JSONL produced by ``generate_captions.py`` and writes:

- ``levels.jsonl`` : one row per MusicCaps id where ALL 5 levels are valid
                     (non-empty, roughly the right length). Ready for
                     ``evaluate_curve.py`` and ``evaluate_pairs.py``.
- ``rejected.jsonl`` : rows that failed validation (for later inspection).
- ``stats.json``     : counts + per-level length stats + rejection reasons.

Validation rules (kept minimal on purpose — the real signal comes from the
downstream retrieval evals, not from textual heuristics):

1. ``is_valid`` flag from the generator must be True.
2. All 5 levels must be present and non-empty.
3. Each level must have at least MIN_CHARS characters (default 10) and at
   most MAX_CHAR_RATIO × len(original) (default 3.0) — rejects obvious
   degenerations like a single word or a 10-paragraph riff.
4. No level can be byte-identical to the original caption. (If level_1 is
   identical to the input that is still allowed — the model can legitimately
   decide the caption is already bare literal. We only reject if ALL 5
   levels are identical to the input, which is a clear failure mode.)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from prompts import LEVELS
from utils import iter_jsonl, write_jsonl


MIN_CHARS = 10
MAX_CHAR_RATIO = 3.0


def validate_row(row: dict) -> Tuple[bool, str]:
    if not row.get("is_valid", False):
        return False, f"generator:{row.get('parse_error', 'unknown')}"

    original = str(row.get("original", "")).strip()
    if not original:
        return False, "empty_original"

    levels = row.get("levels") or {}
    # Accept both {"1": ...} and {"level_1": ...} shapes.
    norm: Dict[str, str] = {}
    for lvl in LEVELS:
        k = lvl.split("_")[1]
        if k in levels:
            norm[k] = str(levels[k] or "").strip()
        elif lvl in levels:
            norm[k] = str(levels[lvl] or "").strip()
        else:
            return False, f"missing:{lvl}"

    for k, v in norm.items():
        if not v:
            return False, f"empty:level_{k}"
        if len(v) < MIN_CHARS:
            return False, f"too_short:level_{k}"
        if len(v) > MAX_CHAR_RATIO * max(len(original), MIN_CHARS):
            return False, f"too_long:level_{k}"

    # All-identical-to-original ⇒ degenerate
    if all(norm[str(i)].strip() == original.strip() for i in range(1, 6)):
        return False, "all_identical_to_original"

    return True, "ok"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--in-path",
        type=Path,
        default=Path("musiccaps_figurative/raw_levels.jsonl"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("musiccaps_figurative"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(iter_jsonl(args.in_path))
    print(f"Loaded {len(rows)} raw rows from {args.in_path}")

    kept: List[dict] = []
    rejected: List[dict] = []
    reasons: Counter[str] = Counter()

    seen_ids = set()
    per_level_lens: Dict[str, List[int]] = {f"level_{i}": [] for i in range(1, 6)}

    for row in rows:
        ok, reason = validate_row(row)
        if not ok:
            rejected.append({**row, "_reason": reason})
            reasons[reason] += 1
            continue
        _id = str(row["id"])
        if _id in seen_ids:
            # If generate was re-run with --force, keep the most recent row.
            # Remove the earlier one. Small dataset → linear scan is fine.
            kept = [r for r in kept if r["id"] != _id]
        seen_ids.add(_id)

        flat = {
            "id": _id,
            "original": str(row["original"]).strip(),
        }
        levels = row.get("levels") or {}
        for lvl in LEVELS:
            k = lvl.split("_")[1]
            val = str(levels.get(k, levels.get(lvl, ""))).strip()
            flat[lvl] = val
            per_level_lens[lvl].append(len(val))
        kept.append(flat)

    out_kept = args.out_dir / "levels.jsonl"
    out_rej = args.out_dir / "rejected.jsonl"
    out_stats = args.out_dir / "stats.json"

    write_jsonl(out_kept, kept)
    write_jsonl(out_rej, rejected)

    def _length_stats(xs: List[int]) -> dict:
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "min": min(xs),
            "max": max(xs),
            "avg": round(sum(xs) / len(xs), 2),
        }

    stats = {
        "total": len(rows),
        "kept": len(kept),
        "rejected": len(rejected),
        "reject_reasons": dict(reasons),
        "per_level_char_lengths": {
            k: _length_stats(v) for k, v in per_level_lens.items()
        },
    }
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    with open(out_stats, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Kept   : {len(kept)}  → {out_kept}")
    print(f"Rejected: {len(rejected)}  → {out_rej}")
    print(f"Stats  : {out_stats}")
    if reasons:
        print(f"Reject reasons: {dict(reasons)}")


if __name__ == "__main__":
    main()
