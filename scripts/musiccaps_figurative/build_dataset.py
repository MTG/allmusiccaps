#!/usr/bin/env python
"""Turn the cleaned ``levels.jsonl`` into per-level caption files.

For each level in {level_1, ..., level_5} we emit a JSONL of the form:

    {"ytid": <id>, "caption_ground_truth": <rewritten caption>}

The rows match ``MusicCaps.annotations`` columns, so the evaluator can load
these directly, replace the caption column on a MusicCaps dataset object, and
call ``query_processor`` without touching any retrieval code.

Also emits ``pairs.jsonl`` for Experiment A:

    {"ytid": <id>, "original": ..., "level_1": ..., ..., "level_5": ...}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from prompts import LEVELS
from utils import iter_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--levels-path",
        type=Path,
        default=Path("musiccaps_figurative/levels.jsonl"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("musiccaps_figurative/per_level"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(iter_jsonl(args.levels_path))
    print(f"Loaded {len(rows)} rows from {args.levels_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Per-level caption files: minimal columns that MusicCaps.annotations needs.
    for lvl in LEVELS:
        out_rows: List[dict] = []
        for r in rows:
            val = str(r.get(lvl, "") or "").strip()
            if not val:
                continue
            out_rows.append({"ytid": r["id"], "caption_ground_truth": val})
        out_path = args.out_dir / f"{lvl}.jsonl"
        write_jsonl(out_path, out_rows)
        print(f"  {lvl}: {len(out_rows)} rows → {out_path}")

    # A single paired file for Experiment A.
    pairs_path = args.out_dir.parent / "pairs.jsonl"
    pair_rows: List[dict] = []
    for r in rows:
        pair_rows.append(
            {
                "ytid": r["id"],
                "original": r.get("original", ""),
                **{lvl: r.get(lvl, "") for lvl in LEVELS},
            }
        )
    write_jsonl(pairs_path, pair_rows)
    print(f"  pairs: {len(pair_rows)} rows → {pairs_path}")


if __name__ == "__main__":
    main()
