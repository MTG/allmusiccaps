#!/usr/bin/env python
"""Dump a small qualitative sample of (original → level_1 .. level_5) rewrites.

Use this for a quick sanity check *after* generate_captions.py + postprocess.py.
Not used by any quantitative evaluation — purely a human-readable preview.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from prompts import LEVELS
from utils import iter_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--levels-path",
        type=Path,
        default=Path("musiccaps_figurative/levels.jsonl"),
    )
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(iter_jsonl(args.levels_path))
    if not rows:
        print("empty input")
        return
    random.Random(args.seed).shuffle(rows)
    sample = rows[: args.n]

    for i, r in enumerate(sample):
        print(f"\n=== [{i + 1}/{len(sample)}] id={r['id']} ===")
        print(f"  original: {r.get('original', '').strip()}")
        for lvl in LEVELS:
            print(f"  {lvl}:  {str(r.get(lvl, '')).strip()}")


if __name__ == "__main__":
    main()
