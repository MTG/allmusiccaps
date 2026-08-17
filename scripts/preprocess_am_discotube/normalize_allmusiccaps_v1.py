"""Normalize the AllMusicCaps v0 release into a well-typed v1.

v0 stores the raw second-stage LLM output. For about 1.7% of the rows that have
quote captions, the model wrapped its answer in something other than a list of
strings: a nested list, a dict keyed by attribute name, or an occasional scalar.
`generated_quotes_captions` therefore has an unstable type, and Arrow-backed
readers (`datasets.load_dataset`, `pandas.read_json(lines=True)`) fail outright
on the file.

v1 keeps every row and every caption, and only fixes the container:

  ["a", "b"]                      -> unchanged
  [["a", "b"]]                    -> ["a", "b"]              (unwrap nesting)
  [{"description": "a"}]          -> ["a"]                   (take dict values)
  [{"mood": "x", "genre": "y"}]   -> ["x", "y"]              (one per value)
  [null] / [1] / [true]           -> []                      (drop non-text)

Structured captions are left untouched: they are already a fixed-key object.

Usage:
    python scripts/preprocess_am_discotube/normalize_allmusiccaps_v1.py \
        --input  data/allmusiccaps/allmusiccaps_v0.jsonl \
        --output data/allmusiccaps/allmusiccaps_v1.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

# Placeholders the LLM emitted instead of leaving a field out.
NULL_TEXTS = {
    "",
    "none",
    "null",
    "n/a",
    "na",
    "not specified",
    "unspecified",
    "unknown",
}

MAX_DEPTH = 4


def _clean(text: str) -> str | None:
    """Strip a caption, or return None if it carries no information."""
    text = " ".join(text.split())
    if text.strip().lower().strip(".") in NULL_TEXTS:
        return None
    return text or None


def _looks_like_prose(key: str) -> bool:
    """Whether a dict key is caption text rather than an LLM field name.

    Field names are short identifiers ("mood", "rhythm/tempo", "description").
    A key that is long and contains spaces is the front half of a caption that a
    stray quotation mark split in two.
    """
    return len(key) > 25 and " " in key


def _flatten(value, depth: int = 0) -> list[str]:
    """Collect the caption strings out of an arbitrarily wrapped value."""
    if depth > MAX_DEPTH:
        return []

    if isinstance(value, str):
        cleaned = _clean(value)
        return [cleaned] if cleaned else []

    if isinstance(value, list):
        return [c for item in value for c in _flatten(item, depth + 1)]

    if isinstance(value, dict):
        # Attribute-keyed variants ({"mood": ..., "genre": ...}) and single-field
        # wrappers ({"text": ...}) reduce to their values: the keys are LLM field
        # names, not caption text.
        #
        # A stray quotation mark inside a caption can also split one sentence
        # across a key and its value, e.g.
        #   {"A remixed version of \"Experience": "Expanded\" with ..."}
        # Those keys are prose, so rejoin them instead of dropping half the
        # sentence.
        out = []
        for key, item in value.items():
            if _looks_like_prose(key):
                # The `": "` that JSON consumed as the key/value separator was
                # part of the sentence, so put it back.
                rejoined = _clean(f'{key}": {item}' if isinstance(item, str) else key)
                if rejoined:
                    out.append(rejoined)
                    continue
            out.extend(_flatten(item, depth + 1))
        return out

    # Numbers and booleans are malformed output with no recoverable text.
    return []


def normalize_row(row: dict) -> dict:
    quotes = row.get("generated_quotes_captions")
    if quotes is not None:
        captions = _flatten(quotes)
        # Preserve "no captions" as null rather than an empty list, so the
        # has-quotes test stays the same as in v0.
        row["generated_quotes_captions"] = captions or None
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stats = Counter()
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            row = json.loads(line)
            before = row.get("generated_quotes_captions")
            row = normalize_row(row)
            after = row.get("generated_quotes_captions")

            stats["rows"] += 1
            if before is None:
                stats["no_quotes_in"] += 1
            elif isinstance(before, list) and all(isinstance(x, str) for x in before):
                stats["already_clean"] += 1
            else:
                stats["repaired"] += 1
            if after is None:
                stats["no_quotes_out"] += 1
            else:
                stats["captions_out"] += len(after)

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"rows                 {stats['rows']}")
    print(f"  already well-typed {stats['already_clean']}")
    print(f"  repaired           {stats['repaired']}")
    print(f"  had no quotes      {stats['no_quotes_in']}")
    print(f"rows without quotes after normalization: {stats['no_quotes_out']}")
    print(f"total captions       {stats['captions_out']}")


if __name__ == "__main__":
    main()
