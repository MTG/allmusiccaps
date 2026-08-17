#!/usr/bin/env python
"""
Gather individual JSON response files into a single JSONL file.

Each line in the output JSONL contains a dict where the key is the YouTube ID
(extracted from the JSON filename) and the value is the dict from the JSON file.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the sharded JSON response files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSONL file path (default: <input_dir>.jsonl)",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {args.input_dir}")

    output_path = args.output or args.input_dir.with_suffix(".jsonl")

    suffix = ".response.json"
    json_files = sorted(args.input_dir.rglob(f"*{suffix}"))

    if not json_files:
        print(f"No {suffix} files found in {args.input_dir}")
        return

    print(f"Found {len(json_files)} response files")

    count = 0
    errors = 0
    with open(output_path, "w") as out_f:
        for json_path in json_files:
            # Extract YouTube ID from filename (remove .response.json suffix)
            yt_id = json_path.name[: -len(suffix)]

            try:
                with open(json_path, "r") as f:
                    payload = json.load(f)

                line = json.dumps({yt_id: payload})
                out_f.write(line + "\n")
                count += 1
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error reading {json_path}: {e}")
                errors += 1

    print(f"Wrote {count} entries to {output_path}")
    if errors:
        print(f"Skipped {errors} files due to errors")


if __name__ == "__main__":
    main()
