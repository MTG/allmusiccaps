import argparse
from pathlib import Path
import json


def extract_keys(input_file: Path, output_file: Path):
    """
    Reads a JSONL file where each line is a dictionary with a single key and writes
    all keys to the output file, one per line.
    """
    with (
        input_file.open("r", encoding="utf-8") as infile,
        output_file.open("w", encoding="utf-8") as outfile,
    ):
        for line in infile:
            try:
                data = json.loads(line)
                if len(data) == 1:
                    key = next(iter(data.keys()))
                    outfile.write(key + "\n")
                else:
                    raise ValueError(
                        "Expected a dictionary with a single key, got: {}".format(data)
                    )
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Process a JSONL file to extract all dictionary keys and write them to an output file."
    )
    parser.add_argument("input", type=Path, help="Path to the input JSONL file.")
    parser.add_argument(
        "output", type=Path, help="Path to the output file where keys will be written."
    )

    args = parser.parse_args()

    extract_keys(args.input, args.output)


if __name__ == "__main__":
    main()
