"""Create train/val splits for M4-RAG dataset matched with discotube audio.

This script:
1. Loads the M4-RAG dataset from HuggingFace
2. Loads the discotube filelist (YouTube IDs with audio)
3. Finds the intersection
4. Creates 85/15 train/val random splits
5. Saves the splits and matched metadata

Usage:
    python scripts/create_m4rag_splits.py \
        --discotube_filelist filelist_discotube \
        --output_dir /path/to/output
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def extract_youtube_id(path: str) -> str:
    """Extract YouTube ID from discotube path."""
    # Path format: ./discotube-2020-09/7Y/7YxD9SxD0KQ.mmap
    return Path(path).stem


def normalize_path(path: str) -> str:
    """Normalize path for cluster use (strip leading ./)."""
    if path.startswith("./"):
        return path[2:]
    return path


def main():
    parser = argparse.ArgumentParser(description="Create M4-RAG train/val splits")
    parser.add_argument(
        "--discotube_filelist",
        type=Path,
        default=Path("filelist_discotube"),
        help="Path to discotube filelist",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/m4rag_splits"),
        help="Output directory for splits",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.85,
        help="Train split ratio (default: 0.85)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load discotube filelist and create ID to path mapping
    print("Loading discotube filelist...")
    discotube_paths = {}
    with open(args.discotube_filelist, "r") as f:
        for line in tqdm(f, desc="Reading filelist"):
            line = line.strip()
            if line:
                yt_id = extract_youtube_id(line)
                discotube_paths[yt_id] = line

    print(f"Loaded {len(discotube_paths)} discotube entries")

    # Load M4-RAG dataset
    print("Loading M4-RAG dataset...")
    m4rag = load_dataset("sander-wood/m4-rag", split="train")
    print(f"Loaded {len(m4rag)} M4-RAG entries")

    # Find intersection
    print("Finding intersection...")
    matched_entries = []
    for entry in tqdm(m4rag, desc="Matching"):
        yt_id = entry["id"]
        if yt_id in discotube_paths:
            matched_entries.append(
                {
                    "id": yt_id,
                    "path": discotube_paths[yt_id],
                    "title": entry["title"],
                    "artists": entry["artists"],
                    "genres": entry["genres"],
                    "tags": entry["tags"],
                    "background": entry["background"],
                    "analysis": entry["analysis"],
                    "description": entry["description"],
                    "scene": entry["scene"],
                }
            )

    print(f"Found {len(matched_entries)} matched entries")

    if len(matched_entries) == 0:
        print("ERROR: No matches found!")
        return

    # Shuffle and split
    random.shuffle(matched_entries)
    split_idx = int(len(matched_entries) * args.train_ratio)
    train_entries = matched_entries[:split_idx]
    val_entries = matched_entries[split_idx:]

    print(f"Train: {len(train_entries)}, Val: {len(val_entries)}")

    # Save splits
    train_filelist = args.output_dir / "filelist_train.txt"
    val_filelist = args.output_dir / "filelist_val.txt"
    metadata_file = args.output_dir / "m4rag_metadata.jsonl"

    # Write filelists (normalize paths for cluster)
    with open(train_filelist, "w") as f:
        for entry in train_entries:
            f.write(normalize_path(entry["path"]) + "\n")

    with open(val_filelist, "w") as f:
        for entry in val_entries:
            f.write(normalize_path(entry["path"]) + "\n")

    # Write metadata as JSONL
    with open(metadata_file, "w") as f:
        for entry in matched_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Saved train filelist to {train_filelist}")
    print(f"Saved val filelist to {val_filelist}")
    print(f"Saved metadata to {metadata_file}")

    # Print summary
    print("\n=== Summary ===")
    print(f"Total matched: {len(matched_entries)}")
    print(
        f"Train samples: {len(train_entries)} ({len(train_entries) / len(matched_entries) * 100:.1f}%)"
    )
    print(
        f"Val samples: {len(val_entries)} ({len(val_entries) / len(matched_entries) * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
