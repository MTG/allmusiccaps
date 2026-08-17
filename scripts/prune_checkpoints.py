"""Prune redundant checkpoints from a Lightning/WandB experiment directory.

Keeps:
  - The "best" checkpoint (best.ckpt or *-v1.ckpt pattern)
  - The last checkpoint (highest step number)
  - The closest checkpoint to each 20K-step milestone (20K, 40K, 60K, ...)

Usage:
  python scripts/prune_checkpoints.py <checkpoint_dir> [<checkpoint_dir> ...]
  python scripts/prune_checkpoints.py /projects/<group>/logs/clap-mtg/R07/checkpoints

Add --dry-run to preview deletions without removing files.
"""

import argparse
import re
from pathlib import Path

MILESTONE_INTERVAL = 20_000


def parse_step(filename: str) -> int | None:
    """Extract step number from a Lightning checkpoint filename."""
    m = re.search(r"step=(\d+)", filename)
    if m:
        return int(m.group(1))
    return None


def find_checkpoints(checkpoint_dir: Path) -> list[Path]:
    """Find all .ckpt files in a directory."""
    return sorted(checkpoint_dir.glob("*.ckpt"))


def select_keepers(ckpt_paths: list[Path]) -> set[Path]:
    """Select which checkpoints to keep."""
    if not ckpt_paths:
        return set()

    keepers: set[Path] = set()

    # Separate checkpoints with parseable steps from special ones
    step_map: dict[int, Path] = {}
    for p in ckpt_paths:
        # Always keep "best" or "last" named checkpoints
        name_lower = p.name.lower()
        if "best" in name_lower or "last" in name_lower:
            keepers.add(p)
            continue

        step = parse_step(p.name)
        if step is not None:
            # If multiple checkpoints have the same step (e.g., -v1 variants),
            # keep the latest version
            if step not in step_map or p.name > step_map[step].name:
                step_map[step] = p
        else:
            # Can't parse step — keep to be safe
            keepers.add(p)

    if not step_map:
        return keepers

    steps_sorted = sorted(step_map.keys())

    # Keep the last checkpoint (highest step)
    last_step = steps_sorted[-1]
    keepers.add(step_map[last_step])

    # Keep the closest checkpoint to each 20K milestone
    max_step = last_step
    milestone = MILESTONE_INTERVAL
    while milestone <= max_step:
        closest_step = min(steps_sorted, key=lambda s: abs(s - milestone))
        keepers.add(step_map[closest_step])
        milestone += MILESTONE_INTERVAL

    return keepers


def prune_directory(checkpoint_dir: Path, dry_run: bool) -> tuple[int, int]:
    """Prune checkpoints in a single directory. Returns (kept, removed) counts."""
    ckpts = find_checkpoints(checkpoint_dir)
    if not ckpts:
        print(f"  No .ckpt files found in {checkpoint_dir}")
        return 0, 0

    keepers = select_keepers(ckpts)
    to_remove = [p for p in ckpts if p not in keepers]

    print(f"\n  Found {len(ckpts)} checkpoints")
    print(f"  Keeping {len(keepers)}:")
    for p in sorted(keepers, key=lambda x: x.name):
        step = parse_step(p.name)
        label = f" (step {step})" if step is not None else ""
        print(f"    + {p.name}{label}")

    if to_remove:
        print(f"  Removing {len(to_remove)}:")
        for p in sorted(to_remove, key=lambda x: x.name):
            step = parse_step(p.name)
            label = f" (step {step})" if step is not None else ""
            print(f"    - {p.name}{label}")

        if not dry_run:
            for p in to_remove:
                p.unlink()
            print(f"  Deleted {len(to_remove)} files.")
        else:
            print("  (dry run — no files deleted)")
    else:
        print("  Nothing to remove.")

    return len(keepers), len(to_remove)


def main():
    parser = argparse.ArgumentParser(
        description="Prune redundant Lightning checkpoints"
    )
    parser.add_argument("dirs", nargs="+", help="Checkpoint directories to prune")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without deleting"
    )
    args = parser.parse_args()

    total_kept = 0
    total_removed = 0

    for d in args.dirs:
        checkpoint_dir = Path(d)
        if not checkpoint_dir.is_dir():
            print(f"WARNING: {d} is not a directory, skipping")
            continue

        print(f"\nProcessing: {checkpoint_dir}")
        kept, removed = prune_directory(checkpoint_dir, args.dry_run)
        total_kept += kept
        total_removed += removed

    print(f"\nTotal: kept {total_kept}, removed {total_removed}")
    if args.dry_run:
        print("(dry run — re-run without --dry-run to delete)")


if __name__ == "__main__":
    main()
