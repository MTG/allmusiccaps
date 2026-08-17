"""Compare non-probing downstream results with vs. without the audio-type token.

Reads `downstream_results_with_token/` and `downstream_results_without_token/`
after they have been rsynced from the cluster, and prints a per-(model, task)
table showing both values and which variant wins. Also writes a CSV.

Usage:
  python scripts/compare_token_ablation.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WITH_ROOT = REPO_ROOT / "downstream_results_with_token"
WITHOUT_ROOT = REPO_ROOT / "downstream_results_without_token"
OUT_CSV = REPO_ROOT / "downstream_results" / "_aggregates" / "token_ablation.csv"

LAST_CKPT_MODELS = [
    "R01",
    "R02",
    "R03",
    "R04",
    "R05",
    "R06",
    "R07",
    "R08",
    "c0u3izks",
    "o26r43l0",
]
TE_MODELS = ["R13", "R14"]
TE_STEP = 60000

# (task_dir, json_filename, metric_key, display_name, higher_is_better)
TASK_SPECS = [
    ("gtzan_zsl", "results.json", "accuracy", "GTZAN Acc", True),
    ("fma_small_zsl", "results.json", "accuracy", "FMA-S Acc", True),
    ("dimsim", "results.json", "accuracy", "DimSim Acc", True),
    ("music_caps", "caption.json", "mean_reciprocal_rank", "MuCaps MRR", True),
    ("song_describer", "caption.json", "mean_reciprocal_rank", "SongD MRR", True),
]


def find_step_dir(root: Path, model_id: str, pinned_step: int | None) -> Path | None:
    """Return `<root>/<model_id>/step=<N>` (highest N, or pinned if given)."""
    model_dir = root / model_id
    if not model_dir.is_dir():
        return None
    step_dirs = sorted(
        model_dir.glob("step=*"), key=lambda p: int(p.name.split("=")[1])
    )
    if not step_dirs:
        return None
    if pinned_step is not None:
        for d in step_dirs:
            if int(d.name.split("=")[1]) == pinned_step:
                return d
        return None
    return step_dirs[-1]


def load_metric(
    step_dir: Path, task_dir: str, json_file: str, key: str
) -> float | None:
    p = step_dir / task_dir / json_file
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return data.get(key)


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    models = [(m, None) for m in LAST_CKPT_MODELS] + [(m, TE_STEP) for m in TE_MODELS]

    header = ["model", "step", "task"] + [
        "without_token",
        "with_token",
        "delta (with - without)",
        "winner",
    ]
    print("\t".join(header))

    for model_id, pinned in models:
        without_dir = find_step_dir(WITHOUT_ROOT, model_id, pinned)
        with_dir = find_step_dir(WITH_ROOT, model_id, pinned)
        ref_dir = without_dir or with_dir
        if ref_dir is None:
            continue
        step_label = ref_dir.name

        for task_dir, json_file, metric_key, display, higher in TASK_SPECS:
            wo = (
                load_metric(without_dir, task_dir, json_file, metric_key)
                if without_dir
                else None
            )
            wi = (
                load_metric(with_dir, task_dir, json_file, metric_key)
                if with_dir
                else None
            )

            if wo is None and wi is None:
                continue

            if wo is not None and wi is not None:
                delta = wi - wo
                if abs(delta) < 1e-9:
                    winner = "tie"
                elif (delta > 0) == higher:
                    winner = "with"
                else:
                    winner = "without"
            else:
                delta = None
                winner = "with" if wi is not None else "without"

            fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else ""
            delta_str = fmt(delta) if delta is not None else ""
            print(
                "\t".join(
                    [
                        model_id,
                        step_label,
                        display,
                        fmt(wo),
                        fmt(wi),
                        delta_str,
                        winner,
                    ]
                )
            )
            rows.append(
                {
                    "model": model_id,
                    "step": step_label,
                    "task": display,
                    "without_token": wo,
                    "with_token": wi,
                    "delta": delta,
                    "winner": winner,
                }
            )

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
            if rows
            else [
                "model",
                "step",
                "task",
                "without_token",
                "with_token",
                "delta",
                "winner",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote CSV: {OUT_CSV}")

    # Per-model summary: count wins.
    print("\nPer-model win counts (with_token wins / without_token wins / ties):")
    per_model: dict[str, dict[str, int]] = {}
    for r in rows:
        d = per_model.setdefault(r["model"], {"with": 0, "without": 0, "tie": 0})
        d[r["winner"]] = d.get(r["winner"], 0) + 1
    for m, c in per_model.items():
        print(
            f"  {m}: with={c.get('with', 0)}  without={c.get('without', 0)}  tie={c.get('tie', 0)}"
        )


if __name__ == "__main__":
    main()
