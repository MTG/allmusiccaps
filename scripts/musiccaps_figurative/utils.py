"""Shared helpers for the MusicCaps figurative-caption experiment (E1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from prompts import LEVELS


# ---------- dataset I/O ----------


def iter_jsonl(path: Path) -> Iterable[dict]:
    """Yield one parsed JSON object per line."""
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Write an iterable of dicts to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_levels_file(path: Path) -> Dict[str, Dict[str, str]]:
    """Load levels.jsonl → {id: {level_1, ..., level_5, original}}.

    Expects each row to carry at least:
      - "id"       : the MusicCaps ytid
      - "original" : the original caption
      - "levels"   : {"1": ..., "5": ...} OR flat level_1/.../level_5 keys
    """
    out: Dict[str, Dict[str, str]] = {}
    for row in iter_jsonl(path):
        _id = row["id"]
        rec: Dict[str, str] = {"original": row.get("original", "")}
        if "levels" in row and isinstance(row["levels"], dict):
            for k in ("1", "2", "3", "4", "5"):
                rec[f"level_{k}"] = str(row["levels"].get(k, "")).strip()
        else:
            for lvl in LEVELS:
                rec[lvl] = str(row.get(lvl, "")).strip()
        out[_id] = rec
    return out


# ---------- MusicCaps dataset helpers (hook point for the evaluator) ----------


def patch_musiccaps_captions(
    dataset,
    id_to_caption: Dict[str, str],
    drop_missing: bool = True,
) -> Tuple[int, int]:
    """Replace the MusicCaps caption column in place with a new mapping.

    This is the "zero changes to core retrieval" hook: the retrieval pipeline
    reads captions via ``dataset.annotations[dataset.caption_col]``, so as long
    as we overwrite that column (and keep the id column aligned), ``query_processor``
    will treat the rewritten captions as the new queries.

    Args:
        dataset: an instantiated MusicCaps dataset object
        id_to_caption: {ytid: new caption string}
        drop_missing: if True, drop rows whose id is not in id_to_caption

    Returns:
        (n_kept, n_dropped)
    """
    df = dataset.annotations.copy()
    id_col = dataset.id_col
    cap_col = dataset.caption_col

    mask = df[id_col].astype(str).isin(set(id_to_caption.keys()))
    n_drop = int((~mask).sum())
    n_keep = int(mask.sum())

    if drop_missing:
        df = df[mask].reset_index(drop=True)

    df[cap_col] = df[id_col].astype(str).map(id_to_caption).fillna(df[cap_col])
    # Drop any rows whose new caption is empty — MusicCaps query_processor
    # already filters out "" but we do it explicitly to keep counts honest.
    df = df[df[cap_col].astype(str).str.strip() != ""].reset_index(drop=True)

    dataset.annotations = df
    # Keep ytid_to_idx aligned (used by MusicCaps.__getitem__ in training paths).
    if hasattr(dataset, "ytid_to_idx"):
        dataset.ytid_to_idx = {
            ytid: idx for idx, ytid in enumerate(dataset.annotations[id_col])
        }
    return n_keep, n_drop


# ---------- sanity ----------


def group_rows_by_id(rows: List[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for r in rows:
        out.setdefault(r["id"], []).append(r)
    return out
