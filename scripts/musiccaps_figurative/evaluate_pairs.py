#!/usr/bin/env python
"""Experiment A: paired literal↔figurative retrieval for MusicCaps.

For each MusicCaps id we have 5 captions, all pointing to the same audio
ground-truth. We want to know, for each model:

- Does the model retrieve the same top-k set for literal (level_1) and
  figurative (level_5) versions of the same caption? (``overlap@k``)
- Does the retrieved rank of the correct track degrade as we move from
  level_1 to level_5? (``rank_by_level``)

This complements ``evaluate_curve.py``: the curve measures mean retrieval
quality per level, while this script measures *paired* behavior per item.

Audio features are computed once. Text features are computed once per level.

Output: ``<out_dir>/<model_tag>/pairs.json`` with:

    {
      "n_items": int,
      "overlap@1": {"level_1_vs_level_5": float, ...},
      "overlap@5": {...},
      "overlap@10": {...},
      "mean_rank_by_level": {"level_1": float, ..., "level_5": float},
      "delta_rank": {"level_1_to_level_5": float, ...},
      "per_id_ranks": {"<ytid>": {"level_1": int, ..., "level_5": int}}
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from prompts import LEVELS
from utils import load_levels_file


def _import_retrieval():
    here = Path(__file__).resolve()
    repo_root = here.parent.parent.parent
    sys.path.append(str(repo_root / "src"))
    sys.path.append(str(repo_root / "src" / "downstream"))
    from retrieval.musicaps_dataset import MusicCaps  # noqa: WPS433
    from retrieval.eval_utils import get_task_predictions  # noqa: WPS433

    return {"MusicCaps": MusicCaps, "get_task_predictions": get_task_predictions}


def load_clap_model(cfg_file: str, device: str, ckpt_step: int | None):
    here = Path(__file__).resolve()
    repo_root = here.parent.parent.parent
    sys.path.append(str(repo_root / "src"))
    from amclap import get_model  # noqa: WPS433

    return get_model(
        config_file=cfg_file,
        device=device,
        weights_only=False,
        ckpt_step=ckpt_step,
    )


def compute_audio_features(model, dataset, unique_tracks: List[str], device: str):
    feats = []
    for ytid in tqdm(unique_tracks, desc="audio"):
        audio = dataset.get_audio(ytid).to(device)
        with torch.inference_mode():
            rep = model.forward_audio(audio)
        rep = rep.mean(0, keepdim=True)
        feats.append(rep.detach().cpu())
    return torch.cat(feats, dim=0)


def compute_text_features_for_level(
    model,
    id_order: List[str],
    id_to_levels: Dict[str, Dict[str, str]],
    level: str,
    use_audio_type_token: bool,
) -> torch.Tensor:
    feats = []
    for _id in tqdm(id_order, desc=f"text-{level}"):
        caption = id_to_levels[_id][level]
        q = f"[MUSIC] {caption}" if use_audio_type_token else caption
        with torch.inference_mode():
            emb = model.forward_text([q])
        feats.append(emb.detach().cpu())
    return torch.cat(feats, dim=0)  # (N, C) in id_order


def topk_indices(similarity: torch.Tensor, k: int) -> torch.Tensor:
    """Return top-k column indices per row (Q, k)."""
    return torch.topk(similarity, k=k, dim=1).indices


def compute_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-row overlap ratio between two top-k index tensors."""
    n_rows, k = a.shape
    overlaps = []
    for i in range(n_rows):
        s_a = set(a[i].tolist())
        s_b = set(b[i].tolist())
        overlaps.append(len(s_a & s_b) / k)
    return float(sum(overlaps) / len(overlaps)) if overlaps else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cfg-file", type=str, required=True)
    p.add_argument("--ckpt-step", type=int, default=None)
    p.add_argument("--model-tag", type=str, required=True)
    p.add_argument(
        "--levels-path", type=Path, default=Path("musiccaps_figurative/levels.jsonl")
    )
    p.add_argument("--data-dir", type=str, default="../../dataset")
    p.add_argument("--audio-loader", type=str, default="ffmpeg")
    p.add_argument("--audio-enc", type=str, default=".wav")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--sr", type=int, default=24000)
    p.add_argument("--segment-size", type=float, default=10.0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--use-audio-type-token", action="store_true")
    p.add_argument(
        "--out-dir", type=Path, default=Path("downstream_results/figurative")
    )
    p.add_argument("--k-values", type=int, nargs="*", default=[1, 5, 10])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    id_to_levels = load_levels_file(args.levels_path)
    print(f"Loaded {len(id_to_levels)} rows")

    R = _import_retrieval()
    MusicCaps = R["MusicCaps"]
    get_task_predictions = R["get_task_predictions"]

    model = load_clap_model(args.cfg_file, args.device, args.ckpt_step)

    base_dataset = MusicCaps(
        data_dir=args.data_dir,
        split=args.split,
        audio_loader=args.audio_loader,
        caption_type="caption",
        sr=args.sr,
        duration=args.segment_size,
        audio_enc=args.audio_enc,
    )

    # Keep only ids present in our levels file. Sort for determinism.
    valid_ids = [
        i
        for i in id_to_levels.keys()
        if all(id_to_levels[i].get(lvl) for lvl in LEVELS)
    ]
    valid_ids = sorted(
        set(valid_ids)
        & set(base_dataset.annotations[base_dataset.id_col].astype(str).tolist())
    )
    print(f"Paired pool size: {len(valid_ids)}")

    base_dataset.annotations = base_dataset.annotations[
        base_dataset.annotations[base_dataset.id_col].astype(str).isin(set(valid_ids))
    ].reset_index(drop=True)

    # Deterministic id ordering → both queries and tracks share this order.
    id_order = valid_ids
    track2idx = {t: i for i, t in enumerate(id_order)}

    audio_features = compute_audio_features(model, base_dataset, id_order, args.device)

    # Per-level text features, aligned row-by-row with id_order.
    text_by_level: Dict[str, torch.Tensor] = {}
    for lvl in LEVELS:
        text_by_level[lvl] = compute_text_features_for_level(
            model, id_order, id_to_levels, lvl, args.use_audio_type_token
        )

    # For each level, compute the similarity matrix (Q, N) and top-k sets.
    sims: Dict[str, torch.Tensor] = {}
    topk_by_level: Dict[int, Dict[str, torch.Tensor]] = {k: {} for k in args.k_values}
    for lvl in LEVELS:
        sims[lvl] = torch.tensor(
            get_task_predictions(text_by_level[lvl], audio_features)
        )
        for k in args.k_values:
            topk_by_level[k][lvl] = topk_indices(sims[lvl], k)

    # ---- overlap@k between each pair of levels ----
    overlap: Dict[str, Dict[str, float]] = {f"overlap@{k}": {} for k in args.k_values}
    lvl_list = list(LEVELS)
    for i, a in enumerate(lvl_list):
        for b in lvl_list[i + 1 :]:
            key = f"{a}_vs_{b}"
            for k in args.k_values:
                overlap[f"overlap@{k}"][key] = compute_overlap(
                    topk_by_level[k][a], topk_by_level[k][b]
                )

    # ---- rank of the correct track per query per level ----
    # Each query's correct track sits at the same index as the query, by construction.
    per_id_ranks: Dict[str, Dict[str, int]] = {}
    rank_by_level: Dict[str, List[int]] = {lvl: [] for lvl in LEVELS}
    for lvl in LEVELS:
        sim_m = sims[lvl]  # (Q, N)
        # For each row q, rank of track q = (# tracks with higher sim than sim[q,q]) + 1
        diag = sim_m.diag().unsqueeze(1)  # (Q, 1)
        ranks = (sim_m > diag).sum(dim=1) + 1  # (Q,)
        for idx, _id in enumerate(id_order):
            per_id_ranks.setdefault(_id, {})[lvl] = int(ranks[idx].item())
            rank_by_level[lvl].append(int(ranks[idx].item()))

    mean_rank_by_level = {
        lvl: float(sum(v) / len(v)) if v else float("nan")
        for lvl, v in rank_by_level.items()
    }

    delta_rank: Dict[str, float] = {}
    for i, a in enumerate(lvl_list):
        for b in lvl_list[i + 1 :]:
            delta_rank[f"{a}_to_{b}"] = mean_rank_by_level[b] - mean_rank_by_level[a]

    payload = {
        "model_tag": args.model_tag,
        "cfg_file": args.cfg_file,
        "ckpt_step": args.ckpt_step,
        "n_items": len(id_order),
        "k_values": args.k_values,
        **overlap,
        "mean_rank_by_level": mean_rank_by_level,
        "delta_rank": delta_rank,
        "per_id_ranks": per_id_ranks,
    }

    out_model_dir = args.out_dir / args.model_tag
    out_model_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_model_dir / "pairs.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nPair results saved to {out_path}")
    print(f"mean_rank_by_level: {mean_rank_by_level}")
    print(f"delta_rank: {delta_rank}")


if __name__ == "__main__":
    main()
