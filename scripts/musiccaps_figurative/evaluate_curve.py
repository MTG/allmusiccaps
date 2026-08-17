#!/usr/bin/env python
"""Experiment B: figurative-sensitivity curve (level → retrieval metric).

For each figurative level in {level_1, ..., level_5} and each model, run the
exact same MusicCaps retrieval pipeline that the rest of the downstream suite
uses, but swap ``dataset.annotations[caption_col]`` for the rewritten
captions. Audio embeddings are computed **once** per model and reused across
levels (only text embeddings are recomputed). Saves per-level metrics to:

    <out_dir>/<model_tag>/curve.json
    <out_dir>/<model_tag>/<level>/caption.json

``curve.json`` has the shape:

    {
      "model_tag": "R04",
      "levels": ["level_1", ..., "level_5"],
      "metrics": {
          "level_1": {"recall@1": ..., "mean_reciprocal_rank": ..., ...},
          ...
      },
      "n_tracks": <int>,
      "n_queries_per_level": {"level_1": <int>, ...}
    }

This script intentionally duplicates a small amount of the retrieval glue from
``src/downstream/downstream_retrieval.py``. We do NOT import that script
because it is an argparse entry point with module-level ``parser.parse_args()``
which would break import. The core retrieval module code (``retrieval/``) is
reused via a ``sys.path`` append.
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
from utils import load_levels_file, patch_musiccaps_captions


# --------------------------- retrieval core wiring --------------------------


def _import_retrieval():
    """Make ``src/downstream`` importable so we can reuse MusicCaps + metrics."""
    here = Path(__file__).resolve()
    # scripts/musiccaps_figurative/ -> repo root
    repo_root = here.parent.parent.parent
    sys.path.append(str(repo_root / "src"))
    sys.path.append(str(repo_root / "src" / "downstream"))
    from retrieval.musicaps_dataset import MusicCaps  # noqa: WPS433
    from retrieval.query_utils import query_processor  # noqa: WPS433
    from retrieval.eval_utils import get_query2target_idx, get_task_predictions  # noqa: WPS433
    from retrieval.metrics import (  # noqa: WPS433
        mean_average_precision,
        mean_reciprocal_rank,
        median_rank,
        recall,
    )

    return {
        "MusicCaps": MusicCaps,
        "query_processor": query_processor,
        "get_query2target_idx": get_query2target_idx,
        "get_task_predictions": get_task_predictions,
        "recall": recall,
        "map": mean_average_precision,
        "mrr": mean_reciprocal_rank,
        "medrank": median_rank,
    }


def load_clap_model(cfg_file: str, device: str, ckpt_step: int | None):
    """Load a CLAP model the same way ``downstream_retrieval.py`` does."""
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


# ------------------------------ audio features ------------------------------


def compute_audio_features(model, dataset, unique_tracks: List[str], device: str):
    """Compute per-track audio embeddings (mean-pooled across segments)."""
    feats = []
    for ytid in tqdm(unique_tracks, desc="audio"):
        audio = dataset.get_audio(ytid).to(device)
        with torch.inference_mode():
            rep = model.forward_audio(audio)
        rep = rep.mean(0, keepdim=True)  # (1, C)
        feats.append(rep.detach().cpu())
    return torch.cat(feats, dim=0)  # (N, C)


def compute_text_features(model, queries: List[str], use_audio_type_token: bool):
    feats = []
    for q in tqdm(queries, desc="text"):
        q_in = f"[MUSIC] {q}" if use_audio_type_token else q
        with torch.inference_mode():
            emb = model.forward_text([q_in])
        feats.append(emb.detach().cpu())
    return torch.cat(feats, dim=0)  # (Q, C)


# ---------------------------------- main -----------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cfg-file",
        type=str,
        required=True,
        help="Path to gin config of the CLAP model checkpoint",
    )
    p.add_argument("--ckpt-step", type=int, default=None)
    p.add_argument(
        "--model-tag",
        type=str,
        required=True,
        help="Short label used as output subdir (e.g., R04)",
    )
    p.add_argument(
        "--levels-path", type=Path, default=Path("musiccaps_figurative/levels.jsonl")
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default="../../dataset",
        help="Dataset root (for MusicCaps audio)",
    )
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
    p.add_argument(
        "--levels",
        type=str,
        nargs="*",
        default=list(LEVELS),
        help="Subset of levels to evaluate (default: all 5)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    id_to_levels = load_levels_file(args.levels_path)
    print(f"Loaded {len(id_to_levels)} rows from {args.levels_path}")

    R = _import_retrieval()
    MusicCaps = R["MusicCaps"]
    query_processor = R["query_processor"]
    get_query2target_idx = R["get_query2target_idx"]
    get_task_predictions = R["get_task_predictions"]
    recall = R["recall"]
    mean_average_precision = R["map"]
    mean_reciprocal_rank = R["mrr"]
    median_rank = R["medrank"]

    print(f"Loading model from {args.cfg_file}")
    model = load_clap_model(args.cfg_file, args.device, args.ckpt_step)

    # Build the canonical MusicCaps dataset once to discover the track pool.
    base_dataset = MusicCaps(
        data_dir=args.data_dir,
        split=args.split,
        audio_loader=args.audio_loader,
        caption_type="caption",
        sr=args.sr,
        duration=args.segment_size,
        audio_enc=args.audio_enc,
    )

    # Restrict the track pool to ids we actually have rewrites for — so the
    # comparison across levels is on the same fixed pool.
    kept_ids = set(id_to_levels.keys())
    base_dataset.annotations = base_dataset.annotations[
        base_dataset.annotations[base_dataset.id_col].astype(str).isin(kept_ids)
    ].reset_index(drop=True)
    base_dataset.ytid_to_idx = {
        ytid: idx
        for idx, ytid in enumerate(base_dataset.annotations[base_dataset.id_col])
    }

    # The track pool is the full set of unique ytids in the restricted annotations.
    unique_tracks = sorted(
        set(base_dataset.annotations[base_dataset.id_col].astype(str).tolist())
    )
    print(f"Track pool size after filtering: {len(unique_tracks)}")

    # ---- Compute audio features once ----
    audio_features = compute_audio_features(
        model, base_dataset, unique_tracks, args.device
    )
    track2idx = {t: i for i, t in enumerate(unique_tracks)}

    out_model_dir = args.out_dir / args.model_tag
    out_model_dir.mkdir(parents=True, exist_ok=True)

    curve: Dict[str, Dict[str, float]] = {}
    n_queries_per_level: Dict[str, int] = {}

    # ---- Per level text features + metrics ----
    for lvl in args.levels:
        print(f"\n=== {lvl} ===")
        id_to_caption = {_id: d[lvl] for _id, d in id_to_levels.items() if d.get(lvl)}

        # Fresh dataset with swapped captions for THIS level.
        level_dataset = MusicCaps(
            data_dir=args.data_dir,
            split=args.split,
            audio_loader=args.audio_loader,
            caption_type="caption",
            sr=args.sr,
            duration=args.segment_size,
            audio_enc=args.audio_enc,
        )
        n_keep, n_drop = patch_musiccaps_captions(level_dataset, id_to_caption)
        print(f"  patched: kept={n_keep} dropped={n_drop}")

        _, _, query2track = query_processor(level_dataset, "caption")
        # Use the SAME track ordering as the audio features.
        unique_query = list(query2track.keys())
        query2track_idx = get_query2target_idx(query2track, track2idx)

        n_queries_per_level[lvl] = len(unique_query)
        print(f"  #queries={len(unique_query)}")

        query_features = compute_text_features(
            model, unique_query, args.use_audio_type_token
        )

        query2audio = get_task_predictions(query_features, audio_features)
        medrank_val, _ = median_rank(unique_query, query2track_idx, query2audio)

        metrics = {
            "recall@1": recall(query2audio, unique_query, query2track_idx, top_k=1),
            "recall@5": recall(query2audio, unique_query, query2track_idx, top_k=5),
            "recall@10": recall(query2audio, unique_query, query2track_idx, top_k=10),
            "map@10": mean_average_precision(
                query2audio, unique_query, query2track_idx, top_k=10
            ),
            "mean_reciprocal_rank": mean_reciprocal_rank(
                query2audio, unique_query, query2track_idx
            ),
            "median_rank": medrank_val,
        }
        print(f"  {metrics}")
        curve[lvl] = metrics

        lvl_dir = out_model_dir / lvl
        lvl_dir.mkdir(parents=True, exist_ok=True)
        with open(lvl_dir / "caption.json", "w") as f:
            json.dump(metrics, f, indent=2)

    curve_payload = {
        "model_tag": args.model_tag,
        "cfg_file": args.cfg_file,
        "ckpt_step": args.ckpt_step,
        "n_tracks": len(unique_tracks),
        "n_queries_per_level": n_queries_per_level,
        "metrics": curve,
    }
    with open(out_model_dir / "curve.json", "w") as f:
        json.dump(curve_payload, f, indent=2)
    print(f"\nCurve saved to {out_model_dir / 'curve.json'}")


if __name__ == "__main__":
    main()
