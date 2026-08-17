"""Evaluate music similarity using the DimSim dataset.

The DimSim dataset contains ~4,000 triplets of 3-second audio clips from the
Million Song Dataset. Each triplet has an anchor, song1, and song2, with human
annotations indicating which song sounds more similar to the anchor.

This script computes audio embeddings for each clip, measures cosine distances
in the latent space, and reports accuracy of predicting human judgments.
"""

import argparse
import csv
import json
import os
import essentia.standard as es

import torch
import torch.nn.functional as F
from tqdm import tqdm

torch.set_float32_matmul_precision("medium")

# Add src directory to path for imports

from .. import get_model


def parse_metadata(metadata_file):
    """Parse DimSim CSV metadata into a list of triplets.

    Returns a list of dicts with keys:
        anchor_id, anchor_start, song1_id, song1_start, song2_id, song2_start,
        song1_vote (1 if song1 is more similar to anchor, 0 otherwise)
    """
    triplets = []
    with open(metadata_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            triplets.append(
                {
                    "anchor_id": row["anchor_id"],
                    "anchor_start": float(row["anchor_start_seconds"]),
                    "song1_id": row["song1_id"],
                    "song1_start": float(row["song1_start_seconds"]),
                    "song2_id": row["song2_id"],
                    "song2_start": float(row["song2_start_seconds"]),
                    "song1_vote": int(row["song1_vote"]),
                }
            )
    return triplets


def collect_unique_clips(triplets):
    """Collect all unique (track_id, start_seconds) pairs across triplets."""
    clips = set()
    for t in triplets:
        clips.add((t["anchor_id"], t["anchor_start"]))
        clips.add((t["song1_id"], t["song1_start"]))
        clips.add((t["song2_id"], t["song2_start"]))
    return clips


def load_clip_audio(audio_dir, track_id, start_seconds, sr, clip_duration=3.0):
    """Load a 3-second audio clip from an MP3 file."""

    audio_path = os.path.join(audio_dir, f"{track_id}.mp3")
    audio = es.MonoLoader(filename=audio_path, sampleRate=sr)()

    # TODO: Check if the dataset is full audio or already the segment
    start_sample = int(start_seconds * sr)
    end_sample = start_sample + int(clip_duration * sr)

    # Clamp to audio length
    end_sample = min(end_sample, len(audio))

    segment = audio[start_sample:end_sample]

    return torch.tensor(segment).unsqueeze(0)  # (1, T)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate music similarity with DimSim triplets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cfg_file",
        type=str,
        required=True,
        help="Path to model gin config file.",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing MP3 audio files (flat).",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        required=True,
        help="Path to DimSim CSV metadata file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for inference.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results.",
    )
    parser.add_argument(
        "--new_freq",
        type=int,
        default=24000,
        help="Audio sample rate in Hz.",
    )
    parser.add_argument(
        "--use_audio_type_token",
        action="store_true",
        help="Unused, kept for consistency with other scripts.",
    )
    parser.add_argument(
        "--ckpt_step",
        type=int,
        default=None,
        help="Load checkpoint at this specific training step",
    )
    parser.add_argument(
        "--avg_last_n",
        type=int,
        default=None,
        help="Average the last N checkpoints",
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.cfg_file}")
    model = get_model(
        config_file=args.cfg_file,
        device=args.device,
        weights_only=False,
        ckpt_step=args.ckpt_step,
        avg_last_n=args.avg_last_n,
    )
    model.eval()

    # Parse metadata
    print(f"Loading metadata from {args.metadata_file}")
    triplets = parse_metadata(args.metadata_file)
    print(f"Loaded {len(triplets)} triplets")

    # Collect unique clips
    unique_clips = collect_unique_clips(triplets)
    print(f"Found {len(unique_clips)} unique clips")

    # Compute embeddings for all unique clips
    embeddings = {}
    skipped = 0
    for track_id, start_seconds in tqdm(unique_clips, desc="Computing embeddings"):
        try:
            audio = load_clip_audio(
                args.audio_dir, track_id, start_seconds, args.new_freq
            )
            audio = audio.to(args.device)

            with torch.inference_mode():
                emb = model.forward_audio(audio)  # (1, D)

            embeddings[(track_id, start_seconds)] = emb.detach().cpu()
        except Exception as e:
            print(f"Warning: failed to process {track_id} at {start_seconds}s: {e}")
            skipped += 1

    if skipped > 0:
        print(f"Skipped {skipped}/{len(unique_clips)} clips due to errors")

    # Evaluate triplets
    correct = 0
    total = 0
    for t in triplets:
        anchor_key = (t["anchor_id"], t["anchor_start"])
        song1_key = (t["song1_id"], t["song1_start"])
        song2_key = (t["song2_id"], t["song2_start"])

        if (
            anchor_key not in embeddings
            or song1_key not in embeddings
            or song2_key not in embeddings
        ):
            continue

        anchor_emb = embeddings[anchor_key]
        song1_emb = embeddings[song1_key]
        song2_emb = embeddings[song2_key]

        # Cosine distance = 1 - cosine_similarity
        dist1 = 1 - F.cosine_similarity(anchor_emb, song1_emb, dim=-1).item()
        dist2 = 1 - F.cosine_similarity(anchor_emb, song2_emb, dim=-1).item()

        # Predict the closer song
        pred_song1 = dist1 < dist2
        gt_song1 = t["song1_vote"] == 1

        if pred_song1 == gt_song1:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nDimSim Accuracy: {accuracy:.4f} ({correct}/{total})")

    # Save results
    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "skipped_clips": skipped,
    }

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
