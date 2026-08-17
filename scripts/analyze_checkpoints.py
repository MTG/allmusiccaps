"""Analyze checkpoints to find optimal stopping point.

This script evaluates multiple checkpoints from a training run on downstream
tasks to identify when overfitting begins (i.e., when train/val loss decreases
but downstream performance degrades).

Usage:
    python scripts/analyze_checkpoints.py \
        --checkpoint_dir /path/to/checkpoints \
        --eval_task gtzan \
        --output_dir ./checkpoint_analysis
"""

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def find_checkpoints(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    """Find all checkpoints and extract their step/epoch numbers."""
    checkpoints = []

    for ckpt_file in checkpoint_dir.glob("*.ckpt"):
        # Try to extract step or epoch from filename
        # Common patterns: epoch=X-step=Y.ckpt, step=Y.ckpt, epoch_X.ckpt
        name = ckpt_file.stem

        step_match = re.search(r"step[=_](\d+)", name)
        epoch_match = re.search(r"epoch[=_](\d+)", name)

        if step_match:
            step = int(step_match.group(1))
        elif epoch_match:
            # Approximate step from epoch (rough estimate)
            step = int(epoch_match.group(1)) * 1000  # placeholder
        else:
            continue

        checkpoints.append((step, ckpt_file))

    # Sort by step
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def evaluate_gtzan(model, device: str = "cuda") -> float:
    """Quick GTZAN zero-shot evaluation."""
    # GTZAN genre labels
    genres = [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock",
    ]

    # Create text queries
    queries = [f"This is a {genre} song." for genre in genres]

    # Get text embeddings
    with torch.no_grad():
        text_embeddings = model.forward_text(queries)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    # Load GTZAN test set
    try:
        from datasets import load_dataset

        dataset = load_dataset("marsyas/gtzan", split="train")
    except Exception as e:
        print(f"Could not load GTZAN: {e}")
        return -1.0

    correct = 0
    total = 0

    for sample in tqdm(dataset, desc="Evaluating GTZAN"):
        audio = torch.tensor(sample["audio"]["array"]).float()
        sr = sample["audio"]["sampling_rate"]
        label = sample["genre"]

        # Resample if needed (GTZAN is 22050 Hz, model expects 24000 Hz)
        if sr != 24000:
            import torchaudio

            audio = torchaudio.functional.resample(audio, sr, 24000)

        # Take 10-second segment
        segment_samples = 24000 * 10
        if len(audio) > segment_samples:
            start = (len(audio) - segment_samples) // 2
            audio = audio[start : start + segment_samples]
        elif len(audio) < segment_samples:
            # Pad
            audio = torch.nn.functional.pad(audio, (0, segment_samples - len(audio)))

        audio = audio.unsqueeze(0).to(device)

        # Get audio embedding
        with torch.no_grad():
            audio_embedding = model.forward_audio(audio)
            audio_embedding = audio_embedding / audio_embedding.norm(
                dim=-1, keepdim=True
            )

        # Compute similarities
        similarities = (audio_embedding @ text_embeddings.T).squeeze()
        predicted = similarities.argmax().item()

        if genres[predicted] == label:
            correct += 1
        total += 1

    accuracy = correct / total
    return accuracy


def evaluate_retrieval_quick(
    model, dataset_name: str = "musiccaps", device: str = "cuda", max_samples: int = 100
) -> float:
    """Quick retrieval evaluation on a subset."""
    try:
        from datasets import load_dataset

        if dataset_name == "musiccaps":
            dataset = load_dataset("google/MusicCaps", split="test")
        elif dataset_name == "songdescriber":
            dataset = load_dataset("mulab-mir/song-describer", split="train")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    except Exception as e:
        print(f"Could not load {dataset_name}: {e}")
        return -1.0

    # Subsample
    indices = list(range(min(max_samples, len(dataset))))

    # This is a simplified version - for full evaluation use downstream_retrieval.py
    # Here we just compute average similarity between matched pairs

    total_sim = 0
    count = 0

    for idx in tqdm(indices, desc=f"Evaluating {dataset_name}"):
        sample = dataset[idx]
        caption = sample.get("caption", "")

        if not caption:
            continue

        # Get text embedding
        with torch.no_grad():
            text_emb = model.forward_text([caption])
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        # Note: Full audio loading would be needed here
        # For now, this is a placeholder
        count += 1

    return total_sim / count if count > 0 else -1.0


def load_model_from_checkpoint(
    checkpoint_path: Path, config_path: Path = None, device: str = "cuda"
):
    """Load CLAP model from a checkpoint."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from amclap import get_model

    # If config provided, use it
    if config_path and config_path.exists():
        model = get_model(config_file=str(config_path), device=device)
    else:
        # Try to infer config from checkpoint directory
        model = get_model(config_file=str(checkpoint_path), device=device)

    return model


def main():
    parser = argparse.ArgumentParser(description="Analyze checkpoints for overfitting")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing checkpoints",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Gin config file for model loading",
    )
    parser.add_argument(
        "--eval_task",
        type=str,
        default="gtzan",
        choices=["gtzan", "musiccaps_quick", "songdescriber_quick"],
        help="Evaluation task",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoint_analysis",
        help="Output directory",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for evaluation"
    )
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=None,
        help="Maximum number of checkpoints to evaluate",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find checkpoints
    checkpoints = find_checkpoints(checkpoint_dir)
    print(f"Found {len(checkpoints)} checkpoints")

    if args.max_checkpoints:
        # Sample evenly
        step = max(1, len(checkpoints) // args.max_checkpoints)
        checkpoints = checkpoints[::step]
        print(f"Sampling {len(checkpoints)} checkpoints")

    # Evaluate each checkpoint
    results = []

    for step, ckpt_path in checkpoints:
        print(f"\nEvaluating checkpoint at step {step}: {ckpt_path.name}")

        try:
            # Load model
            model = load_model_from_checkpoint(
                ckpt_path,
                Path(args.config_file) if args.config_file else None,
                args.device,
            )
            model.eval()

            # Evaluate
            if args.eval_task == "gtzan":
                score = evaluate_gtzan(model, args.device)
            elif args.eval_task == "musiccaps_quick":
                score = evaluate_retrieval_quick(model, "musiccaps", args.device)
            elif args.eval_task == "songdescriber_quick":
                score = evaluate_retrieval_quick(model, "songdescriber", args.device)

            results.append(
                {
                    "step": step,
                    "checkpoint": ckpt_path.name,
                    "score": score,
                }
            )

            print(f"  Score: {score:.4f}")

            # Clean up
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Error: {e}")
            results.append(
                {
                    "step": step,
                    "checkpoint": ckpt_path.name,
                    "score": None,
                    "error": str(e),
                }
            )

    # Save results
    results_path = output_dir / "checkpoint_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Plot results
    valid_results = [r for r in results if r.get("score") is not None]
    if valid_results:
        steps = [r["step"] for r in valid_results]
        scores = [r["score"] for r in valid_results]

        plt.figure(figsize=(10, 6))
        plt.plot(steps, scores, "b-o", markersize=8)
        plt.xlabel("Training Step")
        plt.ylabel(f"{args.eval_task} Score")
        plt.title("Downstream Performance vs Training Step")
        plt.grid(True, alpha=0.3)

        # Mark best checkpoint
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        plt.axvline(
            x=steps[best_idx],
            color="r",
            linestyle="--",
            label=f"Best: step={steps[best_idx]}, score={scores[best_idx]:.4f}",
        )
        plt.legend()

        plt.tight_layout()
        plot_path = output_dir / "checkpoint_performance.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot to {plot_path}")

        # Print summary
        print("\n=== Summary ===")
        print(f"Best checkpoint: step={steps[best_idx]}, score={scores[best_idx]:.4f}")
        print(f"Final checkpoint: step={steps[-1]}, score={scores[-1]:.4f}")
        if scores[-1] < scores[best_idx]:
            print(
                f"⚠️  Overfitting detected: final is {scores[best_idx] - scores[-1]:.4f} worse than best"
            )


if __name__ == "__main__":
    main()
