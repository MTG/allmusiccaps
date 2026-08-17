#!/usr/bin/env python
"""
Analyze prompt token lengths to help tune MAX_MODEL_LEN.

This script loads the dataset, generates prompts for all samples,
and reports token count statistics to help optimize vLLM configuration.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from prompt_templates import format_prompt, prompt_v3

import transformers.tokenization_utils_base as tub


def _no_patch(*args, **kwargs):
    return args[0]


tub.PreTrainedTokenizerBase._patch_mistral_regex = staticmethod(_no_patch)


def load_data(data_path: Path) -> pd.DataFrame:
    """Load the pickle dataset.

    Args:
        data_path: Path to the pickle file

    Returns:
        DataFrame with all samples
    """
    with open(data_path, "rb") as f:
        df = pickle.load(f)

    if df is None or len(df) == 0:
        raise ValueError(f"Loaded dataset from {data_path} is empty or invalid")

    return df


def analyze_prompt_lengths(
    df: pd.DataFrame,
    model_name: str,
    max_samples: int = None,
    max_model_len: int = None,
    use_vllm: bool = False,
) -> dict:
    """Analyze prompt lengths across the dataset.

    Args:
        df: DataFrame with metadata
        model_name: Name or path of model (used for tokenizer)
        max_samples: Optional limit on number of samples to analyze
        max_model_len: Optional max context length to check against
        use_vllm: If True, use vLLM's tokenizer (recommended for matching inference)

    Returns:
        Dictionary with statistics, the longest prompt, and IDs of oversized samples
    """
    if use_vllm:
        print(f"Loading vLLM-compatible tokenizer from model: {model_name}")
        from vllm.tokenizers import get_tokenizer

        # Use vLLM's tokenizer loader without loading the model weights
        tokenizer = get_tokenizer(model_name, trust_remote_code=True)
        print("Using vLLM's tokenizer loader (matches inference pipeline)")
    else:
        print(f"Loading transformers tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("WARNING: Using transformers tokenizer. Results may differ from vLLM!")

    if max_samples:
        df = df.head(max_samples)

    print(f"Analyzing {len(df)} samples...")

    token_counts = []
    char_counts = []
    longest_prompt = ""
    longest_tokens = 0
    longest_idx = None
    oversized_ids = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        prompt = format_prompt(row, prompt_v3)
        char_count = len(prompt)
        token_count = len(tokenizer.encode(prompt))

        token_counts.append(token_count)
        char_counts.append(char_count)

        if max_model_len and token_count > max_model_len:
            oversized_ids.append(idx)

        if token_count > longest_tokens:
            longest_tokens = token_count
            longest_prompt = prompt
            longest_idx = idx

    stats = {
        "total_samples": len(df),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "mean_tokens": sum(token_counts) / len(token_counts),
        "median_tokens": sorted(token_counts)[len(token_counts) // 2],
        "min_chars": min(char_counts),
        "max_chars": max(char_counts),
        "mean_chars": sum(char_counts) / len(char_counts),
        "longest_prompt": longest_prompt,
        "longest_idx": longest_idx,
        "max_model_len": max_model_len,
        "oversized_ids": oversized_ids,
        "num_oversized": len(oversized_ids),
    }

    # Calculate percentiles
    sorted_tokens = sorted(token_counts)
    for percentile in [90, 95, 99]:
        idx = int(len(sorted_tokens) * percentile / 100)
        stats[f"p{percentile}_tokens"] = sorted_tokens[idx]

    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("../../notebooks/allmusic_youtube_discogs_reviews.pkl"),
        help="Path to pickle dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path (used to load tokenizer)",
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Use vLLM's tokenizer instead of transformers (recommended, matches inference)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit analysis to N samples (useful for quick tests)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Target max model context length to check against",
    )
    parser.add_argument(
        "--show-longest",
        action="store_true",
        help="Print the longest prompt to stdout",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_path}")
    df = load_data(args.data_path)
    print(f"Loaded {len(df)} total samples")

    # Analyze
    stats = analyze_prompt_lengths(
        df, args.model, args.max_samples, args.max_model_len, args.use_vllm
    )

    # Report
    print("\n" + "=" * 60)
    print("PROMPT LENGTH STATISTICS")
    print("=" * 60)
    print(f"Samples analyzed:     {stats['total_samples']}")
    print(f"\nCharacter counts:")
    print(f"  Min:                {stats['min_chars']:,}")
    print(f"  Mean:               {stats['mean_chars']:,.1f}")
    print(f"  Max:                {stats['max_chars']:,}")
    print(f"\nToken counts:")
    print(f"  Min:                {stats['min_tokens']}")
    print(f"  Mean:               {stats['mean_tokens']:.1f}")
    print(f"  Median:             {stats['median_tokens']}")
    print(f"  90th percentile:    {stats['p90_tokens']}")
    print(f"  95th percentile:    {stats['p95_tokens']}")
    print(f"  99th percentile:    {stats['p99_tokens']}")
    print(f"  Max:                {stats['max_tokens']}")
    print(f"\nLongest prompt index: {stats['longest_idx']}")

    if stats["max_model_len"]:
        print(f"\nTarget max_model_len: {stats['max_model_len']}")
        print(
            f"Samples exceeding limit: {stats['num_oversized']} / {stats['total_samples']}"
        )
        if stats["num_oversized"] > 0:
            print(
                f"  Percentage: {100 * stats['num_oversized'] / stats['total_samples']:.2f}%"
            )
    print("=" * 60)

    # Recommendations
    print("\nRECOMMENDATIONS:")
    max_tokens = stats["max_tokens"]
    # Add buffer for output tokens (e.g., 512)
    output_buffer = 512
    recommended = max_tokens + output_buffer

    # Round up to nice numbers
    if recommended <= 2048:
        suggested = 2048
    elif recommended <= 4096:
        suggested = 4096
    elif recommended <= 8192:
        suggested = 8192
    else:
        suggested = ((recommended // 1024) + 1) * 1024

    print(f"  Input tokens (max):  {max_tokens}")
    print(f"  Output buffer:       {output_buffer}")
    print(f"  Total needed:        {recommended}")
    print(f"  Suggested value:     {suggested}")
    print(f"\n  Set: MAX_MODEL_LEN={suggested}")
    print("=" * 60)

    if not args.use_vllm:
        print("\n  WARNING: You used transformers tokenizer!")
        print("  Re-run with --use-vllm for accurate token counts matching inference.")

    # Write oversized sample IDs to file if any exist
    if stats["num_oversized"] > 0:
        output_file = Path("oversized_samples.txt")
        with open(output_file, "w") as f:
            for sample_id in stats["oversized_ids"]:
                f.write(f"{sample_id}\n")
        print(f"\nWrote {stats['num_oversized']} oversized sample IDs to {output_file}")

    if args.show_longest:
        print("\n" + "=" * 60)
        print("LONGEST PROMPT:")
        print("=" * 60)
        print(stats["longest_prompt"])
        print("=" * 60)


if __name__ == "__main__":
    main()
