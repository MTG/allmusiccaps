#!/usr/bin/env python
"""
Generate structured music descriptions with a Transformers backend.

This script mirrors the transformers-based portion of notebooks/model_comp.ipynb.
It loads review/tag metadata, builds prompts, and uses an Outlines generator to
produce structured fields defined by the MusicDescription schema.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import outlines
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from prompt_templates import format_prompt
from vllm import LLM, SamplingParams


class MusicDescription(BaseModel):
    music_style: str
    vibe: str
    tempo_energy: str
    instrumentation: str
    production_notes: str


def load_sample(data_path: Path, n_samples: int, seed: int) -> pd.DataFrame:
    """Load a pickle dataset and return a sampled DataFrame.

    Args:
        data_path: Path to the pickle file containing the dataset
        n_samples: Number of samples to extract (clamped to available data)
        seed: Random seed for reproducible sampling

    Returns:
        A sampled DataFrame with at most n_samples rows

    Raises:
        ValueError: If the loaded dataset is empty or invalid
    """
    with open(data_path, "rb") as f:
        df = pickle.load(f)

    if df is None or len(df) == 0:
        raise ValueError(f"Loaded dataset from {data_path} is empty or invalid")

    # Clamp sample size to valid range [1, len(df)]
    clamped_n_samples = max(1, min(n_samples, len(df)))
    return df.sample(n=clamped_n_samples, random_state=seed)


def build_generator(args: argparse.Namespace) -> outlines.Generator:
    """Build an Outlines generator with VLLM backend for structured output.

    Args:
        args: Command-line arguments containing model configuration

    Returns:
        An Outlines Generator configured to produce MusicDescription objects
    """
    # Initialize VLLM with configuration from args
    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer or None,
        tensor_parallel_size=args.tensor_parallel,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.enable_prefix_caching,
        enforce_eager=args.enforce_eager,
    )

    # Wrap VLLM model for Outlines compatibility
    model = outlines.from_vllm_offline(llm)
    return outlines.Generator(model, MusicDescription)


def build_sampling_params(args: argparse.Namespace) -> SamplingParams:
    """Build sampling parameters from command-line arguments.

    Args:
        args: Command-line arguments containing sampling configuration

    Returns:
        SamplingParams object for text generation
    """
    stop_tokens = args.stop if args.stop else None
    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=args.n,
        stop=stop_tokens,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        repetition_penalty=args.repetition_penalty,
    )


def parse_music_json(text: str) -> Tuple[Dict[str, str], bool]:
    """Parse raw model output into structured music description fields.

    Args:
        text: Raw text output from the model

    Returns:
        Tuple of (parsed_dict, is_valid) where parsed_dict contains all schema
        keys and is_valid indicates if parsing succeeded
    """
    schema_keys = [
        "music_style",
        "vibe",
        "tempo_energy",
        "instrumentation",
        "production_notes",
    ]
    default_payload = {key: "N/A" for key in schema_keys}

    # Try direct JSON parsing
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return {key: str(obj.get(key, "N/A")) for key in schema_keys}, True
    except json.JSONDecodeError:
        pass

    # Try extracting JSON between braces
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            obj = json.loads(text[first_brace : last_brace + 1])
            if isinstance(obj, dict):
                return {key: str(obj.get(key, "N/A")) for key in schema_keys}, True
        except json.JSONDecodeError:
            pass

    return default_payload, False


def generate_batch(
    rows: List[dict],
    generator: outlines.Generator,
    sampling_params: SamplingParams,
) -> List[dict]:
    """Generate structured music descriptions for a batch of input rows.

    Args:
        rows: List of input metadata dictionaries for this batch
        generator: Outlines generator for structured output
        sampling_params: Parameters controlling text generation

    Returns:
        List of result dictionaries containing input, prompt, output, and validity
    """
    prompts = [format_prompt(pd.Series(row)) for row in rows]
    outputs = generator.batch(prompts, sampling_params=sampling_params)

    # Length-aware ordering
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    prompts = [prompts[i] for i in order]
    rows = [rows[i] for i in order]

    results = []
    for row, prompt, output in zip(rows, prompts, outputs):
        output_text = output[0] if isinstance(output, list) else output
        parsed_data, is_valid = parse_music_json(output_text)

        results.append(
            {
                "input_metadata": row,
                "prompt": prompt,
                "raw_text": output_text,
                "generated": parsed_data,
                "is_valid_json": is_valid,
            }
        )

    return results


def save_jsonl(records: List[dict], out_path: Path, append: bool = False) -> None:
    """Save records to a JSON Lines file.

    Args:
        records: List of dictionaries to save
        out_path: Path to the output .jsonl file
        append: If True, append to existing file; otherwise overwrite
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(out_path, mode) as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def discover_existing_ids(out_dir: Path) -> Set[str]:
    """Find all sample IDs that have already been processed.

    Args:
        out_dir: Directory to scan for existing outputs

    Returns:
        Set of sample IDs that have already been processed
    """
    if not out_dir.exists():
        return set()

    suffix = ".response.json"
    return {path.name[: -len(suffix)] for path in out_dir.rglob(f"*{suffix}")}


def persist_sample_outputs(
    out_dir: Path, sample_id: str, prompt: str, payload: Dict[str, str]
) -> None:
    """Save prompt and generated output for a single sample.

    Organizes outputs in sharded directories (first 2 chars of sample_id)
    to avoid filesystem limitations with too many files in one directory.

    Args:
        out_dir: Base output directory
        sample_id: Unique identifier for the sample
        prompt: Input prompt that was used
        payload: Generated structured output
    """
    # Create shard directory based on first 2 characters of sample_id
    # This helps distribute files across subdirectories
    shard_dir = out_dir / sample_id[:2]
    shard_dir.mkdir(parents=True, exist_ok=True)

    query_path = shard_dir / f"{sample_id}.query"
    response_path = shard_dir / f"{sample_id}.response.json"

    query_path.write_text(prompt)
    response_path.write_text(json.dumps(payload, indent=2))


def find_forbidden_tokens(df_sample: pd.DataFrame) -> set[str]:
    """Extract artist names to check for data leakage.

    Args:
        df_sample: DataFrame containing 'artist' column

    Returns:
        Set of lowercase artist names from the dataset
    """
    return set(str(artist).lower() for artist in df_sample["artist"].dropna().unique())


def count_artist_leakage(results: List[dict], forbidden_tokens: set[str]) -> int:
    """Count how many generated outputs contain forbidden artist names.

    This helps detect if the model is leaking training data by directly
    mentioning artist names that should not appear in the output.

    Args:
        results: List of generation results with 'generated' fields
        forbidden_tokens: Set of lowercase artist names to check for

    Returns:
        Number of generated text fields containing forbidden tokens
    """

    def contains_forbidden(text: str) -> bool:
        """Check if text contains any forbidden tokens."""
        text_lower = text.lower()
        return any(token in text_lower for token in forbidden_tokens)

    violation_count = 0
    for result in results:
        for text in result["generated"].values():
            if isinstance(text, str) and text != "N/A" and contains_forbidden(text):
                violation_count += 1

    return violation_count


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    # Dataset and output
    p.add_argument(
        "--data-path", type=Path, default=Path("allmusic_youtube_discogs_reviews.pkl")
    )
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-path", type=Path, default=Path("quick_test_outputs_vllm.jsonl")
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("sample_outputs_vllm"),
        help="Directory to store per-sample query/response files",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite even if outputs already exist",
    )
    p.add_argument("--check-artist-leakage", action="store_true")

    # Model / engine
    p.add_argument("--model", type=str, required=True, help="Model name or local path")
    p.add_argument(
        "--tokenizer", type=str, default=None, help="Optional tokenizer name or path"
    )
    p.add_argument(
        "--tensor-parallel", type=int, default=1, help="Tensor parallel size (GPUs)"
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Model dtype: auto|float16|bfloat16|float32",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading models with custom code",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Max model length (tokens)",
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization target [0-1]",
    )
    p.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="Enable prefix caching for shared prefixes",
    )
    p.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Force eager execution (can aid debugging)",
    )

    # Inference controls
    p.add_argument(
        "--max-tokens", type=int, default=512, help="Max new tokens per completion"
    )
    p.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    p.add_argument("--top-p", type=float, default=1.0, help="Top-p nucleus sampling")
    p.add_argument(
        "--top-k", type=int, default=-1, help="Top-k sampling (<=0 disables)"
    )
    p.add_argument("--n", type=int, default=1, help="Number of candidates per prompt")
    p.add_argument(
        "--stop", type=str, nargs="*", default=None, help="Optional stop strings"
    )
    p.add_argument("--presence-penalty", type=float, default=0.0)
    p.add_argument("--frequency-penalty", type=float, default=0.0)
    p.add_argument("--repetition-penalty", type=float, default=1.0)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load and prepare data
    df_sample = load_sample(args.data_path, args.n_samples, args.seed)
    df_sample["sample_id"] = df_sample.index.astype(str)
    print(f"Loaded {len(df_sample)} examples from {args.data_path}")

    # Filter out already-processed samples
    rows = df_sample.to_dict(orient="records")
    if not args.force:
        existing_ids = discover_existing_ids(args.out_dir)
        if existing_ids:
            print(f"Found {len(existing_ids)} existing samples, filtering...")
            rows = [r for r in rows if r["sample_id"] not in existing_ids]
            print(f"Processing {len(rows)} new samples")

    if not rows:
        print("No samples to process; exiting.")
        return

    # Generate descriptions batch by batch
    generator = build_generator(args)
    sampling_params = build_sampling_params(args)

    total_valid = 0
    total_invalid = 0

    for i in tqdm(range(0, len(rows), args.batch_size), desc="Processing batches"):
        batch_rows = rows[i : i + args.batch_size]
        batch_results = generate_batch(batch_rows, generator, sampling_params)

        # Filter valid results
        valid_results = [r for r in batch_results if r["is_valid_json"]]
        total_valid += len(valid_results)
        total_invalid += len(batch_results) - len(valid_results)

        # Persist individual sample outputs
        for r in valid_results:
            persist_sample_outputs(
                args.out_dir,
                r["input_metadata"]["sample_id"],
                r["prompt"],
                r["generated"],
            )

        # Append valid results to jsonl file
        if valid_results:
            save_jsonl(valid_results, args.out_path, append=(i > 0))

    if total_valid == 0:
        print("No valid outputs produced; nothing saved.")
        return

    print(f"Saved {total_valid} valid results to {args.out_path} and {args.out_dir}")
    if total_invalid > 0:
        print(f"Warning: {total_invalid} samples had invalid JSON")

    # Optional artist leakage check
    if args.check_artist_leakage and total_valid > 0:
        forbidden_tokens = find_forbidden_tokens(df_sample)
        # Read results from saved file
        with open(args.out_path) as f:
            saved_results = [json.loads(line) for line in f]
        leakage_count = count_artist_leakage(saved_results, forbidden_tokens)
        print(f"Artist leakage: {leakage_count} violations")


if __name__ == "__main__":
    main()
