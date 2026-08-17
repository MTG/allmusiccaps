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
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple

import outlines
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from prompt_templates import format_prompt, prompt_v3
from vllm import LLM, SamplingParams


class MusicDescription(BaseModel):
    music_style: str
    mood: str
    energy: str
    tempo: str
    instrumentation: str
    production_style: str


def load_sample(data_path: Path, n_samples: int, seed: int) -> pd.DataFrame:
    """Load a pickle dataset and return a sampled DataFrame.

    Args:
        data_path: Path to the pickle file containing the dataset
        n_samples: Number of samples to extract; -1 means full dataset
        seed: Random seed for reproducible sampling (ignored when n_samples=-1)

    Returns:
        A sampled DataFrame with at most n_samples rows, or full dataset if -1

    Raises:
        ValueError: If the loaded dataset is empty or invalid
    """
    with open(data_path, "rb") as f:
        df = pickle.load(f)

    if df is None or len(df) == 0:
        raise ValueError(f"Loaded dataset from {data_path} is empty or invalid")

    # -1 means use the whole dataset
    if n_samples == -1:
        return df

    # Clamp sample size to valid range [1, len(df)]
    clamped_n_samples = max(1, min(n_samples, len(df)))
    return df.sample(n=clamped_n_samples, random_state=seed)


def build_generator(
    args: argparse.Namespace, quantization_bits: int = 0
) -> outlines.Generator:
    """Build an Outlines generator with VLLM backend for structured output.

    Args:
        args: Command-line arguments containing model configuration

    Returns:
        An Outlines Generator configured to produce MusicDescription objects
    """

    quantization = None
    if quantization_bits == 4:
        quantization = "bitsandbytes"
    elif quantization_bits == 8:
        raise ValueError(
            "8-bit quantization is not currently supported with VLLM backend."
        )
    elif quantization_bits != 0:
        raise ValueError(
            f"Unsupported quantization_bits: {quantization_bits}. Choose 4, 8, or None."
        )
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
        quantization=quantization,
        max_num_seqs=args.max_num_seqs,
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


def parse_music_json(text: str) -> Tuple[Dict[str, str], bool, str]:
    """Parse raw model output into structured music description fields.

    Args:
        text: Raw text output from the model

    Returns:
        Tuple of (parsed_dict, is_valid, error_reason) where parsed_dict contains all schema
        keys, is_valid indicates if parsing succeeded, and error_reason explains why it failed
    """
    schema_keys = [
        "music_style",
        "mood",
        "tempo",
        "energy",
        "instrumentation",
        "production_style",
    ]
    default_payload = {key: "None" for key in schema_keys}

    # Check if output is empty
    if not text or text.strip() == "":
        return (
            default_payload,
            False,
            "empty_output",
        )

    # Try direct JSON parsing
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            parsed = {key: str(obj.get(key, "None")) for key in schema_keys}
            # Check for missing fields
            missing = [k for k in schema_keys if k not in obj]
            if missing:
                return parsed, False, f"missing_fields:{','.join(missing)}"
            return parsed, True, "success"
        else:
            return default_payload, False, "not_dict"
    except json.JSONDecodeError as e:
        error_msg = str(e).split("\n")[0][:50]  # Truncate error message
        pass

    # Try extracting JSON between braces
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            obj = json.loads(text[first_brace : last_brace + 1])
            if isinstance(obj, dict):
                parsed = {key: str(obj.get(key, "None")) for key in schema_keys}
                missing = [k for k in schema_keys if k not in obj]
                if missing:
                    return (
                        parsed,
                        False,
                        f"extracted_missing_fields:{','.join(missing)}",
                    )
                return parsed, True, "success_extracted"
        except json.JSONDecodeError as e:
            return default_payload, False, "extracted_invalid_json"

    # No JSON found at all
    if "{" not in text and "}" not in text:
        return default_payload, False, "no_json_markers"

    return default_payload, False, "unknown_parse_error"


def is_default_payload(payload: Dict[str, str]) -> bool:
    """Detect model outputs that return the placeholder/default payload."""

    expected_keys = {
        "music_style",
        "mood",
        "tempo",
        "energy",
        "instrumentation",
        "production_style",
    }
    normalized = {
        key: str(payload.get(key, "")).strip().upper() for key in expected_keys
    }
    return all(val in {"N/A", "NA"} for val in normalized.values())


def normalize_text_input(text: str) -> str:
    """Normalize text before sending it to the tokenizer."""
    if not isinstance(text, str):
        text = str(text)
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.strip()


def load_blacklist(path: Path) -> Set[str]:
    """Load blacklist of sample_ids to skip."""
    if not path.exists():
        return set()
    with open(path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def generate_batch(
    rows: List[dict],
    generator: outlines.Generator,
    sampling_params: SamplingParams,
    prompot_template: str,
) -> List[dict]:
    """Generate structured music descriptions for a batch of input rows.

    Args:
        rows: List of input metadata dictionaries for this batch
        generator: Outlines generator for structured output
        sampling_params: Parameters controlling text generation

    Returns:
        List of result dictionaries containing input, prompt, output, and validity
    """
    # Build and normalize prompts
    prompts = [format_prompt(pd.Series(row), prompot_template) for row in rows]
    prompts = [normalize_text_input(p) for p in prompts]

    # Length-aware ordering for potential packing efficiency in vLLM
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    ordered_prompts = [prompts[i] for i in order]
    ordered_rows = [rows[i] for i in order]

    # Generate once with ordered prompts
    outputs = generator.batch(ordered_prompts, sampling_params=sampling_params)

    results = []
    for row, prompt, output in zip(ordered_rows, ordered_prompts, outputs):
        output_text = output[0] if isinstance(output, list) else output
        parsed_data, is_valid, parse_error = parse_music_json(output_text)

        results.append(
            {
                "input_metadata": row,
                "prompt": prompt,
                "raw_text": output_text,
                "generated": parsed_data,
                "is_valid_json": is_valid,
                "parse_error": parse_error,
                "raw_text_length": len(output_text),
            }
        )

    return results


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
    out_dir: Path,
    sample_id: str,
    prompt: str,
    payload: Dict[str, str],
    raw_text: str = None,
) -> None:
    """Save prompt and generated output for a single sample.

    Organizes outputs in sharded directories (first 2 chars of sample_id)
    to avoid filesystem limitations with too many files in one directory.

    Args:
        out_dir: Base output directory
        sample_id: Unique identifier for the sample
        prompt: Input prompt that was used
        payload: Generated structured output
        raw_text: Raw model output (for debugging)
    """
    # Create shard directory based on first 2 characters of sample_id
    # This helps distribute files across subdirectories
    shard_dir = out_dir / sample_id[:2]
    shard_dir.mkdir(parents=True, exist_ok=True)

    query_path = shard_dir / f"{sample_id}.query"
    response_path = shard_dir / f"{sample_id}.response.json"
    raw_path = shard_dir / f"{sample_id}.raw.txt"

    query_path.write_text(prompt)
    response_path.write_text(json.dumps(payload, indent=2))

    # Store raw model output for debugging
    if raw_text:
        raw_path.write_text(raw_text)


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


def load_saved_results(out_dir: Path) -> List[dict]:
    """Load per-sample response files into a list of results dicts."""

    results: List[dict] = []
    if not out_dir.exists():
        return results

    suffix = ".response.json"
    for path in out_dir.rglob(f"*{suffix}"):
        try:
            payload = json.loads(path.read_text())
            results.append({"generated": payload})
        except json.JSONDecodeError:
            continue

    return results


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
        "--out-dir",
        type=Path,
        default=Path("sample_outputs_vllm"),
        help="Directory to store per-sample query/response files",
    )
    p.add_argument(
        "--disable-per-sample-save",
        action="store_true",
        help="Skip writing per-sample query/response files to reduce I/O and speed up inference",
    )
    p.add_argument(
        "--save-raw-text",
        action="store_true",
        help="Also save raw model outputs (.raw.txt). Disabled by default for speed",
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
    p.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Maximum number of sequences to process in parallel",
    )
    p.add_argument(
        "--quantization-bits",
        type=int,
        choices=[0, 4],
        default=0,
        help="Optional quantization: 0 (none) or 4 (bitsandbytes)",
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

    # Filtering
    p.add_argument(
        "--blacklist-file",
        type=Path,
        default=Path("blacklist.txt"),
        help="Path to a file with one sample_id per line to skip",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load and prepare data
    df_sample = load_sample(args.data_path, args.n_samples, args.seed)
    df_sample["sample_id"] = df_sample.index.astype(str)
    print(f"Loaded {len(df_sample)} examples from {args.data_path}")

    # Remove any blacklisted sample_ids
    blacklist = load_blacklist(args.blacklist_file)
    if blacklist:
        before = len(df_sample)
        df_sample = df_sample[~df_sample["sample_id"].isin(blacklist)]
        removed = before - len(df_sample)
        print(f"Skipped {removed} blacklisted samples (file: {args.blacklist_file})")

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
    generator = build_generator(args, quantization_bits=args.quantization_bits)
    sampling_params = build_sampling_params(args)

    total_valid = 0
    total_invalid = 0
    error_counts = {}  # Track error types
    output_lengths = []  # Track raw output lengths
    default_payload_count = 0  # Track how often the model returns placeholder payloads

    for i in tqdm(range(0, len(rows), args.batch_size), desc="Processing batches"):
        batch_rows = rows[i : i + args.batch_size]
        batch_results = generate_batch(
            batch_rows, generator, sampling_params, prompt_v3
        )

        # Filter valid results
        valid_results = [r for r in batch_results if r["is_valid_json"]]
        invalid_results = [r for r in batch_results if not r["is_valid_json"]]
        total_valid += len(valid_results)
        total_invalid += len(invalid_results)

        # Track error types and output lengths
        for r in invalid_results:
            error_type = r.get("parse_error", "unknown")
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            output_lengths.append(r.get("raw_text_length", 0))

        for r in valid_results:
            output_lengths.append(r.get("raw_text_length", 0))
            if is_default_payload(r["generated"]):
                default_payload_count += 1

        # Persist invalid sample outputs for debugging (optional)
        if not args.disable_per_sample_save:
            for r in invalid_results:
                persist_sample_outputs(
                    args.out_dir,
                    r["input_metadata"]["sample_id"],
                    r["prompt"],
                    r["generated"],
                    r["raw_text"] if args.save_raw_text else None,
                )

        # Persist individual sample outputs (optional)
        if not args.disable_per_sample_save:
            for r in valid_results:
                persist_sample_outputs(
                    args.out_dir,
                    r["input_metadata"]["sample_id"],
                    r["prompt"],
                    r["generated"],
                    r["raw_text"] if args.save_raw_text else None,
                )

    if total_valid == 0:
        print("No valid outputs produced; nothing saved.")
        return

    print(f"Saved {total_valid} valid results to {args.out_dir}")
    if total_invalid > 0:
        print(f"Warning: {total_invalid} samples had invalid JSON")

    # Save diagnostics report
    diagnostics = {
        "total_processed": total_valid + total_invalid,
        "total_valid": total_valid,
        "total_invalid": total_invalid,
        "valid_rate": f"{100 * total_valid / (total_valid + total_invalid):.1f}%",
        "default_payload_count": default_payload_count,
        "default_payload_rate": f"{100 * default_payload_count / total_valid:.1f}%",
        "error_breakdown": error_counts,
        "output_length_stats": {
            "min": min(output_lengths) if output_lengths else 0,
            "max": max(output_lengths) if output_lengths else 0,
            "avg": sum(output_lengths) / len(output_lengths) if output_lengths else 0,
        },
    }
    diagnostics_path = args.out_dir / "diagnostics.json"
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diagnostics_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\nDiagnostics saved to {diagnostics_path}")
    print(f"Error breakdown: {error_counts}")

    # Optional artist leakage check
    if args.check_artist_leakage and total_valid > 0:
        forbidden_tokens = find_forbidden_tokens(df_sample)
        saved_results = load_saved_results(args.out_dir)
        if not saved_results:
            print("Artist leakage check skipped (no saved results found).")
        else:
            leakage_count = count_artist_leakage(saved_results, forbidden_tokens)
            print(f"Artist leakage: {leakage_count} violations")

    if default_payload_count > 0:
        print(
            f"Default placeholder payloads: {default_payload_count} "
            f"({100 * default_payload_count / total_valid:.1f}% of valid)"
        )


if __name__ == "__main__":
    main()
