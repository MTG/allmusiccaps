#!/usr/bin/env python
"""Generate 5-level figurative rewrites of MusicCaps captions using vLLM + Outlines.

Loads the test split of ``seungheondoh/LP-MusicCaps-MC`` from HuggingFace,
extracts ``caption_ground_truth`` per ``ytid``, and asks an instruction-tuned
LLM for a structured ``FigurativeLevels`` JSON with keys level_1 .. level_5.

Output: a JSONL file with one row per MusicCaps id:

    {"id": "<ytid>", "original": "<caption>", "levels": {"1": ..., "5": ...},
     "is_valid": true, "parse_error": "success"}

Design notes
------------
- Structured decoding via Outlines means we do not need JSON-extraction fallbacks;
  the generator is constrained to emit a ``FigurativeLevels`` instance. We still
  save a JSONL trail so later analysis / curation is possible without re-running.
- Deterministic decoding (``temperature=0``) so the experiment is reproducible.
- Caching: if the output JSONL already has a row for a given id and ``--force``
  is not set, the id is skipped. Safe to re-run after a crash.
- No dependency on the discotube prompt templates or pickle data layout — this
  script is intentionally self-contained.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import outlines
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

from prompts import LEVELS, FigurativeLevels, format_prompt


# --------------------------- generator / sampling ---------------------------


def build_generator(args: argparse.Namespace):
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
        max_num_seqs=args.max_num_seqs,
    )
    model = outlines.from_vllm_offline(llm)
    return outlines.Generator(model, FigurativeLevels)


def build_sampling_params(args: argparse.Namespace) -> SamplingParams:
    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=1,
        repetition_penalty=args.repetition_penalty,
    )


# ------------------------------- data loading ------------------------------


def load_musiccaps_captions(split: str) -> List[Tuple[str, str]]:
    """Return a list of (ytid, caption_ground_truth) pairs from MusicCaps.

    MusicCaps has one caption per ytid in the HF dataset, so the list has
    the same length as the split.
    """
    ds = load_dataset("seungheondoh/LP-MusicCaps-MC")
    split_data = ds[split]
    pairs: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    for row in split_data:
        ytid = str(row["ytid"])
        caption = str(row.get("caption_ground_truth", "") or "").strip()
        if not caption:
            continue
        if ytid in seen:
            continue
        seen.add(ytid)
        pairs.append((ytid, caption))
    return pairs


def load_existing_ids(out_path: Path) -> Set[str]:
    """Load ids already written to out_path (supports re-run / resume)."""
    if not out_path.exists():
        return set()
    ids: Set[str] = set()
    with open(out_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            _id = obj.get("id")
            if _id:
                ids.add(str(_id))
    return ids


# ------------------------------ batch generation ----------------------------


def normalize_text(text: str) -> str:
    return (
        unicodedata.normalize("NFC", text)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )


def generate_batch(
    batch: List[Tuple[str, str]],
    generator,
    sampling_params: SamplingParams,
) -> List[dict]:
    """Generate FigurativeLevels for a batch of (id, caption) pairs."""
    prompts = [normalize_text(format_prompt(cap)) for _, cap in batch]

    # Length-aware ordering for vLLM batching efficiency.
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    ordered_prompts = [prompts[i] for i in order]
    ordered_batch = [batch[i] for i in order]

    outputs = generator.batch(ordered_prompts, sampling_params=sampling_params)

    results: List[dict] = []
    for (ytid, caption), output in zip(ordered_batch, outputs):
        raw = output[0] if isinstance(output, list) else output
        is_valid = True
        parse_error = "success"
        levels: Dict[str, str] = {}
        try:
            if isinstance(raw, FigurativeLevels):
                parsed = raw.model_dump()
            elif isinstance(raw, dict):
                parsed = raw
            else:
                parsed = json.loads(str(raw))
            for lvl in LEVELS:
                val = str(parsed.get(lvl, "") or "").strip()
                if not val:
                    is_valid = False
                    parse_error = f"empty:{lvl}"
                levels[lvl.split("_")[1]] = val
        except Exception as e:  # noqa: BLE001 - structured decoder failure is rare
            is_valid = False
            parse_error = f"exception:{type(e).__name__}"
            levels = {str(i): "" for i in range(1, 6)}

        results.append(
            {
                "id": ytid,
                "original": caption,
                "levels": levels,
                "is_valid": is_valid,
                "parse_error": parse_error,
            }
        )

    return results


# ----------------------------------- CLI -----------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    # Data
    p.add_argument(
        "--split",
        type=str,
        default="test",
        help="HuggingFace split of seungheondoh/LP-MusicCaps-MC to rewrite",
    )
    p.add_argument(
        "--out-path",
        type=Path,
        default=Path("musiccaps_figurative/raw_levels.jsonl"),
        help="JSONL file to append generated rows to",
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=-1,
        help="Optional cap on total items processed (debug/sanity). -1 = all.",
    )
    p.add_argument("--force", action="store_true", help="Regenerate existing ids")
    p.add_argument("--batch-size", type=int, default=16)

    # Model / engine
    p.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="HF model id / local path for the rewriter",
    )
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--enable-prefix-caching", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-num-seqs", type=int, default=128)

    # Decoding (deterministic by default)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--repetition-penalty", type=float, default=1.0)

    return p.parse_args()


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    args = parse_args()

    pairs = load_musiccaps_captions(args.split)
    print(f"Loaded {len(pairs)} (id, caption) pairs from MusicCaps/{args.split}")

    if not args.force:
        existing = load_existing_ids(args.out_path)
        if existing:
            before = len(pairs)
            pairs = [p for p in pairs if p[0] not in existing]
            print(
                f"Skipping {before - len(pairs)} already-generated ids "
                f"(use --force to regenerate)"
            )

    if args.max_items > 0:
        pairs = pairs[: args.max_items]

    if not pairs:
        print("Nothing to generate; exiting.")
        return

    print(f"Generating 5-level rewrites for {len(pairs)} captions")
    print(
        f"Model: {args.model} | TP={args.tensor_parallel} | "
        f"max_model_len={args.max_model_len}"
    )

    generator = build_generator(args)
    sampling_params = build_sampling_params(args)

    total_valid = 0
    total_invalid = 0
    errors: Dict[str, int] = {}

    for i in tqdm(range(0, len(pairs), args.batch_size), desc="Rewriting"):
        batch = pairs[i : i + args.batch_size]
        results = generate_batch(batch, generator, sampling_params)

        append_jsonl(args.out_path, results)

        for r in results:
            if r["is_valid"]:
                total_valid += 1
            else:
                total_invalid += 1
                errors[r["parse_error"]] = errors.get(r["parse_error"], 0) + 1

    print(f"\nDone. Valid: {total_valid}  Invalid: {total_invalid}")
    if errors:
        print(f"Error breakdown: {errors}")
    print(f"Output: {args.out_path}")


if __name__ == "__main__":
    main()
