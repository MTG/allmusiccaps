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
from typing import Iterable, List

import outlines
import pandas as pd
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
import transformers.tokenization_utils_base as tub
from pydantic import BaseModel
from tqdm import tqdm

from prompt_templates import format_prompt


def _no_patch(*args, **kwargs):
    return args[0]


tub.PreTrainedTokenizerBase._patch_mistral_regex = staticmethod(_no_patch)


class MusicDescription(BaseModel):
    music_style: str
    vibe: str
    tempo_energy: str
    instrumentation: str
    production_notes: str


def load_sample(data_path: Path, n_samples: int, seed: int) -> pd.DataFrame:
    with open(data_path, "rb") as f:
        df = pickle.load(f)
    return df.sample(n=n_samples, random_state=seed)


def build_generator(
    model_name: str, quantization_bits: int | None = None
) -> outlines.Generator:
    bnb_config = None
    if quantization_bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )
    elif quantization_bits == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif quantization_bits != 0:
        raise ValueError(
            f"Unsupported quantization_bits: {quantization_bits}. Choose 4, 8, or None."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
    )
    model = outlines.from_transformers(
        model,
        tokenizer,
    )

    return outlines.Generator(
        model,
        MusicDescription,
    )


def generate_descriptions(
    rows: List[dict], generator: outlines.Generator, batch_size: int
) -> list[dict]:
    results: list[dict] = []
    for i in tqdm(range(0, len(rows), batch_size), desc="Batches"):
        batch = rows[i : i + batch_size]
        prompts = [format_prompt(pd.Series(r)) for r in batch]
        print(prompts[0])
        outputs = generator.batch(prompts, max_new_tokens=512)
        print(outputs[0])

        for row, output in zip(batch, outputs):
            payload = output.model_dump() if isinstance(output, BaseModel) else output
            results.append(
                {
                    "input_metadata": row,
                    "generated": payload,
                }
            )
    return results


def save_jsonl(records: Iterable[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def find_forbidden_tokens(df_sample: pd.DataFrame) -> set[str]:
    return set(str(x).lower() for x in df_sample["artist"].dropna().unique())


def count_artist_leakage(results: list[dict], forbidden_tokens: set[str]) -> int:
    def contains_forbidden(text: str) -> bool:
        t = text.lower()
        return any(tok in t for tok in forbidden_tokens)

    violations = 0
    for r in results:
        for text in r["generated"].values():
            if isinstance(text, str) and text != "N/A" and contains_forbidden(text):
                violations += 1
    return violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", type=Path, default=Path("allmusic_youtube_discogs_reviews.pkl")
    )
    parser.add_argument(
        "--model-name", type=str, default="meta-llama/Llama-3.3-70B-Instruct"
    )
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quantization-bits",
        type=int,
        default=0,
        choices=[4, 8, 0],
        help="Quantization precision: 4-bit, 8-bit, or 0 for no quantization (default: 0)",
    )
    parser.add_argument(
        "--out-path", type=Path, default=Path("quick_test_outputs_transformers.jsonl")
    )
    parser.add_argument("--check-artist-leakage", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    generator = build_generator(args.model_name, args.quantization_bits)

    df_sample = load_sample(args.data_path, args.n_samples, args.seed)
    print(f"Loaded {len(df_sample)} examples from {args.data_path}")

    if args.quantization_bits is None:
        print("Using no quantization (full precision)")
    else:
        print(f"Using {args.quantization_bits}-bit quantization")

    rows = df_sample.to_dict(orient="records")
    results = generate_descriptions(rows, generator, args.batch_size)

    save_jsonl(results, args.out_path)
    print(f"Saved {len(results)} examples to {args.out_path}")

    if args.check_artist_leakage:
        forbidden_tokens = find_forbidden_tokens(df_sample)
        leakage_count = count_artist_leakage(results, forbidden_tokens)
        print(f"Artist leakage count: {leakage_count}")


if __name__ == "__main__":
    main()
