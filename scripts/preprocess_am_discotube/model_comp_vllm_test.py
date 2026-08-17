import argparse
import json
import os
from typing import List, Dict, Any

from vllm import LLM, SamplingParams


def read_prompts(path: str) -> List[str]:
    """
    Read prompts from a file.
    - .txt: each line is a prompt
    - .jsonl: each line is a JSON object with 'prompt' or 'text' field
    """
    ext = os.path.splitext(path)[1].lower()
    prompts: List[str] = []
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    val = obj.get("prompt") or obj.get("text")
                    if isinstance(val, str):
                        prompts.append(val)
                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)
    return prompts


def write_outputs_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    if chunk_size <= 0:
        return [items]
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def build_llm(args: argparse.Namespace) -> LLM:
    return LLM(
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


def build_sampling_params(args: argparse.Namespace) -> SamplingParams:
    stop = args.stop if args.stop else None
    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=args.n,
        stop=stop,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        repetition_penalty=args.repetition_penalty,
    )


def generate_with_vllm(
    llm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams,
    batch_size: int,
    return_metadata: bool,
) -> List[Dict[str, Any]]:
    """
    Generate outputs with vLLM, chunked for memory stability.
    Returns a list of dicts with prompt, completion and optional metadata.
    """
    rows: List[Dict[str, Any]] = []
    for batch_prompts in chunk_list(prompts, batch_size):
        request_outputs = llm.generate(batch_prompts, sampling_params)
        for req in request_outputs:
            # vLLM returns multiple candidates per request in 'outputs'
            candidates = req.outputs or []
            texts = [c.text for c in candidates]
            # default to first candidate for convenience
            text_primary = texts[0] if texts else ""
            row: Dict[str, Any] = {
                "prompt": req.prompt,
                "completion": text_primary,
                "completions": texts,
            }
            if return_metadata:
                row["request_id"] = req.request_id
                row["prompt_token_count"] = (
                    getattr(req, "prompt_token_ids", None)
                    and len(req.prompt_token_ids)
                    or None
                )
                # Include token-level metadata for the primary candidate if available
                if candidates:
                    cand0 = candidates[0]
                    row["token_ids"] = getattr(cand0, "token_ids", None)
                    row["logprobs"] = getattr(cand0, "logprobs", None)
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch inference using vLLM backend")
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
        "--max-model-len", type=int, default=8192, help="Max model length (tokens)"
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

    # IO
    p.add_argument(
        "--input", type=str, required=True, help="Input prompts file: .txt or .jsonl"
    )
    p.add_argument("--output", type=str, required=True, help="Output file: .jsonl")

    # Inference controls
    p.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for submission to vLLM"
    )
    p.add_argument(
        "--max-tokens", type=int, default=128, help="Max new tokens per completion"
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

    # Metadata
    p.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include request and token-level metadata in output",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    prompts = read_prompts(args.input)
    if not prompts:
        raise SystemExit("No prompts found in input file")

    llm = build_llm(args)
    sampling_params = build_sampling_params(args)

    rows = generate_with_vllm(
        llm=llm,
        prompts=prompts,
        sampling_params=sampling_params,
        batch_size=args.batch_size,
        return_metadata=args.include_metadata,
    )

    # Ensure output dir exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_outputs_jsonl(args.output, rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
