"""Classify captions as objective/descriptive, figurative/narrative, or hard-to-decide.

Uses Llama-3.3-70B via vLLM (same setup as judge_complexity_vllm.py).

Reads caption2rank.json for both SongDescriber and MuCaps.
Writes scripts/lexical_slices/{songd,mucaps}_register.json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from vllm import LLM, SamplingParams

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_BASE = REPO_ROOT / "downstream_results"
OUT_DIR = Path(__file__).parent
DATASETS = {
    "song_describer": "songd_register.json",
    "music_caps": "mucaps_register.json",
}

LABELS = ("objective", "figurative", "hard")

SYSTEM = """You classify short music captions into exactly one of three register categories.

Categories:
- objective -- Descriptive and factual. Enumerates genre, instruments, tempo, key, language, vocal type, mood as a plain label. Sticks to what can be verified by listening. Examples: "Upbeat pop song with male vocals, drums, and guitar", "Slow jazz piano trio in a minor key".
- figurative -- Narrative, metaphorical, or evaluative. Uses imagery, similes, scenes, reviewer-style subjective judgement, or describes a progression/arc over time. Examples: "Feels like walking through a foggy forest at dawn", "An anguished track that builds tension as more instruments join", "Reminiscent of early 2010s alternative rock".
- hard -- Mixed or ambiguous. Contains both registers in comparable amounts, or lacks enough signal to decide confidently.

Reply with strict JSON only, no prose: {"label": "objective" | "figurative" | "hard", "rationale": "<one short clause>"}"""


def load_queries(dataset: str) -> list[dict]:
    path = RESULTS_BASE / "R04" / dataset / "caption2rank.json"
    with open(path) as f:
        data = json.load(f)
    return [{"index": r["index"], "query": r["query"]} for r in data]


JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse(text: str) -> tuple[str, str]:
    m = JSON_RE.search(text)
    if not m:
        return "noparse", text[:80]
    try:
        obj = json.loads(m.group(0))
        label = str(obj["label"]).strip().lower()
        if label not in LABELS:
            # tolerant match
            for L in LABELS:
                if label.startswith(L[:4]):
                    label = L
                    break
            else:
                return "noparse", f"bad label: {label}"
        return label, str(obj.get("rationale", ""))[:200]
    except Exception as e:
        return "noparse", f"err {e}: {text[:80]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--tensor-parallel", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS)
    )
    args = ap.parse_args()

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        disable_custom_all_reduce=True,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)

    for ds in args.datasets:
        out_path = OUT_DIR / DATASETS[ds]
        queries = load_queries(ds)
        if args.limit:
            queries = queries[: args.limit]
        print(f"\n=== dataset={ds}  queries={len(queries)}  out={out_path} ===")

        prompts = []
        for q in queries:
            msgs = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Caption: {q['query']}"},
            ]
            prompts.append(
                tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            )

        outs = llm.generate(prompts, sp)

        rows = []
        n_fail = 0
        label_counts = {L: 0 for L in LABELS}
        for q, o in zip(queries, outs):
            text = o.outputs[0].text
            label, rationale = parse(text)
            if label == "noparse":
                n_fail += 1
            else:
                label_counts[label] += 1
            rows.append(
                {
                    "index": q["index"],
                    "query": q["query"],
                    "label": label,
                    "rationale": rationale,
                    "raw": text.strip()[:200],
                }
            )

        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        dist = "  ".join(f"{L}={label_counts[L]}" for L in LABELS)
        print(f"  wrote {len(rows)} rows  parse_failures={n_fail}   {dist}")


if __name__ == "__main__":
    main()
