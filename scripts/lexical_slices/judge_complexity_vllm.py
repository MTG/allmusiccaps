"""Rate SongDescriber query complexity with Llama-3.3-70B via vLLM.

Rubric: "how hard would this caption be to express with tags alone?"
Scale 1 (tag-expressible) -> 5 (deeply narrative / metaphorical / evaluative).

Reads queries from downstream_results/R04/song_describer/caption2rank.json.
Writes scripts/lexical_slices/songd_complexity.json.
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
    "song_describer": "songd_complexity.json",
    "music_caps": "mucaps_complexity.json",
}

SYSTEM = """You rate the complexity of short music captions.

The scale measures how hard the caption would be to express using only a fixed vocabulary of tags (genre, instrument, mood, tempo, language, era).

Scale:
1 -- Purely tag-expressible. Enumerates genre/instruments/mood with no narrative, metaphor, evaluative stance, or contextual framing.
2 -- Mostly tag-expressible with one or two light descriptors (tempo, simple mood) that tags can approximate.
3 -- Mixed. Has some narrative ("builds up", "starts with"), era/lineage reference, or subjective evaluation partially captured by tags.
4 -- Largely narrative, evaluative, figurative, or context-dependent. References progression over time, reviewer-style subjective adjectives, metaphors, or imagined scenes.
5 -- Deeply dependent on non-tag signal: extended metaphor, narrative arc through the piece, reviewer idiom, cultural or literary allusion, or evaluative judgement no tag vocabulary could convey.

Reply with strict JSON only, no prose: {"score": <int 1-5>, "rationale": "<one short clause>"}"""


def load_queries(dataset: str) -> list[dict]:
    path = RESULTS_BASE / "R04" / dataset / "caption2rank.json"
    with open(path) as f:
        data = json.load(f)
    return [{"index": r["index"], "query": r["query"]} for r in data]


JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse(text: str) -> tuple[int, str]:
    m = JSON_RE.search(text)
    if not m:
        return -1, f"noparse: {text[:80]}"
    try:
        obj = json.loads(m.group(0))
        score = int(obj["score"])
        if not 1 <= score <= 5:
            return -1, f"oor: {score}"
        return score, str(obj.get("rationale", ""))[:200]
    except Exception as e:
        return -1, f"err {e}: {text[:80]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--tensor-parallel", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dataset", choices=list(DATASETS), default="song_describer")
    args = ap.parse_args()

    out_path = OUT_DIR / DATASETS[args.dataset]
    queries = load_queries(args.dataset)
    if args.limit:
        queries = queries[: args.limit]
    print(f"dataset={args.dataset}  queries={len(queries)}  out={out_path}")

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

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)
    outs = llm.generate(prompts, sp)

    rows = []
    n_fail = 0
    for q, o in zip(queries, outs):
        text = o.outputs[0].text
        score, rationale = parse(text)
        if score < 0:
            n_fail += 1
        rows.append(
            {
                "index": q["index"],
                "query": q["query"],
                "score": score,
                "rationale": rationale,
                "raw": text.strip()[:200],
            }
        )

    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"wrote {len(rows)} rows -> {out_path}  (parse_failures={n_fail})")


if __name__ == "__main__":
    main()
