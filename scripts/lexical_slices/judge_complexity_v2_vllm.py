"""Rate caption complexity with Llama-3.3-70B via vLLM (v2: 4-level scale).

The v2 rubric replaces the v1 5-point narrative-heavy scale with a 4-level
descriptive-depth scale. The v1 scale was strongly bimodal across datasets:
SongDescriber (short, tag-like captions) clumped at 1-2; MusicCaps (long
reviewer-style captions) clumped at 3-4; levels 1 and 5 were almost empty on
MusicCaps. The v2 rubric anchors on descriptive depth and the quantity of
non-tag signal (progression, production/recording qualia, structural
references, subjective framing). Both datasets are expected to cover all four
levels with usable sample counts.

Reads queries from downstream_results/R04/<dataset>/caption2rank.json.
Writes scripts/lexical_slices/<dataset>_complexity_v2.json.
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
    "song_describer": "songd_complexity_v2.json",
    "music_caps": "mucaps_complexity_v2.json",
}

SYSTEM = """You rate the complexity of music captions on a 4-level scale.

The scale measures how much of the caption's content lives OUTSIDE a fixed tag vocabulary (genre, instrument, mood, tempo, language, era, vocal gender). Longer captions are not automatically more complex; score by the fraction and depth of non-tag content, not by word count alone.

Non-tag signal includes any of:
- temporal/structural description ("starts with", "builds", "drops", "second half")
- production or recording qualia ("lo-fi", "reverb-heavy", "amateur recording", "mono", "noisy")
- subjective/evaluative framing ("sophisticated", "passionate", "sleepy vocal")
- scene, function, or usage framing ("perfect for a film noir scene", "for a movie montage")
- metaphor, narrative arc, cultural or literary allusion
- specific playing style, technique, or performance detail beyond instrument naming ("fingerstyle", "walking bassline", "shuffled backbeat")

Scale:
1 -- Tag-like. Enumerates a small set of genre/instrument/mood/tempo attributes. No non-tag signal, or at most one incidental adjective. Typically short and flat.
2 -- Descriptive. Multiple concrete attributes plus one or two light non-tag elements (a playing style, a simple evaluative adjective, or a simple tempo/feel cue). Still mostly tag-reducible.
3 -- Rich. Several non-tag elements present: production/recording details, structural cues, specific performance techniques, or clearly subjective framing. A listener using tags alone would miss meaningful content.
4 -- Highly non-tag. The caption's identity depends on non-tag content: extended structural narrative across the piece, strong evaluative or scene/function framing, metaphor, or dense production and performance detail that tags cannot approximate.

Reply with strict JSON only, no prose: {"score": <int 1-4>, "rationale": "<one short clause>"}"""


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
        if not 1 <= score <= 4:
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
