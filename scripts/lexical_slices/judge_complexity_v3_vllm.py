"""Rate caption complexity on two orthogonal binary axes with Llama-3.3-70B (v3).

v1 (5-level absolute) and v2 (4-level absolute) both collapsed under the
caption-length prior: SongDescriber clustered at the low end, MusicCaps at the
middle, and the extreme levels were empty on at least one dataset.

v3 replaces the single "complexity" axis with TWO independent binary axes so
the 4 buckets (2x2) each receive mass on both datasets by construction:

- density  ("thin" vs "dense"):   how many concrete sonic attributes
                                  (instruments, rhythm, vocal, tempo, feel)
                                  are specified.
- framing  ("none" vs "present"): whether the caption contains content a
                                  tag vocabulary cannot capture (temporal
                                  structure, production qualia, scene/use
                                  framing, evaluative stance, metaphor).

The final bucket is the pair (density, framing). A legacy integer "score" in
1..4 is emitted for drop-in compatibility with the existing analyze/plot
scripts:
    1 = thin  + none
    2 = thin  + present
    3 = dense + none
    4 = dense + present

Reads queries from downstream_results/R04/<dataset>/caption2rank.json.
Writes scripts/lexical_slices/<dataset>_complexity_v3.json.
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
    "song_describer": "songd_complexity_v3.json",
    "music_caps": "mucaps_complexity_v3.json",
}

SYSTEM = """You classify music captions along TWO independent binary axes. Judge each axis on its own; do not let one bias the other.

Axis 1 -- density: how many concrete sonic attributes the caption specifies.
  thin   = one or two broad attributes only (e.g. genre + mood, or genre + one instrument). Short or sparse.
  dense  = three or more concrete attributes stacked (instruments, rhythm/groove, vocal, tempo, dynamics, feel). Rich enumeration.

Axis 2 -- framing: whether the caption contains content that a fixed tag vocabulary (genre, instrument, mood, tempo, language, era, vocal gender) cannot capture.
  none    = pure attribute enumeration. No structure, no production qualia, no scene, no metaphor, no evaluative adjectives beyond a single mood word.
  present = at least ONE of:
            - temporal/structural cue ("starts with", "builds", "second half", "drops")
            - production or recording qualia ("lo-fi", "reverb-heavy", "amateur recording", "distorted", "mono")
            - scene, function, or use framing ("for a film noir scene", "movie montage", "tutorial")
            - evaluative stance beyond a single mood word ("sophisticated", "sleepy vocal", "passionate")
            - metaphor, narrative arc, cultural or literary allusion
            - specific playing style or performance technique ("fingerstyle", "walking bassline", "shuffled backbeat")

Judge the two axes independently. A short caption can still be framed; a long caption can still be pure enumeration.

Reply with strict JSON only, no prose: {"density": "thin"|"dense", "framing": "none"|"present", "rationale": "<one short clause>"}"""


def load_queries(dataset: str) -> list[dict]:
    path = RESULTS_BASE / "R04" / dataset / "caption2rank.json"
    with open(path) as f:
        data = json.load(f)
    return [{"index": r["index"], "query": r["query"]} for r in data]


JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)

_DENSITY = {"thin", "dense"}
_FRAMING = {"none", "present"}


def _axes_to_score(density: str, framing: str) -> int:
    if density == "thin" and framing == "none":
        return 1
    if density == "thin" and framing == "present":
        return 2
    if density == "dense" and framing == "none":
        return 3
    if density == "dense" and framing == "present":
        return 4
    return -1


def parse(text: str) -> tuple[int, str, str, str]:
    m = JSON_RE.search(text)
    if not m:
        return -1, "", "", f"noparse: {text[:80]}"
    try:
        obj = json.loads(m.group(0))
        density = str(obj.get("density", "")).strip().lower()
        framing = str(obj.get("framing", "")).strip().lower()
        if density not in _DENSITY or framing not in _FRAMING:
            return -1, density, framing, f"bad axes: d={density} f={framing}"
        rationale = str(obj.get("rationale", ""))[:200]
        return _axes_to_score(density, framing), density, framing, rationale
    except Exception as e:
        return -1, "", "", f"err {e}: {text[:80]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--tensor-parallel", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-tokens", type=int, default=100)
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
        score, density, framing, rationale = parse(text)
        if score < 0:
            n_fail += 1
        rows.append(
            {
                "index": q["index"],
                "query": q["query"],
                "score": score,
                "density": density,
                "framing": framing,
                "rationale": rationale,
                "raw": text.strip()[:200],
            }
        )

    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"wrote {len(rows)} rows -> {out_path}  (parse_failures={n_fail})")


if __name__ == "__main__":
    main()
