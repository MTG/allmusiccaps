"""Rate SongDescriber query complexity with Claude Haiku 4.5.

Complexity rubric: "how hard would this caption be to express with tags alone?"
Scale 1 (trivially expressible as tags) to 5 (deeply dependent on narrative,
metaphor, evaluation, or context that tags cannot capture).

Outputs scripts/lexical_slices/songd_complexity.json:
    {"index": int, "query": str, "score": int, "rationale": str}

Uses prompt caching on the system prompt, batches one query per call (cheap
enough with Haiku for 746 queries), and writes incrementally so reruns resume.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import anthropic

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_BASE = REPO_ROOT / "downstream_results"
OUT_PATH = Path(__file__).parent / "songd_complexity.json"

MODEL = "claude-haiku-4-5-20251001"
REVIEW_MODEL_DIR = "R04"

SYSTEM = """You rate the complexity of short music captions.

The scale measures how hard the caption would be to express using only a
fixed vocabulary of tags (genre, instrument, mood, tempo, language, era).

Scale:
1 -- Purely tag-expressible. Enumerates genre/instruments/mood with no
     narrative, metaphor, evaluative stance, or contextual framing. Example:
     "upbeat pop song with male vocals, drums, and guitar".
2 -- Mostly tag-expressible with one or two light descriptors that tags can
     approximate (tempo, simple mood).
3 -- Mixed. Has some narrative ("builds up", "starts with"), era/lineage
     reference, or subjective evaluation that tags partially capture.
4 -- Largely narrative, evaluative, figurative, or context-dependent.
     References progression over time, reviewer-style subjective adjectives,
     metaphors, or imagined scenes.
5 -- Deeply dependent on non-tag signal: extended metaphor, narrative arc
     through the piece, reviewer idiom ("idiosyncratic", "feels like X"),
     cultural or literary allusion, or evaluative judgement that no tag
     vocabulary could convey.

Reply in strict JSON only, no prose: {"score": <int 1-5>, "rationale": "<one short clause>"}"""


def load_queries() -> list[dict]:
    path = RESULTS_BASE / REVIEW_MODEL_DIR / "song_describer" / "caption2rank.json"
    with open(path) as f:
        data = json.load(f)
    return [{"index": r["index"], "query": r["query"]} for r in data]


def load_existing() -> dict[int, dict]:
    if not OUT_PATH.exists():
        return {}
    with open(OUT_PATH) as f:
        return {r["index"]: r for r in json.load(f)}


def save(rows: dict[int, dict]) -> None:
    out = sorted(rows.values(), key=lambda r: r["index"])
    tmp = OUT_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    tmp.replace(OUT_PATH)


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse(text: str) -> tuple[int, str]:
    m = JSON_RE.search(text)
    if not m:
        raise ValueError(f"no JSON found: {text!r}")
    obj = json.loads(m.group(0))
    score = int(obj["score"])
    if score < 1 or score > 5:
        raise ValueError(f"score out of range: {score}")
    return score, obj.get("rationale", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="rate only first N (0 = all)")
    ap.add_argument("--save-every", type=int, default=20)
    args = ap.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic()
    queries = load_queries()
    if args.limit:
        queries = queries[: args.limit]

    existing = load_existing()
    todo = [q for q in queries if q["index"] not in existing]
    print(f"total={len(queries)}  done={len(existing)}  todo={len(todo)}")

    rows = dict(existing)
    t0 = time.time()
    for i, q in enumerate(todo):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=120,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": f"Caption: {q['query']}"}],
            )
            text = resp.content[0].text
            score, rationale = parse(text)
            rows[q["index"]] = {
                "index": q["index"],
                "query": q["query"],
                "score": score,
                "rationale": rationale,
            }
        except Exception as e:
            print(f"[{i}] idx={q['index']} FAIL: {e}", file=sys.stderr)
            continue

        if (i + 1) % args.save_every == 0:
            save(rows)
            rate = (i + 1) / (time.time() - t0)
            eta = (len(todo) - i - 1) / rate if rate > 0 else 0
            print(
                f"[{i + 1}/{len(todo)}]  score={rows[q['index']]['score']}  rate={rate:.1f}/s  ETA={eta:.0f}s"
            )

    save(rows)
    print(f"done. wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
