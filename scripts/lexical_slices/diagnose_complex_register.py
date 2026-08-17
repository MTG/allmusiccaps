"""Diagnose what kind of non-tag content each dataset has at v1 score >= 3.

Hypothesis: SongDescriber complex captions skew listener-side (scene, function,
narrative, evaluative); MusicCaps complex captions skew production-side
(recording qualia: lo-fi, amateur, mono, room noise). Reports trigger-term
hit rates per dataset at score >= 3 and at score >= 4.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
CSV_PATH = ROOT / "complex_register_diagnostic.csv"
DATASETS = {
    "SongDescriber": "songd_complexity.json",
    "MusicCaps": "mucaps_complexity.json",
}

# Production / recording-qualia terms.
PRODUCTION = [
    r"low quality",
    r"lo-?fi",
    r"\bnoisy\b",
    r"\bnoise\b",
    r"amateur recording",
    r"home recording",
    r"amateur",
    r"\bmono\b",
    r"\bstereo\b",
    r"\bmic\b",
    r"\bmicrophone\b",
    r"\bdistorted\b",
    r"\bclipping\b",
    r"distant\b",
    r"reverb-?heavy",
    r"\breverb\b",
    r"\bmuffled\b",
    r"\bhiss\b",
    r"\bstatic\b",
    r"\blive performance\b",
    r"\bcrowd noise\b",
    r"phone recording",
    r"recording quality",
    r"chair squeak",
    r"footsteps",
]

# Listener-side framing: scene, function, narrative arc, evaluative stance.
LISTENER = [
    r"\bbuilds\b",
    r"builds up",
    r"starts with",
    r"begins with",
    r"second half",
    r"first half",
    r"\btransitions?\b",
    r"\bdrops\b",
    r"\bcinematic\b",
    r"\bmontage\b",
    r"\bmovie\b",
    r"\bfilm\b",
    r"\bsoundtrack\b",
    r"perfect for",
    r"ideal for",
    r"\bsuited for\b",
    r"reminds me",
    r"reminds you",
    r"feels like",
    r"sounds like",
    r"makes you feel",
    r"makes one",
    r"one can",
    r"\bimagin\w*",
    r"\bdreamlike\b",
    r"\bdreamy\b",
    r"\bnostalgic\b",
    r"\bspiritual\b",
    r"\bmelanchol\w*",
    r"\bevoke\w*",
    r"\bcall to\b",
    r"\bjourney\b",
    r"\bnarrat\w*",
]


def compile_pats(terms: list[str]) -> list[re.Pattern]:
    return [re.compile(t, re.IGNORECASE) for t in terms]


PROD_PATS = compile_pats(PRODUCTION)
LIST_PATS = compile_pats(LISTENER)


def hit_terms(text: str, pats: list[re.Pattern]) -> list[str]:
    out = []
    for p in pats:
        if p.search(text):
            out.append(p.pattern)
    return out


def stats(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "prod_pct": float("nan"),
            "list_pct": float("nan"),
            "both_pct": float("nan"),
            "neither_pct": float("nan"),
            "top_prod": [],
            "top_list": [],
        }
    prod_hits = []
    list_hits = []
    prod_terms = Counter()
    list_terms = Counter()
    for r in rows:
        q = r["query"]
        ph = hit_terms(q, PROD_PATS)
        lh = hit_terms(q, LIST_PATS)
        prod_hits.append(len(ph) > 0)
        list_hits.append(len(lh) > 0)
        for t in ph:
            prod_terms[t] += 1
        for t in lh:
            list_terms[t] += 1
    prod = sum(prod_hits) / n
    listr = sum(list_hits) / n
    both = sum(p and l for p, l in zip(prod_hits, list_hits)) / n
    neither = sum((not p) and (not l) for p, l in zip(prod_hits, list_hits)) / n
    return {
        "n": n,
        "prod_pct": prod * 100,
        "list_pct": listr * 100,
        "both_pct": both * 100,
        "neither_pct": neither * 100,
        "top_prod": prod_terms.most_common(8),
        "top_list": list_terms.most_common(8),
    }


def report(label: str, s: dict) -> None:
    print(
        f"  {label:<14} n={s['n']:5d}   "
        f"production={s['prod_pct']:5.1f}%   "
        f"listener={s['list_pct']:5.1f}%   "
        f"both={s['both_pct']:5.1f}%   "
        f"neither={s['neither_pct']:5.1f}%"
    )


SLICES = [
    ("ALL", lambda r: True),
    ("score 1-2", lambda r: r["score"] in (1, 2)),
    ("score >=3", lambda r: r["score"] >= 3),
    ("score >=4", lambda r: r["score"] >= 4),
]


def main() -> None:
    print(f"Production triggers ({len(PRODUCTION)}): {PRODUCTION}\n")
    print(f"Listener triggers ({len(LISTENER)}): {LISTENER}\n")

    csv_rows = []
    for name, fname in DATASETS.items():
        rows = json.load(open(ROOT / fname))
        print(f"=== {name} ===")
        for slice_name, pred in SLICES:
            sub = [r for r in rows if pred(r)]
            s = stats(sub)
            report(slice_name, s)
            csv_rows.append(
                {
                    "dataset": name,
                    "slice": slice_name,
                    "n": s["n"],
                    "production_pct": round(s["prod_pct"], 2) if s["n"] else "",
                    "listener_pct": round(s["list_pct"], 2) if s["n"] else "",
                    "both_pct": round(s["both_pct"], 2) if s["n"] else "",
                    "neither_pct": round(s["neither_pct"], 2) if s["n"] else "",
                }
            )
        s_hi = stats([r for r in rows if r["score"] >= 3])
        print(f"    top production terms (score>=3): {s_hi['top_prod']}")
        print(f"    top listener terms   (score>=3): {s_hi['top_list']}")
        print()

    fields = [
        "dataset",
        "slice",
        "n",
        "production_pct",
        "listener_pct",
        "both_pct",
        "neither_pct",
    ]
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
