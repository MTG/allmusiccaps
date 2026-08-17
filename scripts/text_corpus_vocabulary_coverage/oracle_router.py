"""Oracle-router upper bound for quotes vs struct CLAP models.

For each query, take min(rank_quotes, rank_struct) — the rank an oracle that
picks the right model per query would achieve. Compare aggregate MRR / R@k
to each individual model. Large gaps mean the corpora teach complementary
retrieval skills.

Inputs (already on disk):
    downstream_results/<model_id>/{music_caps,song_describer}/caption2rank.json

Pairings:
    aux_data:  R04 (quotes+mu+so) vs R05 (struct+mu+so)
    no_aux:    R02 (quotes only)  vs R03 (struct only)

Also computes:
    - 4-way oracle (best of all four Table 1 models) as a sanity ceiling
    - 50/50 random router baseline (expected = average of the two MRRs)
    - quotes-only / struct-only baselines

Usage:
    python oracle_router.py
"""

from __future__ import annotations

import json
from pathlib import Path

PAIRS = [
    ("aux_data", "R04", "R05"),  # quotes+mu+so vs struct+mu+so
    ("no_aux", "R02", "R03"),  # quotes only   vs struct only
]
ALL_MODELS = ["R04", "R05", "R03", "R02"]
DATASETS = ["music_caps", "song_describer"]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_BASE = REPO_ROOT / "downstream_results"


def load_ranks(model_id: str, dataset: str) -> dict[int, int]:
    path = RESULTS_BASE / model_id / dataset / "caption2rank.json"
    items = json.loads(path.read_text())
    return {it["index"]: it["min_rank"] for it in items}


def metrics_from_ranks(ranks: list[int]) -> dict:
    """Compute MRR + R@k from a list of 0-indexed ranks (one per query)."""
    n = len(ranks)
    mrr = sum(1.0 / (r + 1) for r in ranks) / n
    return {
        "n": n,
        "mrr": round(mrr, 4),
        "r@1": round(sum(1 for r in ranks if r < 1) / n, 4),
        "r@5": round(sum(1 for r in ranks if r < 5) / n, 4),
        "r@10": round(sum(1 for r in ranks if r < 10) / n, 4),
        "r@50": round(sum(1 for r in ranks if r < 50) / n, 4),
        "median_rank": sorted(ranks)[n // 2],
    }


def main():
    out = {}
    for dataset in DATASETS:
        per_model = {m: load_ranks(m, dataset) for m in ALL_MODELS}
        common = sorted(set.intersection(*[set(d) for d in per_model.values()]))

        # Per-model baselines (restricted to the common index set so all
        # numbers are comparable)
        baselines = {
            m: metrics_from_ranks([per_model[m][i] for i in common]) for m in ALL_MODELS
        }

        # Per-pair oracles
        pair_oracles = {}
        for pair_name, q_id, s_id in PAIRS:
            ranks_q = [per_model[q_id][i] for i in common]
            ranks_s = [per_model[s_id][i] for i in common]
            ranks_oracle = [min(rq, rs) for rq, rs in zip(ranks_q, ranks_s)]
            ranks_random = ranks_q + ranks_s  # union, equivalent to mean MRR
            pair_oracles[pair_name] = {
                "quotes_model": q_id,
                "struct_model": s_id,
                "quotes_only": metrics_from_ranks(ranks_q),
                "struct_only": metrics_from_ranks(ranks_s),
                "random_router": metrics_from_ranks(ranks_random),
                "oracle_router": metrics_from_ranks(ranks_oracle),
                "quotes_uniquely_in_top10": sum(
                    1 for rq, rs in zip(ranks_q, ranks_s) if rq < 10 and rs >= 10
                ),
                "struct_uniquely_in_top10": sum(
                    1 for rq, rs in zip(ranks_q, ranks_s) if rs < 10 and rq >= 10
                ),
                "both_in_top10": sum(
                    1 for rq, rs in zip(ranks_q, ranks_s) if rq < 10 and rs < 10
                ),
                "neither_in_top10": sum(
                    1 for rq, rs in zip(ranks_q, ranks_s) if rq >= 10 and rs >= 10
                ),
            }

        # 4-way oracle ceiling
        ranks_4way = [min(per_model[m][i] for m in ALL_MODELS) for i in common]
        four_way_oracle = metrics_from_ranks(ranks_4way)

        out[dataset] = {
            "n_common_queries": len(common),
            "baselines": baselines,
            "pair_oracles": pair_oracles,
            "four_way_oracle": four_way_oracle,
        }

    # Console summary
    for dataset, d in out.items():
        print("=" * 100)
        print(
            f"DATASET: {dataset}    (n={d['n_common_queries']} queries common to all 4 models)"
        )
        print("=" * 100)
        print(
            f"{'System':<40} {'MRR':>8} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'R@50':>8} {'medR':>6}"
        )
        print("-" * 100)
        # Each pair
        for pair_name, po in d["pair_oracles"].items():
            print(
                f"\n  -- pair: {pair_name}  ({po['quotes_model']} vs {po['struct_model']}) --"
            )
            for label in [
                "quotes_only",
                "struct_only",
                "random_router",
                "oracle_router",
            ]:
                m = po[label]
                print(
                    f"  {label:<40} {m['mrr']:>8} {m['r@1']:>8} {m['r@5']:>8} {m['r@10']:>8} {m['r@50']:>8} {m['median_rank']:>6}"
                )
            n = po["quotes_only"]["n"]
            print(
                f"  top-10 partition over {n} queries:  "
                f"both={po['both_in_top10']}  quotes-only={po['quotes_uniquely_in_top10']}  "
                f"struct-only={po['struct_uniquely_in_top10']}  neither={po['neither_in_top10']}"
            )

        # 4-way ceiling
        m = d["four_way_oracle"]
        print(
            f"\n  {'4-way oracle (all 4 models)':<40} {m['mrr']:>8} {m['r@1']:>8} {m['r@5']:>8} {m['r@10']:>8} {m['r@50']:>8} {m['median_rank']:>6}"
        )
        print()

    # Save full report
    out_path = (
        REPO_ROOT / "scripts" / "text_corpus_vocabulary_coverage" / "oracle_router.json"
    )
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
