"""Learned router: predict per-caption which model (quotes vs struct) will win.

Pipeline:
    1. For each query in MusicCaps and SongDescriber, label = sign(rank_struct - rank_quotes).
       (>0 → quotes wins; <0 → struct wins; 0 → tie, dropped)
    2. Train a logistic-regression classifier on TF-IDF features of the caption.
    3. Use 5-fold CV to get out-of-fold predictions, route per query, recompute MRR.
    4. Compare to: quotes_only, struct_only, oracle, random_router.

This is the cheapest possible router; if it captures even a meaningful fraction of
the oracle-vs-baseline gap, the routing signal is in the caption text itself —
which is the publishable claim.

Usage:
    python learned_router.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

PAIRS = [
    ("aux_data", "R04", "R05"),
    ("no_aux", "R02", "R03"),
]
DATASETS = ["music_caps", "song_describer"]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_BASE = REPO_ROOT / "downstream_results"

N_FOLDS = 5
RNG = 0


def load_caption2rank(model_id: str, dataset: str) -> dict[int, dict]:
    path = RESULTS_BASE / model_id / dataset / "caption2rank.json"
    items = json.loads(path.read_text())
    return {it["index"]: it for it in items}


def metrics_from_ranks(ranks: list[int]) -> dict:
    n = len(ranks)
    if n == 0:
        return {"n": 0, "mrr": 0.0, "r@1": 0.0, "r@5": 0.0, "r@10": 0.0}
    return {
        "n": n,
        "mrr": round(sum(1.0 / (r + 1) for r in ranks) / n, 4),
        "r@1": round(sum(1 for r in ranks if r < 1) / n, 4),
        "r@5": round(sum(1 for r in ranks if r < 5) / n, 4),
        "r@10": round(sum(1 for r in ranks if r < 10) / n, 4),
    }


def evaluate_pair(pair_name: str, q_id: str, s_id: str, dataset: str) -> dict:
    q = load_caption2rank(q_id, dataset)
    s = load_caption2rank(s_id, dataset)
    common = sorted(set(q) & set(s))

    captions = [q[i]["query"] for i in common]
    ranks_q = np.array([q[i]["min_rank"] for i in common])
    ranks_s = np.array([s[i]["min_rank"] for i in common])
    delta = ranks_s - ranks_q  # >0 → quotes wins
    labels = np.sign(delta)  # +1 quotes, -1 struct, 0 tie

    # Drop ties for training; keep them for evaluation (router decides between
    # two models that agree → routing makes no difference)
    train_mask = labels != 0
    y = (labels[train_mask] > 0).astype(int)  # 1 = quotes wins
    X_text = [captions[i] for i in np.where(train_mask)[0]]

    # Out-of-fold predictions on the *full* set
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        lowercase=True,
        token_pattern=r"[A-Za-z][A-Za-z\-']+",
    )
    # We need OOF predictions over ALL captions (train_mask + ties),
    # so vectorize on all captions but only fit/train on non-tie folds.
    Xall = vec.fit_transform(captions)
    Xtrain_full = Xall[train_mask]

    oof_pred = np.zeros(
        len(captions), dtype=int
    )  # default → quotes (arbitrary; ties unaffected)
    oof_proba_quotes = np.full(len(captions), 0.5)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG)
    train_indices = np.where(train_mask)[0]

    for fold_idx, (tr, te) in enumerate(kf.split(train_indices)):
        tr_idx_global = train_indices[tr]  # noqa: F841 (clarity)
        te_idx_global = train_indices[te]
        clf = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1)
        clf.fit(Xtrain_full[tr], y[tr])
        proba = clf.predict_proba(Xall[te_idx_global])[:, 1]  # P(quotes wins)
        oof_proba_quotes[te_idx_global] = proba
        oof_pred[te_idx_global] = (proba >= 0.5).astype(int)

    # For tied queries, prediction does not matter (both ranks equal); use 0.5
    # For evaluating coverage on non-train queries: we have no OOF for ties,
    # so we just predict the majority class (doesn't change ranks for them).
    tie_mask = ~train_mask
    if tie_mask.any():
        # predict majority on ties so they at least have a prediction
        clf_full = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1)
        clf_full.fit(Xtrain_full, y)
        oof_pred[tie_mask] = (
            clf_full.predict_proba(Xall[tie_mask])[:, 1] >= 0.5
        ).astype(int)
        oof_proba_quotes[tie_mask] = clf_full.predict_proba(Xall[tie_mask])[:, 1]

    # Apply router: if oof_pred==1, take quotes rank; else take struct rank
    routed_ranks = np.where(oof_pred == 1, ranks_q, ranks_s).tolist()

    # Baselines & oracles
    quotes_only = metrics_from_ranks(ranks_q.tolist())
    struct_only = metrics_from_ranks(ranks_s.tolist())
    random_router = metrics_from_ranks(ranks_q.tolist() + ranks_s.tolist())  # mean MRR
    oracle_router = metrics_from_ranks(np.minimum(ranks_q, ranks_s).tolist())
    learned_router = metrics_from_ranks(routed_ranks)

    # Routing-decision quality (only on non-ties)
    pred_train = oof_pred[train_mask]
    accuracy = float((pred_train == y).mean())
    quotes_pred_rate = float(pred_train.mean())
    quotes_label_rate = float(y.mean())

    # Realised gap closure: (learned - random) / (oracle - random)
    def gap_closure(metric: str) -> float:
        denom = oracle_router[metric] - random_router[metric]
        if denom <= 0:
            return 0.0
        return round((learned_router[metric] - random_router[metric]) / denom, 3)

    return {
        "pair": pair_name,
        "dataset": dataset,
        "n_queries": len(common),
        "n_train": int(train_mask.sum()),
        "n_ties": int(tie_mask.sum()),
        "router_classifier": {
            "oof_accuracy": round(accuracy, 4),
            "quotes_label_rate": round(quotes_label_rate, 4),
            "quotes_predicted_rate": round(quotes_pred_rate, 4),
        },
        "metrics": {
            "quotes_only": quotes_only,
            "struct_only": struct_only,
            "random_router": random_router,
            "learned_router": learned_router,
            "oracle_router": oracle_router,
        },
        "gap_closure_vs_oracle": {
            "mrr": gap_closure("mrr"),
            "r@1": gap_closure("r@1"),
            "r@5": gap_closure("r@5"),
            "r@10": gap_closure("r@10"),
        },
    }


def main():
    out = []
    for pair_name, q_id, s_id in PAIRS:
        for dataset in DATASETS:
            r = evaluate_pair(pair_name, q_id, s_id, dataset)
            out.append(r)

    # Console summary
    for r in out:
        m = r["metrics"]
        print("=" * 110)
        print(
            f"PAIR: {r['pair']}    DATASET: {r['dataset']}    n={r['n_queries']}    "
            f"(train={r['n_train']}, ties={r['n_ties']})"
        )
        print(
            f"  classifier OOF acc = {r['router_classifier']['oof_accuracy']:.4f}   "
            f"label rate (quotes) = {r['router_classifier']['quotes_label_rate']:.3f}   "
            f"predicted rate = {r['router_classifier']['quotes_predicted_rate']:.3f}"
        )
        print("-" * 110)
        print(f"  {'system':<22} {'MRR':>8} {'R@1':>8} {'R@5':>8} {'R@10':>8}")
        for label in [
            "quotes_only",
            "struct_only",
            "random_router",
            "learned_router",
            "oracle_router",
        ]:
            mm = m[label]
            print(
                f"  {label:<22} {mm['mrr']:>8} {mm['r@1']:>8} {mm['r@5']:>8} {mm['r@10']:>8}"
            )
        gc = r["gap_closure_vs_oracle"]
        print(
            f"  gap closure (learned − random)/(oracle − random):  "
            f"MRR={gc['mrr'] * 100:>5.1f}%  R@1={gc['r@1'] * 100:>5.1f}%  "
            f"R@5={gc['r@5'] * 100:>5.1f}%  R@10={gc['r@10'] * 100:>5.1f}%"
        )
        print()

    out_path = (
        REPO_ROOT
        / "scripts"
        / "text_corpus_vocabulary_coverage"
        / "learned_router.json"
    )
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
