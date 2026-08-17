"""Score generative-music outputs with CLAP-MTG models.

For each (prompt, generated_wav, human_MOS) triple in MusicEval, compute
cosine(forward_text(prompt), forward_audio(wav)). Write per-sample scores plus
an aggregate report with bootstrap CI and Spearman/Kendall correlations against
the human MOS axes (overall musical quality and text-music alignment).

Usage mirrors src/downstream/downstream_retrieval.py, for parity with the
existing evaluate_all.sh harness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm


from .. import get_model  # noqa: E402
from clap_score.musiceval_dataset import MusicEval  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CLAP-score evaluation on MusicEval.")
    p.add_argument("--cfg_file", type=str, required=True)
    p.add_argument(
        "--data_dir", type=str, required=True, help="Path to MusicEval-full/"
    )
    p.add_argument(
        "--split", type=str, default="total", choices=["total", "train", "dev", "test"]
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--segment_size", type=float, default=10.0)
    p.add_argument("--new_freq", type=int, default=24000)
    p.add_argument("--use_audio_type_token", action="store_true")
    p.add_argument("--ckpt_step", type=int, default=None)
    p.add_argument("--avg_last_n", type=int, default=None)
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) / "musiceval"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = get_model(
        config_file=args.cfg_file,
        device=args.device,
        weights_only=False,
        ckpt_step=args.ckpt_step,
        avg_last_n=args.avg_last_n,
    )
    model.eval()

    dataset = MusicEval(
        data_dir=args.data_dir,
        split=args.split,
        sr=args.new_freq,
        duration=args.segment_size,
    )
    print(f"Loaded {len(dataset)} (prompt, audio, MOS) triples from {args.data_dir}")

    prompts = dataset.df["prompt"].tolist()
    unique_prompts = sorted(set(prompts))
    prompt_to_emb: dict[str, torch.Tensor] = {}
    for q in tqdm(unique_prompts, desc="text"):
        q_in = f"[MUSIC] {q}" if args.use_audio_type_token else q
        with torch.no_grad():
            emb = model.forward_text([q_in]).detach().cpu()
        prompt_to_emb[q] = F.normalize(emb, dim=-1)

    rows = []
    for i in tqdm(range(len(dataset)), desc="audio"):
        fname, prompt, audio = dataset[i]
        audio = audio.to(args.device)
        with torch.inference_mode():
            a_emb = model.forward_audio(audio)
        a_emb = a_emb.mean(0, keepdim=True).detach().cpu()
        a_emb = F.normalize(a_emb, dim=-1)
        t_emb = prompt_to_emb[prompt]
        score = float((a_emb @ t_emb.T).squeeze().item())
        meta = dataset.row(i)
        rows.append(
            {
                "fname": fname,
                "system": meta["system"],
                "prompt_id": meta["prompt_id"],
                "prompt": prompt,
                "clap_score": score,
                "mos_overall": meta["mos_overall"],
                "mos_alignment": meta["mos_alignment"],
            }
        )

    with open(out_dir / "per_sample.json", "w") as f:
        json.dump(rows, f, indent=2)

    scores = np.array([r["clap_score"] for r in rows])
    mos_o = np.array([r["mos_overall"] for r in rows])
    mos_a = np.array([r["mos_alignment"] for r in rows])
    mean, lo, hi = bootstrap_mean_ci(scores, args.n_bootstrap, args.seed)

    systems = sorted({r["system"] for r in rows})
    per_system = {}
    for s in systems:
        sel = [r for r in rows if r["system"] == s]
        s_scores = np.array([r["clap_score"] for r in sel])
        s_o = np.array([r["mos_overall"] for r in sel])
        s_a = np.array([r["mos_alignment"] for r in sel])
        per_system[s] = {
            "n": len(sel),
            "mean_clap": float(s_scores.mean()),
            "mean_mos_overall": float(s_o.mean()),
            "mean_mos_alignment": float(s_a.mean()),
        }

    sys_ids = list(per_system)
    sys_clap = np.array([per_system[s]["mean_clap"] for s in sys_ids])
    sys_mos_o = np.array([per_system[s]["mean_mos_overall"] for s in sys_ids])
    sys_mos_a = np.array([per_system[s]["mean_mos_alignment"] for s in sys_ids])

    results = {
        "n_samples": int(len(scores)),
        "mean_clap": mean,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "utterance_level": {
            "spearman_overall": float(stats.spearmanr(scores, mos_o).statistic),
            "spearman_alignment": float(stats.spearmanr(scores, mos_a).statistic),
            "kendall_overall": float(stats.kendalltau(scores, mos_o).statistic),
            "kendall_alignment": float(stats.kendalltau(scores, mos_a).statistic),
            "pearson_overall": float(stats.pearsonr(scores, mos_o).statistic),
            "pearson_alignment": float(stats.pearsonr(scores, mos_a).statistic),
        },
        "system_level": {
            "n_systems": len(sys_ids),
            "spearman_overall": float(stats.spearmanr(sys_clap, sys_mos_o).statistic),
            "spearman_alignment": float(stats.spearmanr(sys_clap, sys_mos_a).statistic),
            "pearson_overall": float(stats.pearsonr(sys_clap, sys_mos_o).statistic),
            "pearson_alignment": float(stats.pearsonr(sys_clap, sys_mos_a).statistic),
        },
        "per_system": per_system,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"n={len(scores)}  mean CLAP={mean:.4f}  CI95=[{lo:.4f},{hi:.4f}]  "
        f"spearman(align)={results['utterance_level']['spearman_alignment']:+.3f}  "
        f"system-level spearman(align)={results['system_level']['spearman_alignment']:+.3f}"
    )


if __name__ == "__main__":
    main()
