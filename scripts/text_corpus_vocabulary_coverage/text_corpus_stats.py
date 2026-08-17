"""Text-corpus vocabulary coverage analysis (Experiment E3).

For each training text corpus (quotes, struct, M4-RAG/r4, MSD, Freesound, PSE) compute:
  - total token count, unique token count (types), type-token ratio (TTR)
  - vocabulary overlap with MusicCaps and SongDescriber query vocabularies

Each corpus is restricted to the IDs present in its training filelist so that
statistics reflect only the data the model actually sees during training.

Outputs a JSON report and prints a summary table to stdout.

Usage (on the cluster, clap env):
    source /projects/<group>/envs/clap/bin/activate
    python text_corpus_stats.py --out-path text_corpus_stats.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_SPLIT_RE = re.compile(r"[^a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on non-alphanumeric."""
    return [tok for tok in _SPLIT_RE.split(text.lower()) if tok]


def vocab_from_texts(texts: Iterable[str]) -> tuple[Counter, int]:
    """Return (token_counter, total_token_count) from an iterable of strings."""
    counter: Counter = Counter()
    total = 0
    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)
        total += len(tokens)
    return counter, total


# ---------------------------------------------------------------------------
# Filelist ID extraction — mirrors the logic in each dataset class
# ---------------------------------------------------------------------------


def _load_ids_stem(filelist_path: str) -> set[str]:
    """Extract IDs as ``Path(line).stem`` (quotes, struct, r4, msd)."""
    ids: set[str] = set()
    with open(filelist_path) as f:
        for line in f:
            ids.add(Path(line.strip()).stem)
    return ids


def _load_ids_freesound(filelist_path: str) -> set[str]:
    """Extract Freesound IDs: ``Path(line).stem.split('_')[0]``."""
    ids: set[str] = set()
    with open(filelist_path) as f:
        for line in f:
            ids.add(Path(line.strip()).stem.split("_")[0])
    return ids


# ---------------------------------------------------------------------------
# Corpus loaders — each yields all text strings for a corpus
# ---------------------------------------------------------------------------


def iter_quotes_texts(
    text_file: str, allowed_ids: set[str] | None = None
) -> Iterable[str]:
    """Yield every sentence from the quotes (discotube clean) JSONL."""
    with open(text_file) as f:
        for line in f:
            entry = json.loads(line)
            for yt_id, releases in entry.items():
                if allowed_ids is not None and yt_id not in allowed_ids:
                    continue
                for _release, sentences in releases.items():
                    for sentence in sentences:
                        if isinstance(sentence, dict):
                            sentence = sentence.get("text", "")
                        if not isinstance(sentence, str):
                            continue
                        s = sentence.strip()
                        if s:
                            yield s


def iter_struct_texts(
    text_file: str, allowed_ids: set[str] | None = None
) -> Iterable[str]:
    """Yield every field value from the structured (discotube v3) JSONL."""
    skip = {"none", "n/a", "na"}
    with open(text_file) as f:
        for line in f:
            entry = json.loads(line)
            for yt_id, fields in entry.items():
                if allowed_ids is not None and yt_id not in allowed_ids:
                    continue
                for _field, value in fields.items():
                    if not isinstance(value, str):
                        continue
                    v = value.strip()
                    if v and v.lower() not in skip:
                        yield v


def iter_m4rag_texts(
    metadata_file: str, allowed_ids: set[str] | None = None
) -> Iterable[str]:
    """Yield text field values from the M4-RAG (r4) JSONL.

    Only yields the fields the model sees during training:
    ``description``, ``background``, ``analysis``, ``scene``.
    Genres and tags are excluded (not used by the model).
    """
    text_fields = ["description", "background", "analysis", "scene"]
    with open(metadata_file) as f:
        for line in f:
            entry = json.loads(line)
            entry_id = entry.get("id", "")
            if allowed_ids is not None and entry_id not in allowed_ids:
                continue
            for field in text_fields:
                value = entry.get(field)
                if isinstance(value, list):
                    value = ", ".join(value)
                if isinstance(value, str):
                    v = value.strip()
                    if v:
                        yield v


def iter_msd_texts(allowed_ids: set[str] | None = None) -> Iterable[str]:
    """Yield pseudo-captions and tag strings from the MSD HF dataset."""
    from datasets import load_dataset

    ds = load_dataset("seungheondoh/enrich-msd", split="train")
    for row in ds:
        track_id = row.get("track_id", "")
        if allowed_ids is not None and track_id not in allowed_ids:
            continue
        caption = (row.get("pseudo_caption") or "").strip()
        if caption:
            yield caption
        tags = row.get("tag_list")
        if tags:
            yield ", ".join(tags)


def iter_freesound_texts(
    text_file: str, allowed_ids: set[str] | None = None
) -> Iterable[str]:
    """Yield descriptions and tag strings from the Freesound JSONL."""
    with open(text_file) as f:
        for line in f:
            entry = json.loads(line)
            for fsid, meta in entry.items():
                if allowed_ids is not None and fsid not in allowed_ids:
                    continue
                desc = (meta.get("description") or "").strip()
                if desc:
                    yield desc
                tags = meta.get("tags")
                if tags:
                    yield ", ".join(tags)


def iter_pse_texts(filelist: str) -> Iterable[str]:
    """Yield taxonomy + description strings derived from PSE file paths.

    PSE is already restricted to the training filelist (text comes from paths).
    """
    digit_re = re.compile(r"\b\d+\b")
    with open(filelist) as f:
        for line in f:
            path = Path(line.strip())
            # taxonomy from parent dirs
            cat1 = path.parent.parent.name
            cat2 = path.parent.name
            tags = f"{cat1}, {cat2}"
            # description from filename
            parts = path.stem.split("_")
            if len(parts) >= 2:
                description = parts[1].strip()
                description = digit_re.sub("", description).strip()
            else:
                description = ""
            text = f"{tags}, {description}" if description else tags
            yield text


# ---------------------------------------------------------------------------
# Query vocabulary loaders
# ---------------------------------------------------------------------------


def load_musiccaps_vocab() -> set[str]:
    """Return the set of unique tokens in MusicCaps test captions."""
    from datasets import load_dataset

    ds = load_dataset("seungheondoh/LP-MusicCaps-MC")
    tokens: set[str] = set()
    for row in ds["test"]:
        caption = row.get("caption_ground_truth", "")
        tokens.update(tokenize(caption))
    return tokens


def load_songdescriber_vocab() -> set[str]:
    """Return the set of unique tokens in SongDescriber (original, valid) captions."""
    from datasets import load_dataset

    ds = load_dataset("seungheondoh/eval-song_describer")
    tokens: set[str] = set()
    for row in ds["original"]:
        if not row.get("is_valid_subset", False):
            continue
        caption = row.get("caption", "")
        tokens.update(tokenize(caption))
    return tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Default cluster paths
DEFAULTS = {
    "quotes_text_file": "/scratch/<group>/discotube/metadata/Qwen_Qwen2.5-32B__chatgpt_v2__t0.5__1.1.jsonl",
    "struct_text_file": "/scratch/<group>/discotube/metadata/llama33_promptv3_captions.jsonl",
    "m4rag_metadata_file": "/scratch/<group>/discotube/metadata/m4rag_splits/m4rag_metadata.jsonl",
    "freesound_text_file": "/scratch/<group>/mmaps_freesound/freesound_metadata.jsonl",
    "pse_filelist": "/scratch/<group>/mmaps_pse/filelist_train.txt",
    # Training filelists — used to restrict text to IDs the model sees
    "quotes_filelist": "/scratch/<group>/discotube/metadata/mmap_ids_train_rel",
    # Use the quotes filelist for struct too so both corpora cover the same
    # 222K YouTube IDs — a fair apples-to-apples comparison.
    "struct_filelist": "/scratch/<group>/discotube/metadata/mmap_ids_train_rel",
    "r4_filelist": "/scratch/<group>/discotube/metadata/m4rag_splits/filelist_train.txt",
    "msd_filelist": "/scratch/<group>/mmaps_msd/filelist_train_mmap.txt",
    "freesound_filelist": "/scratch/<group>/mmaps_freesound/filelist_full_train_mmap.txt",
}


def compute_corpus_stats(name: str, texts: Iterable[str]) -> dict:
    """Compute vocabulary statistics for a single corpus."""
    counter, total = vocab_from_texts(texts)
    types = len(counter)
    ttr = types / total if total > 0 else 0.0
    return {
        "name": name,
        "total_tokens": total,
        "unique_tokens": types,
        "type_token_ratio": round(ttr, 6),
    }


def compute_coverage(corpus_vocab: set[str], query_vocab: set[str]) -> dict:
    """Compute what fraction of query_vocab is covered by corpus_vocab."""
    if not query_vocab:
        return {"covered": 0, "total": 0, "coverage": 0.0}
    covered = len(query_vocab & corpus_vocab)
    return {
        "covered": covered,
        "total": len(query_vocab),
        "coverage": round(covered / len(query_vocab), 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quotes-text-file",
        type=str,
        default=DEFAULTS["quotes_text_file"],
    )
    parser.add_argument(
        "--struct-text-file",
        type=str,
        default=DEFAULTS["struct_text_file"],
    )
    parser.add_argument(
        "--m4rag-metadata-file",
        type=str,
        default=DEFAULTS["m4rag_metadata_file"],
    )
    parser.add_argument(
        "--freesound-text-file",
        type=str,
        default=DEFAULTS["freesound_text_file"],
    )
    parser.add_argument(
        "--pse-filelist",
        type=str,
        default=DEFAULTS["pse_filelist"],
    )
    # Training filelists — restrict text to IDs the model sees
    parser.add_argument(
        "--quotes-filelist",
        type=str,
        default=DEFAULTS["quotes_filelist"],
        help="Training filelist for quotes (discotube clean). IDs = path stem.",
    )
    parser.add_argument(
        "--struct-filelist",
        type=str,
        default=DEFAULTS["struct_filelist"],
        help="Training filelist for struct (discotube v3). IDs = path stem.",
    )
    parser.add_argument(
        "--r4-filelist",
        type=str,
        default=DEFAULTS["r4_filelist"],
        help="Training filelist for r4 (M4-RAG). IDs = path stem.",
    )
    parser.add_argument(
        "--msd-filelist",
        type=str,
        default=DEFAULTS["msd_filelist"],
        help="Training filelist for MSD. IDs = path stem (track_id).",
    )
    parser.add_argument(
        "--freesound-filelist",
        type=str,
        default=DEFAULTS["freesound_filelist"],
        help="Training filelist for Freesound. IDs = stem.split('_')[0].",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("text_corpus_stats.json"),
        help="Path for the JSON output report",
    )
    args = parser.parse_args()

    # --- Load training IDs from filelists ---
    print("Loading training filelist IDs...", flush=True)
    quotes_ids = _load_ids_stem(args.quotes_filelist)
    print(f"  quotes: {len(quotes_ids):,} IDs", flush=True)
    struct_ids = _load_ids_stem(args.struct_filelist)
    print(f"  struct: {len(struct_ids):,} IDs", flush=True)
    r4_ids = _load_ids_stem(args.r4_filelist)
    print(f"  r4: {len(r4_ids):,} IDs", flush=True)
    msd_ids = _load_ids_stem(args.msd_filelist)
    print(f"  msd: {len(msd_ids):,} IDs", flush=True)
    freesound_ids = _load_ids_freesound(args.freesound_filelist)
    print(f"  freesound: {len(freesound_ids):,} IDs", flush=True)

    # --- Build corpus iterators (filtered by training IDs) ---
    corpora = {
        "quotes": lambda: iter_quotes_texts(args.quotes_text_file, quotes_ids),
        "struct": lambda: iter_struct_texts(args.struct_text_file, struct_ids),
        "r4": lambda: iter_m4rag_texts(args.m4rag_metadata_file, r4_ids),
        "msd": lambda: iter_msd_texts(msd_ids),
        "freesound": lambda: iter_freesound_texts(
            args.freesound_text_file, freesound_ids
        ),
        "pse": lambda: iter_pse_texts(args.pse_filelist),
    }

    # --- Compute per-corpus stats ---
    stats: dict[str, dict] = {}
    vocabs: dict[str, set[str]] = {}

    for name, iter_fn in corpora.items():
        print(f"Processing {name}...", flush=True)
        counter, total = vocab_from_texts(iter_fn())
        types = len(counter)
        ttr = types / total if total > 0 else 0.0
        stats[name] = {
            "total_tokens": total,
            "unique_tokens": types,
            "type_token_ratio": round(ttr, 6),
        }
        vocabs[name] = set(counter.keys())
        print(
            f"  {name}: {total:,} tokens, {types:,} types, TTR={ttr:.4f}",
            flush=True,
        )

    # --- Load query vocabularies ---
    print("\nLoading MusicCaps query vocabulary...", flush=True)
    mc_vocab = load_musiccaps_vocab()
    print(f"  MusicCaps: {len(mc_vocab):,} unique tokens", flush=True)

    print("Loading SongDescriber query vocabulary...", flush=True)
    sd_vocab = load_songdescriber_vocab()
    print(f"  SongDescriber: {len(sd_vocab):,} unique tokens", flush=True)

    # --- Coverage ---
    coverage: dict[str, dict] = {}
    for name, vocab in vocabs.items():
        coverage[name] = {
            "musiccaps": compute_coverage(vocab, mc_vocab),
            "songdescriber": compute_coverage(vocab, sd_vocab),
        }

    # --- Pairwise corpus overlap ---
    corpus_names = list(vocabs.keys())
    overlap: dict[str, dict] = {}
    for a in corpus_names:
        overlap[a] = {}
        for b in corpus_names:
            shared = len(vocabs[a] & vocabs[b])
            overlap[a][b] = {
                "shared": shared,
                "frac_of_a": round(shared / len(vocabs[a]), 4) if vocabs[a] else 0.0,
                "frac_of_b": round(shared / len(vocabs[b]), 4) if vocabs[b] else 0.0,
            }

    # --- Exclusive tokens analysis ---
    # Tokens in quotes but not in any other training corpus
    other_vocabs = set()
    for name, vocab in vocabs.items():
        if name != "quotes":
            other_vocabs |= vocab
    quotes_exclusive = vocabs.get("quotes", set()) - other_vocabs

    # Tokens in MusicCaps/SongDescriber covered ONLY by quotes
    mc_only_quotes = (vocabs.get("quotes", set()) & mc_vocab) - other_vocabs
    sd_only_quotes = (vocabs.get("quotes", set()) & sd_vocab) - other_vocabs

    exclusive = {
        "quotes_exclusive_total": len(quotes_exclusive),
        "quotes_exclusive_examples": sorted(quotes_exclusive)[:50],
        "musiccaps_only_covered_by_quotes": len(mc_only_quotes),
        "musiccaps_only_covered_by_quotes_examples": sorted(mc_only_quotes)[:50],
        "songdescriber_only_covered_by_quotes": len(sd_only_quotes),
        "songdescriber_only_covered_by_quotes_examples": sorted(sd_only_quotes)[:50],
    }

    # --- Uncovered query tokens ---
    all_train_vocab = set()
    for vocab in vocabs.values():
        all_train_vocab |= vocab
    mc_uncovered = mc_vocab - all_train_vocab
    sd_uncovered = sd_vocab - all_train_vocab

    uncovered = {
        "musiccaps_uncovered_total": len(mc_uncovered),
        "musiccaps_uncovered_examples": sorted(mc_uncovered)[:50],
        "songdescriber_uncovered_total": len(sd_uncovered),
        "songdescriber_uncovered_examples": sorted(sd_uncovered)[:50],
    }

    # --- Assemble report ---
    training_id_counts = {
        "quotes": len(quotes_ids),
        "struct": len(struct_ids),
        "r4": len(r4_ids),
        "msd": len(msd_ids),
        "freesound": len(freesound_ids),
        "pse": "N/A (text derived from filelist paths)",
    }
    report = {
        "training_id_counts": training_id_counts,
        "corpus_stats": stats,
        "query_vocab_sizes": {
            "musiccaps": len(mc_vocab),
            "songdescriber": len(sd_vocab),
        },
        "coverage": coverage,
        "pairwise_overlap": overlap,
        "exclusive_analysis": exclusive,
        "uncovered_query_tokens": uncovered,
    }

    # --- Write output ---
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {args.out_path}")

    # --- Print summary table ---
    print("\n" + "=" * 80)
    print("CORPUS VOCABULARY STATISTICS")
    print("=" * 80)
    print(f"{'Corpus':<15} {'Total tokens':>14} {'Unique types':>14} {'TTR':>10}")
    print("-" * 55)
    for name, s in stats.items():
        print(
            f"{name:<15} {s['total_tokens']:>14,} {s['unique_tokens']:>14,} "
            f"{s['type_token_ratio']:>10.4f}"
        )

    print("\n" + "=" * 80)
    print("QUERY VOCABULARY COVERAGE")
    print("=" * 80)
    print(f"{'Corpus':<15} {'MusicCaps cov.':>16} {'SongDescriber cov.':>20}")
    print("-" * 55)
    for name in corpus_names:
        mc_cov = coverage[name]["musiccaps"]["coverage"]
        sd_cov = coverage[name]["songdescriber"]["coverage"]
        print(f"{name:<15} {mc_cov:>15.1%} {sd_cov:>19.1%}")

    print("\n" + "=" * 80)
    print("EXCLUSIVE ANALYSIS")
    print("=" * 80)
    print(
        f"Tokens exclusive to quotes (not in struct/r4/MSD/FS/PSE): "
        f"{exclusive['quotes_exclusive_total']:,}"
    )
    print(
        f"MusicCaps query tokens covered ONLY by quotes: "
        f"{exclusive['musiccaps_only_covered_by_quotes']:,}"
    )
    print(
        f"SongDescriber query tokens covered ONLY by quotes: "
        f"{exclusive['songdescriber_only_covered_by_quotes']:,}"
    )

    if mc_only_quotes:
        examples = sorted(mc_only_quotes)[:20]
        print(f"  Examples (MusicCaps): {', '.join(examples)}")
    if sd_only_quotes:
        examples = sorted(sd_only_quotes)[:20]
        print(f"  Examples (SongDescriber): {', '.join(examples)}")


if __name__ == "__main__":
    main()
