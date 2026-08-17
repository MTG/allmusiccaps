"""Cross-reference symbolic music metadata with audio features from DiscoTube .mmap files.

Enriches a CSV of title-matched pairs with audio-derived duration, tempo, and key
estimates, plus flags for whether each pair matches on those features and whether the
track has LLM text descriptions available.
"""

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import essentia.standard as es
import numpy as np
import pandas as pd
from tqdm import tqdm

# Relative major/minor pairs for key matching.
# Each entry maps a key to its relative major or minor counterpart.
RELATIVE_KEYS = {
    "C": "Am",
    "Am": "C",
    "G": "Em",
    "Em": "G",
    "D": "Bm",
    "Bm": "D",
    "A": "F#m",
    "F#m": "A",
    "E": "C#m",
    "C#m": "E",
    "B": "G#m",
    "G#m": "B",
    "F#": "D#m",
    "D#m": "F#",
    "Gb": "Ebm",
    "Ebm": "Gb",
    "Db": "Bbm",
    "Bbm": "Db",
    "C#": "A#m",
    "A#m": "C#",
    "Ab": "Fm",
    "Fm": "Ab",
    "Eb": "Cm",
    "Cm": "Eb",
    "Bb": "Gm",
    "Gm": "Bb",
    "F": "Dm",
    "Dm": "F",
}

# Enharmonic equivalences for normalization.
ENHARMONIC = {
    "C#": "Db",
    "D#": "Eb",
    "F#": "Gb",
    "G#": "Ab",
    "A#": "Bb",
    "C#m": "Dbm",
    "D#m": "Ebm",
    "F#m": "Gbm",
    "G#m": "Abm",
    "A#m": "Bbm",
}


def normalize_key(key: str) -> str:
    """Normalize a key string to a canonical form using flats."""
    return ENHARMONIC.get(key, key)


def keys_match(key_symbolic: str, key_audio: str) -> bool:
    """Check if two keys match (identical or relative major/minor pair)."""
    if not key_symbolic or not key_audio:
        return False

    k1 = normalize_key(key_symbolic)
    k2 = normalize_key(key_audio)

    if k1 == k2:
        return True

    # Check relative major/minor relationship.
    rel = RELATIVE_KEYS.get(k2)
    if rel and normalize_key(rel) == k1:
        return True

    return False


def tempo_matches(tempo_symbolic: float, tempo_audio: float, tol: float) -> bool:
    """Check if two tempos match within tolerance, including half/double tempo."""
    if np.isnan(tempo_symbolic) or np.isnan(tempo_audio):
        return False

    if abs(tempo_audio - tempo_symbolic) <= tol:
        return True
    if abs(tempo_audio * 2 - tempo_symbolic) <= tol * 2:
        return True
    if abs(tempo_audio / 2 - tempo_symbolic) <= tol / 2:
        return True

    return False


def resolve_mmap_path(mmap_base: Path, discotube_id: str) -> Path | None:
    """Resolve the .mmap file path for a DiscoTube ID.

    Mirrors DiscotubeStructuredTextAudioDataset.get_bsc_path.
    """
    rel_path = Path(discotube_id[:2], f"{discotube_id}.mmap")

    opt_1 = Path("discotube-2020-09")
    opt_2 = Path("discotube-2023-03", "audio-new")

    path_1 = mmap_base / opt_1 / rel_path
    path_2 = mmap_base / opt_2 / rel_path

    if path_1.exists():
        return path_1
    elif path_2.exists():
        return path_2
    else:
        return None


def format_key(key: str, scale: str) -> str:
    """Format Essentia key output to compact notation (e.g. 'Bb', 'Bbm')."""
    if scale == "minor":
        return f"{key}m"
    return key


def process_track(args: tuple) -> dict:
    """Analyze a single .mmap track: compute duration, tempo, and key."""
    discotube_id, mmap_base, sample_rate = args

    file_path = resolve_mmap_path(Path(mmap_base), discotube_id)
    if file_path is None:
        return {"discotube_id": discotube_id, "missing": True}

    n_samples = file_path.stat().st_size // 2  # 2 bytes per float16
    duration = n_samples / sample_rate

    audio = np.memmap(file_path, dtype="float16", mode="r", shape=(n_samples,))
    audio = np.array(audio, dtype=np.float32)

    rhythm_extractor = es.RhythmExtractor2013()
    resample = es.Resample(inputSampleRate=sample_rate, outputSampleRate=44100)
    bpm, _ticks, _confidence, _estimates, _intervals = rhythm_extractor(resample(audio))

    key_extractor = es.KeyExtractor(sampleRate=sample_rate)
    key, scale, _strength = key_extractor(audio)

    return {
        "discotube_id": discotube_id,
        "missing": False,
        "duration_audio": round(duration, 2),
        "tempo_audio": round(bpm, 2),
        "key_audio": format_key(key, scale),
    }


def load_text_ids(text_filelist: str) -> set[str]:
    """Load the set of DiscoTube IDs that have LLM text descriptions."""
    ids = set()
    with open(text_filelist) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(Path(line).stem)
    return ids


def main():
    parser = argparse.ArgumentParser(
        description="Enrich symbolic-audio matches with audio features."
    )
    parser.add_argument(
        "--csv-input",
        default="title_matches_processed.csv",
    )
    parser.add_argument(
        "--csv-output",
        default="discotube_audio_text_symbolic_match.csv",
    )
    parser.add_argument(
        "--mmap-base",
        default="/scratch/<group>/mmaps_discotube/",
    )
    parser.add_argument(
        "--text-filelist",
        default="/scratch/<group>/discotube/metadata/llama33_promptv3_filelist",
    )
    parser.add_argument("--duration-tol", type=float, default=10.0)
    parser.add_argument("--tempo-tol", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count())
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--max-rows", type=int, default=None, help="Process only the first N rows."
    )
    args = parser.parse_args()

    # Load CSV.
    df = pd.read_csv(args.csv_input, nrows=args.max_rows)
    print(f"Loaded {len(df)} rows from {args.csv_input}")

    # Build discotube_id -> row indices mapping.
    id_to_rows: dict[str, list[int]] = {}
    for idx, row in df.iterrows():
        did = row["discotube_id"]
        id_to_rows.setdefault(did, []).append(idx)

    unique_ids = list(id_to_rows.keys())
    print(f"Unique DiscoTube IDs: {len(unique_ids)}")

    # Load text IDs.
    text_ids = load_text_ids(args.text_filelist)
    print(f"Text descriptions available for {len(text_ids)} tracks")

    # Initialize new columns.
    df["duration_audio"] = np.nan
    df["tempo_audio"] = np.nan
    df["key_audio"] = ""
    df["duration_match"] = False
    df["tempo_match"] = False
    df["key_match"] = False
    df["has_text"] = False
    df["missing_mmap"] = False

    # Process tracks in parallel.
    worker_args = [(did, args.mmap_base, args.sample_rate) for did in unique_ids]

    results = {}
    with Pool(args.num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_track, worker_args),
            total=len(worker_args),
            desc="Analyzing tracks",
        ):
            results[result["discotube_id"]] = result

    # Assign results to DataFrame rows.
    for did, row_indices in id_to_rows.items():
        result = results[did]
        has_text = did in text_ids

        for idx in row_indices:
            df.at[idx, "has_text"] = has_text

            if result["missing"]:
                df.at[idx, "missing_mmap"] = True
                continue

            df.at[idx, "duration_audio"] = result["duration_audio"]
            df.at[idx, "tempo_audio"] = result["tempo_audio"]
            df.at[idx, "key_audio"] = result["key_audio"]

            # Duration match.
            duration_symbolic = df.at[idx, "duration"]
            if not pd.isna(duration_symbolic):
                df.at[idx, "duration_match"] = (
                    abs(result["duration_audio"] - duration_symbolic)
                    <= args.duration_tol
                )

            # Tempo match.
            tempo_symbolic = df.at[idx, "tempo"]
            if not pd.isna(tempo_symbolic):
                df.at[idx, "tempo_match"] = tempo_matches(
                    tempo_symbolic, result["tempo_audio"], args.tempo_tol
                )

            # Key match.
            key_symbolic = df.at[idx, "key"]
            if isinstance(key_symbolic, str) and key_symbolic:
                df.at[idx, "key_match"] = keys_match(key_symbolic, result["key_audio"])

    # Save enriched CSV.
    df.to_csv(args.csv_output, index=False)
    print(f"\nSaved enriched CSV to {args.csv_output}")

    # Print statistics.
    total = len(df)
    missing = df["missing_mmap"].sum()
    non_missing = df[~df["missing_mmap"]]

    # Only count mismatches for rows that have symbolic metadata.
    has_duration = non_missing["duration"].notna()
    has_tempo = non_missing["tempo"].notna()
    has_key = non_missing["key"].apply(lambda x: isinstance(x, str) and x != "")

    duration_mismatch = (
        has_duration.sum() - non_missing.loc[has_duration, "duration_match"].sum()
    )
    tempo_mismatch = has_tempo.sum() - non_missing.loc[has_tempo, "tempo_match"].sum()
    key_mismatch = has_key.sum() - non_missing.loc[has_key, "key_match"].sum()
    no_text = total - df["has_text"].sum()

    # All audio checks pass (duration + tempo + key, where metadata exists).
    audio_pass = non_missing[
        (non_missing["duration_match"] | ~has_duration)
        & (non_missing["tempo_match"] | ~has_tempo)
        & (non_missing["key_match"] | ~has_key)
    ]
    all_pass = audio_pass[audio_pass["has_text"]]

    print(f"\n{'=' * 50}")
    print("Statistics")
    print(f"{'=' * 50}")
    print(f"Total rows:              {total}")
    print(f"Missing .mmap:           {missing} ({missing / total * 100:.1f}%)")
    print(
        f"Duration mismatches:     {duration_mismatch} ({duration_mismatch / total * 100:.1f}%)"
    )
    print(
        f"Tempo mismatches:        {tempo_mismatch} ({tempo_mismatch / total * 100:.1f}%)"
    )
    print(
        f"Key mismatches:          {key_mismatch} ({key_mismatch / total * 100:.1f}%)"
    )
    print(f"No text match:           {no_text} ({no_text / total * 100:.1f}%)")
    print(
        f"Pass all audio checks:   {len(audio_pass)} ({len(audio_pass) / total * 100:.1f}%)"
    )
    print(
        f"Pass all (incl. text):   {len(all_pass)} ({len(all_pass) / total * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
