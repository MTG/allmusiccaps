import concurrent.futures
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm

import essentia.standard as es
# from librosa import load


def save_as_mmap(waveform, mmap_path):
    """Save waveform as a memory-mapped file in float16 format."""
    waveform = waveform.astype(np.float16)
    mmap_file = np.memmap(
        mmap_path,
        dtype=np.float16,
        mode="w+",
        shape=waveform.shape,
    )
    mmap_file[:] = waveform[:]
    mmap_file.flush()

    # Ensure the file is properly closed
    del mmap_file


def get_output_path(input_path: Path, input_dir: Path, output_dir: Path) -> Path:
    """Get the corresponding output path in the output directory."""
    rel_path = input_path.relative_to(input_dir)
    return output_dir / rel_path.with_suffix(".mmap")


def process_audio_files(
    input_filelist: list,
    sample_rate: float,
):
    for audio_path, mmap_path in tqdm(
        input_filelist, desc="Processing audio files", leave=False
    ):
        try:
            # Skip if the mmap file already exists
            if mmap_path.exists():
                print(f"Skipping {mmap_path}, already exists.")
                continue

            # Ensure the output directory exists
            mmap_path.parent.mkdir(parents=True, exist_ok=True)

            # Load audio
            waveform = es.MonoLoader(
                filename=str(audio_path),
                sampleRate=sample_rate,
                resampleQuality=4,
            )()

            # waveform, _ = load(
            #     audio_path, mono=True, sr=sample_rate, res_type="soxr_lq"
            # )

            save_as_mmap(waveform, mmap_path)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue


if __name__ == "__main__":
    parser = ArgumentParser(description="Process audio files in parallel.")
    parser.add_argument(
        "--n-tasks",
        type=int,
        required=True,
        help="Number of parallel tasks to process",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory containing audio files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory to save processed files",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=16000,
        help="Sample rate for audio processing",
    )
    parser.add_argument(
        "--filelist",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    # List all files

    if args.filelist:
        with open(args.filelist, "r") as f:
            n_files = [Path(args.input_dir, line.strip()) for line in f.readlines()]
    else:
        n_files = list(args.input_dir.rglob("*.*"))

    files = [(f, get_output_path(f, args.input_dir, args.output_dir)) for f in n_files]

    skip = set(
        [
            "TRHCRKN128F427A8D5",
        ]
    )

    # Filter out files that already have corresponding output files
    new_files = []
    for i, o in tqdm(files, desc="Checking existing files"):
        if i.stem in skip:
            continue

        if not o.exists():
            new_files.append((i, o))
    files = new_files
    # files = [(i, o) for i, o in files if not o.exists()]

    # compute output files

    if len(files) == 0:
        print("All files have been processed. Exiting.")
        exit(0)

    chunk_size = len(files) // args.n_tasks

    if chunk_size == 0:
        raise ValueError("Number of tasks exceeds number of files to process.")

    chunks = [files[i * chunk_size : (i + 1) * chunk_size] for i in range(args.n_tasks)]
    # Last chunk gets the remainder
    if args.n_tasks > 0:
        chunks[-1].extend(files[args.n_tasks * chunk_size :])

    n_tasks = min(args.n_tasks, len(files))

    print(f"Total files: {len(n_files)}")
    print(f"Files to process: {len(files)}")
    print(f"Files skipped (already exist): {len(n_files) - len(files)}")
    print(f"Chunk size: {chunk_size}")
    print(f"Running {args.n_tasks} tasks in parallel")

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_tasks) as executor:
        futures = [
            executor.submit(process_audio_files, chunk, args.sample_rate)
            for chunk in chunks
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
