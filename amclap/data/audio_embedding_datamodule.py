import traceback
from collections import OrderedDict
from pathlib import Path
import concurrent

import numpy
import torch
import gin.torch
import lightning.pytorch as L
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Resample
from tqdm import tqdm
from essentia.standard import MetadataReader


class AudioEmbeddingDataset(Dataset):
    """Dataset for loading audio files."""

    def __init__(
        self,
        data_dir: Path,
        file_format: str,
        new_freq: int,
        mono: bool,
        half_precision: bool,
        overlap_ratio: float,
        n_seconds: int,
        last_chunk_ratio: float,
        orig_freq: int | None = None,
        filelist_path: Path | None = None,
        cache_size: int = 1,
        embeddings_dir: Path | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.file_format = file_format

        if "discotube" in str(data_dir):
            self.is_bsc_discotube = True
        else:
            self.is_bsc_discotube = False

        if filelist_path:
            if Path(filelist_path).exists():
                print(f"Loading files from {filelist_path}")
                with open(filelist_path, "r") as f:
                    # Resolve paths relative to data_dir
                    if self.is_bsc_discotube:
                        print("Looking for BSC paths")
                        self.filelist = []
                        for line in f:
                            line = line.strip()
                            try:
                                path = self.get_bsc_path(self.data_dir, Path(line))
                                self.filelist.append(path)
                            except FileNotFoundError:
                                pass
                    else:
                        self.filelist = [
                            Path(self.data_dir, line.strip())
                            for line in f
                            if line.strip()
                        ]
            else:
                raise FileNotFoundError(f"Filelist {filelist_path} not found.")
        else:
            self.filelist = sorted(self.data_dir.rglob(f"*.{file_format}"))

        assert len(self.filelist) > 0, (
            f"No files found in {self.data_dir} (via filelist={filelist_path})"
        )
        print(f"Found {len(self.filelist)} *.{file_format} files.")

        # Skip files whose embeddings already exist
        if embeddings_dir is not None:
            embeddings_dir = Path(embeddings_dir)
            total = len(self.filelist)
            self.filelist = [
                fp
                for fp in self.filelist
                if not (
                    embeddings_dir / Path(fp).stem[:3] / f"{Path(fp).stem}.pt"
                ).exists()
            ]
            skipped = total - len(self.filelist)
            print(f"Skipping {skipped}/{total} files with existing embeddings.")
            if len(self.filelist) == 0:
                print("All embeddings already computed. Nothing to extract.")

        self.new_freq = new_freq
        self.orig_freq = orig_freq
        self.mono = mono
        self.half_precision = half_precision

        self.index = dict()  # idx: (file_path, seg, n_segments, sample_rate)
        self.audio_cache = OrderedDict()  # LRU cache for loaded audio
        self._cache_maxsize = cache_size

        self.overlap_ratio = overlap_ratio
        self.last_chunk_ratio = last_chunk_ratio

        self.n_seconds = n_seconds

        assert self.overlap_ratio >= 0 and self.overlap_ratio < 1, (
            "Overlap ratio must be between 0 and 1."
        )

        if self.file_format == "mmap":
            assert self.orig_freq is not None, (
                "orig_freq is required for mmap file format."
            )
            self.resample = None
            if self.orig_freq != self.new_freq:
                self.resample = Resample(
                    orig_freq=self.orig_freq, new_freq=self.new_freq
                )

        self.compute_dataset_segments()

        print(f"Found {len(self)} segments in {len(self.filelist)} files.")

    def _get_mmap_duration_samples(self, filepath):
        """Get the number of samples in an mmap file (stored as float16)."""
        return filepath.stat().st_size // 2

    def get_bsc_path(self, base: Path, rel_path: Path) -> Path:
        """
        Resolves the existing file path for the given relative path.

        This function searches for the specified file under two possible directory options:
        'discotube-2020-09' and 'discotube-2023-03/audio-new'. If the file is found in either
        location, the corresponding path is returned. If the file does not exist in either
        location, a FileNotFoundError is raised.

        Args:
            base (Path): The base directory where the search begins.
            rel_path (Path): The relative path of the file to locate.

        Returns:
            Path: The resolved path of the file if found in one of the predefined locations.

        Raises:
            FileNotFoundError: If the file is not found in any of the predefined locations.
        """
        opt_1 = Path("discotube-2020-09")
        opt_2 = Path("discotube-2023-03", "audio-new")

        path_1 = base / opt_1 / rel_path
        path_2 = base / opt_2 / rel_path

        if (path_1).exists():
            return path_1
        elif (path_2).exists():
            return path_2
        else:
            raise FileNotFoundError(f"File not found in either location: {rel_path}")

    def _samples_to_chunks(self, n_samples, chunk_size):
        """Compute the number of segments for a file given its sample count."""
        # hop size in samples
        hop_size = int(chunk_size * (1 - self.overlap_ratio))
        # initial number of segments based on full hops
        n_chunks = int(1 + max(0, (n_samples - chunk_size) // hop_size))
        # remaining samples after n_s full hops
        tail_size = n_samples - n_chunks * hop_size

        # add an extra segment if the tail is long enough
        if tail_size >= chunk_size * self.last_chunk_ratio:
            n_chunks += 1
        return n_chunks

    def compute_dataset_segments(self):
        """Compute segments for all audio files (metadata only, no audio loading)."""
        self.index = dict()
        self.audio_cache = OrderedDict()

        print("Computing segments from audio metadata...")

        i = 0
        if self.file_format == "mmap":
            for filepath in tqdm(self.filelist):
                try:
                    n_samples = self._get_mmap_duration_samples(filepath)
                    chunk_size = int(self.n_seconds * self.orig_freq)
                    n_chunks = self._samples_to_chunks(n_samples, chunk_size)

                    for j in range(n_chunks):
                        self.index[i] = (filepath, j, n_chunks, self.orig_freq)
                        i += 1
                except Exception:
                    traceback.print_exc()
                    print(f"Error processing file {filepath}")
                    continue
        else:
            chunk_size = int(self.n_seconds * self.new_freq)

            with concurrent.futures.ProcessPoolExecutor() as executor:
                future_to_path = {
                    executor.submit(self._get_audio_file_duration, filepath): filepath
                    for filepath in self.filelist
                }

                for future in tqdm(
                    concurrent.futures.as_completed(future_to_path),
                    desc="Processing audio metadata",
                    total=len(self.filelist),
                ):
                    try:
                        filepath = future_to_path[future]
                        duration_s, file_sr = future.result()
                        n_samples = int(duration_s * self.new_freq)
                        n_chunks = self._samples_to_chunks(n_samples, chunk_size)

                        for j in range(n_chunks):
                            self.index[i] = (filepath, j, n_chunks, file_sr)
                            i += 1
                    except Exception:
                        traceback.print_exc()
                        print(f"Error processing file {filepath}")
                        continue

    @staticmethod
    def _get_audio_file_duration(filepath):
        """Get audio duration and sample rate using Essentia MetadataReader."""

        # Use Essentia MetadataReader for fast metadata reads (no full decode)
        meta = MetadataReader(filename=str(filepath), failOnError=True)
        # MetadataReader returns many outputs; duration and sampleRate
        # are at fixed positions in the output tuple
        results = meta()
        duration_s = results[8]  # duration in seconds
        file_sr = results[10]  # sample rate in Hz

        return duration_s, file_sr

    def __len__(self):
        return len(self.index)

    def _load_mmap_segment(self, file_path, segment):
        """Load a segment from an mmap file."""
        n_samples = self._get_mmap_duration_samples(file_path)
        n_c = int(self.n_seconds * self.orig_freq)
        n_h = int(n_c * (1 - self.overlap_ratio))

        start = int(segment * n_h)
        end = start + n_c

        if end <= n_samples:
            offset_bytes = start * 2  # float16 = 2 bytes
            mmap = numpy.memmap(
                file_path,
                offset=offset_bytes,
                dtype="float16",
                mode="r",
                shape=(end - start,),
            )
            audio = torch.from_numpy(numpy.array(mmap))
            del mmap
        else:
            # Last segment: load what's available and zero-pad
            remaining = n_samples - start
            if remaining > 0:
                offset_bytes = start * 2
                mmap = numpy.memmap(
                    file_path,
                    offset=offset_bytes,
                    dtype="float16",
                    mode="r",
                    shape=(remaining,),
                )
                audio = torch.from_numpy(numpy.array(mmap))
                del mmap
                pad = torch.zeros(n_c - remaining)
                audio = torch.cat([audio, pad])
            else:
                audio = torch.zeros(n_c)

        # Resample from orig_freq to new_freq
        if self.resample is not None:
            audio = audio.float()
            audio = self.resample(audio)

        return audio

    def _get_cached_audio(self, file_path, file_sr):
        """Get audio from LRU cache, loading on miss."""
        if file_path in self.audio_cache:
            self.audio_cache.move_to_end(file_path)
            return self.audio_cache[file_path]

        # Load audio with Essentia (mono, resampled to new_freq)
        from essentia.standard import MonoLoader

        audio_np = MonoLoader(
            filename=str(file_path),
            sampleRate=self.new_freq,
            resampleQuality=4,
        )()
        audio = torch.from_numpy(audio_np)

        self.audio_cache[file_path] = audio
        # Evict oldest if over cache size
        while len(self.audio_cache) > self._cache_maxsize:
            self.audio_cache.popitem(last=False)

        return audio

    def __getitem__(self, idx):
        # Get the file path, segment index, total segments, and sample rate
        file_path, segment, n_segments, file_sr = self.index[idx]

        try:
            if self.file_format == "mmap":
                audio_segment = self._load_mmap_segment(file_path, segment)
            else:
                # Get audio from LRU cache (lazy-loaded)
                audio = self._get_cached_audio(file_path, file_sr)

                # Calculate segment boundaries
                n_c = int(self.n_seconds * self.new_freq)
                n_h = int(n_c * (1 - self.overlap_ratio))
                start = int(segment * n_h)
                end = start + n_c

                # Extract segment
                if end <= len(audio):
                    audio_segment = audio[start:end]
                else:
                    # Handle last segment that may be shorter
                    audio_segment = audio[start:]
                    # Zero pad if necessary
                    if len(audio_segment) < n_c:
                        pad = torch.zeros(n_c - len(audio_segment))
                        audio_segment = torch.cat([audio_segment, pad])

            # Work with appropriate precision
            if self.half_precision:
                audio_segment = audio_segment.half()
            else:
                audio_segment = audio_segment.float()

            return audio_segment, str(file_path), segment, n_segments

        except Exception:
            print(traceback.format_exc())
            print(f"Error loading file {file_path}")
            # Return a zero tensor so the default collator can batch it
            n_c = int(self.n_seconds * self.new_freq)
            dtype = torch.float16 if self.half_precision else torch.float32
            return torch.zeros(n_c, dtype=dtype), str(file_path), segment, n_segments


@gin.configurable
class AudioEmbeddingDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        file_format: str,
        new_freq: int,
        mono: bool,
        half_precision: bool,
        num_workers: int,
        batch_size: int,
        overlap_ratio: float,
        n_seconds: int,
        last_chunk_ratio: float,
        orig_freq: int = None,
        filelist_path: Path = None,
        embeddings_dir: Path = None,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.file_format = file_format
        self.new_freq = new_freq
        self.orig_freq = orig_freq
        self.mono = mono
        self.half_precision = half_precision
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.overlap_ratio = overlap_ratio
        self.n_seconds = n_seconds
        self.last_chunk_ratio = last_chunk_ratio
        self.filelist_path = filelist_path
        self.embeddings_dir = embeddings_dir

        assert 0 <= self.overlap_ratio < 1, "overlap_ratio must be between 0 and 1."
        assert 0 < self.last_chunk_ratio <= 1, (
            "last_chunk_ratio must be between 0 and 1."
        )

        self.dataset = AudioEmbeddingDataset(
            data_dir=self.data_dir,
            file_format=self.file_format,
            new_freq=self.new_freq,
            mono=self.mono,
            half_precision=self.half_precision,
            overlap_ratio=self.overlap_ratio,
            n_seconds=self.n_seconds,
            last_chunk_ratio=self.last_chunk_ratio,
            orig_freq=self.orig_freq,
            filelist_path=self.filelist_path,
            cache_size=(self.num_workers or 0) + 1,
            embeddings_dir=self.embeddings_dir,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
