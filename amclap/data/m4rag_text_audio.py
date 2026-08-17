"""Text-audio dataset for M4-RAG with Discotube audio."""

import json
import random
from pathlib import Path
from typing import Union

import gin
import lightning.pytorch as L
import torch
from torch.utils.data import DataLoader

from ..clap_modules.audio_augmentations import waveform_augment

from .data_utils import AudioDataset, collate_with_skip


@gin.configurable
class M4RAGTextAudioDataset(AudioDataset):
    """Text-audio dataset for M4-RAG.

    Uses rich music descriptions from M4-RAG dataset matched with Discotube audio.
    Text fields: background, analysis, description, scene, genres, tags.
    """

    def __init__(
        self,
        data_dir: Path,
        filelist: Path,
        metadata: dict,
        num_frames: int,
        text_fields: list[str] = None,
        frame_offset: Union[int, str] = "random",
        use_audio_type_token: bool = False,
        audio_type_token_dropout: float = 0.0,
        n_text_views: int = 1,
        n_audio_views: int = 1,
        audio_noise_std: float = 0.0,
        audio_gain_range: tuple = (1.0, 1.0),
    ):
        super().__init__(
            data_dir=data_dir,
            filelist=filelist,
            frame_offset=frame_offset,
            num_frames=num_frames,
        )

        self.metadata = metadata
        self.text_fields = text_fields or [
            "description",
            "background",
            "analysis",
            "scene",
        ]

        self.use_audio_type_token = use_audio_type_token
        self.audio_type_token_dropout = audio_type_token_dropout
        self.n_text_views = n_text_views
        self.n_audio_views = n_audio_views
        self.audio_noise_std = audio_noise_std
        self.audio_gain_range = audio_gain_range

    def __len__(self):
        return len(self.filelist)

    @staticmethod
    def get_youtube_id(path: Path) -> str:
        return path.stem

    def __getitem__(self, idx):
        max_attempts = 10
        attempt = 0
        while attempt < max_attempts:
            try:
                path = Path(self.filelist[idx])
                file_path = self.data_dir / path
                yt_id = self.get_youtube_id(path)

                audio_views = []
                for _ in range(self.n_audio_views):
                    audio = self.load_audio(file_path, frame_offset=self.frame_offset)
                    assert not torch.isnan(audio).any()
                    audio = waveform_augment(
                        audio, self.audio_noise_std, self.audio_gain_range
                    )
                    audio_views.append(audio)

                text_views = [
                    self.get_text_metadata(yt_id) for _ in range(self.n_text_views)
                ]
                m = None
                return [audio_views, text_views, m]
            except Exception:
                attempt += 1
                idx = random.randint(0, len(self.filelist) - 1)

        return [[None] * self.n_audio_views, [None] * self.n_text_views, None]

    def get_text_metadata(self, yt_id: str) -> str:
        """Get text from M4-RAG metadata."""
        entry = self.metadata.get(yt_id)
        if entry is None:
            raise ValueError(f"No metadata for {yt_id}")

        # Collect available text fields
        texts = []
        for field in self.text_fields:
            if field in entry and entry[field]:
                value = entry[field]
                if isinstance(value, list):
                    value = ", ".join(value)
                if value.strip():
                    texts.append(value.strip())

        if not texts:
            raise ValueError(f"No text available for {yt_id}")

        # Randomly sample one of the text fields
        text = random.choice(texts)

        # Prepend audio type token with dropout
        if (
            self.use_audio_type_token
            and random.random() >= self.audio_type_token_dropout
        ):
            text = f"[MUSIC] {text}"

        return text


@gin.configurable
class M4RAGTextAudioDataModule(L.LightningDataModule):
    """AudioDataModule for M4-RAG dataset with Discotube audio."""

    def __init__(
        self,
        batch_size: int,
        data_dir: Path,
        filelist_train: Path,
        filelist_val: Path,
        metadata_file: Path,
        num_workers: int,
        text_fields: list[str] = None,
        n_text_views: int = 1,
        n_audio_views: int = 1,
        audio_noise_std: float = 0.0,
        audio_gain_range: tuple = (1.0, 1.0),
    ):
        super().__init__()

        self.batch_size = batch_size

        self.data_dir = Path(data_dir)
        self.filelist_train = Path(filelist_train)
        self.filelist_val = Path(filelist_val)
        self.metadata_file = Path(metadata_file)

        self.num_workers = num_workers
        self.text_fields = text_fields
        self.n_text_views = n_text_views
        self.n_audio_views = n_audio_views
        self.audio_noise_std = audio_noise_std
        self.audio_gain_range = audio_gain_range

    def setup(self, stage: str):
        # Load metadata from JSONL
        print(f"Loading M4-RAG metadata from {self.metadata_file}...")
        self.metadata = {}
        with open(self.metadata_file, "r") as f:
            for line in f:
                entry = json.loads(line)
                self.metadata[entry["id"]] = entry
        print(f"Loaded {len(self.metadata)} metadata entries")

        self.dataset_train = M4RAGTextAudioDataset(
            self.data_dir,
            filelist=self.filelist_train,
            metadata=self.metadata,
            text_fields=self.text_fields,
            n_text_views=self.n_text_views,
            n_audio_views=self.n_audio_views,
            audio_noise_std=self.audio_noise_std,
            audio_gain_range=self.audio_gain_range,
        )
        self.dataset_val = M4RAGTextAudioDataset(
            self.data_dir,
            filelist=self.filelist_val,
            metadata=self.metadata,
            text_fields=self.text_fields,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_with_skip,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_with_skip,
        )
