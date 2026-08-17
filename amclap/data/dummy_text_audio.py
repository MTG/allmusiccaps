"""Dummy text-audio dataset for local debugging without real data."""

import random

import gin
import torch
import lightning.pytorch as L
from torch.utils.data import Dataset, DataLoader

from .data_utils import collate_with_skip


TEXT_TEMPLATES = [
    "Electronic track with deep bass and synthesizers",
    "Acoustic guitar melody with folk influences",
    "Classical piano piece in a minor key",
    "Jazz ensemble with saxophone and drums",
    "Heavy metal song with distorted guitars",
    "Ambient soundscape with atmospheric textures",
    "Hip hop beat with sampled drums",
    "Indie rock song with jangly guitars",
    "EDM drop with build-up and release",
    "Orchestral soundtrack with strings and brass",
    "Blues track with slide guitar",
    "Reggae rhythm with offbeat accents",
    "Country song with pedal steel guitar",
    "Punk rock with fast tempo and distortion",
    "Soul music with powerful vocals",
    "Funk groove with slap bass",
    "World music fusion with ethnic instruments",
    "Lo-fi hip hop with vinyl crackle",
    "Progressive rock with complex time signatures",
    "Chillout electronica with soft pads",
]


@gin.configurable
class DummyTextAudioDataset(Dataset):
    """Dataset that generates random audio tensors and samples text from templates.

    Useful for local debugging without needing real audio files or metadata.
    """

    def __init__(
        self,
        num_samples: int = 100,
        num_frames: int = 240000,  # 10s at 24kHz
        sample_rate: int = 24000,
        half_precision: bool = True,
        n_text_views: int = 1,
        n_audio_views: int = 1,
    ):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        self.half_precision = half_precision
        self.n_text_views = n_text_views
        self.n_audio_views = n_audio_views

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        audio_views = []
        for _ in range(self.n_audio_views):
            audio = torch.randn(self.num_frames)
            if self.half_precision:
                audio = audio.half()
            audio_views.append(audio)

        text_views = [random.choice(TEXT_TEMPLATES) for _ in range(self.n_text_views)]
        mask = None

        return [audio_views, text_views, mask]


@gin.configurable
class DummyTextAudioDataModule(L.LightningDataModule):
    """Lightning DataModule for dummy text-audio data.

    Generates synthetic audio and text for local debugging without external files.
    """

    def __init__(
        self,
        batch_size: int = 4,
        num_workers: int = 0,
        num_train_samples: int = 100,
        num_val_samples: int = 20,
        num_frames: int = 240000,
        sample_rate: int = 24000,
        half_precision: bool = True,
        n_text_views: int = 1,
        n_audio_views: int = 1,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        self.half_precision = half_precision
        self.n_text_views = n_text_views
        self.n_audio_views = n_audio_views

    def setup(self, stage: str):
        self.dataset_train = DummyTextAudioDataset(
            num_samples=self.num_train_samples,
            num_frames=self.num_frames,
            sample_rate=self.sample_rate,
            half_precision=self.half_precision,
            n_text_views=self.n_text_views,
            n_audio_views=self.n_audio_views,
        )
        self.dataset_val = DummyTextAudioDataset(
            num_samples=self.num_val_samples,
            num_frames=self.num_frames,
            sample_rate=self.sample_rate,
            half_precision=self.half_precision,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=collate_with_skip,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=collate_with_skip,
        )
