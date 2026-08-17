import random
import warnings
from pathlib import Path
from typing import Union

import gin
import lightning.pytorch as L
import torch
import torch.distributed as dist
import pandas as pd
from torch.utils.data import DataLoader
from datasets import load_dataset

from ..clap_modules.audio_augmentations import waveform_augment

from .data_utils import AudioDataset, collate_with_skip


@gin.configurable
class MSDTextAudioDataset(AudioDataset):
    """Text-audio dataset for MSD.
    It expects a jsonl file with description and tags for each freesound id.
    `description_prob` controls the probability of sampling the description vs the tags.
    """

    def __init__(
        self,
        data_dir: Path,
        filelist: Path,
        text_data: pd.DataFrame,
        description_prob: float,
        num_frames: int,
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

        self.text_data = text_data
        self.description_prob = description_prob

        self.use_audio_type_token = use_audio_type_token
        self.audio_type_token_dropout = audio_type_token_dropout
        self.n_text_views = n_text_views
        self.n_audio_views = n_audio_views
        self.audio_noise_std = audio_noise_std
        self.audio_gain_range = audio_gain_range

    def __len__(self):
        return len(self.filelist)

    @staticmethod
    def get_msd_track_id(path: Path) -> str:
        return path.stem

    def __getitem__(self, idx):
        max_attempts = 10
        attempt = 0
        while attempt < max_attempts:
            try:
                path = Path(self.filelist[idx])
                file_path = self.data_dir / path
                track_id = self.get_msd_track_id(path)

                audio_views = []
                for _ in range(self.n_audio_views):
                    audio = self.load_audio(file_path, frame_offset=self.frame_offset)
                    assert not torch.isnan(audio).any()
                    audio = waveform_augment(
                        audio, self.audio_noise_std, self.audio_gain_range
                    )
                    audio_views.append(audio)

                text_views = [
                    self.get_text_metadata(track_id) for _ in range(self.n_text_views)
                ]
                m = None
                return [audio_views, text_views, m]
            except Exception:
                attempt += 1
                idx = random.randint(0, len(self.filelist) - 1)

        return [[None] * self.n_audio_views, [None] * self.n_text_views, None]

    def get_text_metadata(self, track_id: str) -> str | None:
        """Text preprocessing"""

        description = self.text_data.loc[track_id, "pseudo_caption"]
        description = description.strip()

        tags = self.text_data.loc[track_id, "tag_list"]
        tags = ", ".join(tags)
        tags = tags.strip()

        data = [e for e in [description, tags] if e != ""]

        if len(data) == 0:
            # no text available
            raise ValueError("No sentences available")

        elif len(data) == 1:
            # only one of description or tags available
            text = data[0]

        else:
            # sample between description and tags based on description_prob
            if random.random() < self.description_prob:
                text = description
            else:
                text = tags

        # Prepend audio type token with dropout
        if (
            self.use_audio_type_token
            and random.random() >= self.audio_type_token_dropout
        ):
            text = f"[MUSIC] {text}"

        return text


@gin.configurable
class MSDTextAudioDataModule(L.LightningDataModule):
    """AudioDataModule for the Discogs dataset."""

    def __init__(
        self,
        batch_size: int,
        data_dir: Path,
        filelist_train: Path,
        filelist_val: Path,
        num_workers: int,
        description_prob: float,
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

        self.num_workers = num_workers

        self.description_prob = description_prob
        self.n_text_views = n_text_views
        self.n_audio_views = n_audio_views
        self.audio_noise_std = audio_noise_std
        self.audio_gain_range = audio_gain_range

        self.is_rank_0 = dist.get_rank() == 0 if dist.is_initialized() else 0

        dataset = load_dataset("seungheondoh/enrich-msd", split="train")
        self.text_data = dataset.to_pandas().set_index("track_id")

    def setup(self, stage: str):
        self.dataset_train = MSDTextAudioDataset(
            self.data_dir,
            filelist=self.filelist_train,
            text_data=self.text_data,
            description_prob=self.description_prob,
            n_text_views=self.n_text_views,
            n_audio_views=self.n_audio_views,
            audio_noise_std=self.audio_noise_std,
            audio_gain_range=self.audio_gain_range,
        )
        self.dataset_val = MSDTextAudioDataset(
            self.data_dir,
            filelist=self.filelist_val,
            text_data=self.text_data,
            description_prob=self.description_prob,
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
