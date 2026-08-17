import json
import random
import warnings
from pathlib import Path
from typing import Union

import gin
import lightning.pytorch as L
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..clap_modules.audio_augmentations import waveform_augment

from .data_utils import AudioDataset, collate_with_skip


FREESOUND_CATEGORY_MAP = {
    "Music": "[MUSIC]",
    "Instrument Samples": "[INSTRUMENT_SAMPLES]",
    "Speech": "[SPEECH]",
    "Sound Effects": "[SOUND_EFFECTS]",
    "Soundscapes": "[SOUNDSCAPES]",
}


@gin.configurable
class FreesoundTextAudioDataset(AudioDataset):
    """Text-audio dataset for Freesound.
    It expects a jsonl file with description and tags for each freesound id.
    `description_prob` controls the probability of sampling the description vs the tags.
    """

    def __init__(
        self,
        data_dir: Path,
        filelist: Path,
        text_data: dict,
        description_prob: float,
        num_frames: int,
        frame_offset: Union[int, str] = "random",
        description_heaader: str = "Description: ",
        tags_header: str = "",
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

        self.description_header = description_heaader
        self.tags_header = tags_header

        self.use_audio_type_token = use_audio_type_token
        self.audio_type_token_dropout = audio_type_token_dropout
        self.n_text_views = n_text_views
        self.n_audio_views = n_audio_views
        self.audio_noise_std = audio_noise_std
        self.audio_gain_range = audio_gain_range

    def __len__(self):
        return len(self.filelist)

    @staticmethod
    def get_fsid(path: Path) -> str:
        return path.stem.split("_")[0]

    def __getitem__(self, idx):
        max_attempts = 10
        attempt = 0
        while attempt < max_attempts:
            try:
                path = Path(self.filelist[idx])
                file_path = self.data_dir / path
                fsid = self.get_fsid(path)

                audio_views = []
                for _ in range(self.n_audio_views):
                    audio = self.load_audio(file_path, frame_offset=self.frame_offset)
                    assert not torch.isnan(audio).any()
                    audio = waveform_augment(
                        audio, self.audio_noise_std, self.audio_gain_range
                    )
                    audio_views.append(audio)

                text_views = [
                    self.get_text_metadata(fsid) for _ in range(self.n_text_views)
                ]
                m = None
                return [audio_views, text_views, m]
            except Exception:
                attempt += 1
                idx = random.randint(0, len(self.filelist) - 1)
        return [[None] * self.n_audio_views, [None] * self.n_text_views, None]

    def get_text_metadata(self, fsid: str) -> str | None:
        """Text preprocessing"""

        description = self.text_data[fsid].get("description", "")
        description = description.strip()
        description = self.description_header + description

        tags = self.text_data[fsid].get("tags", "")
        tags = ", ".join(tags)
        tags = tags.strip()

        tags = self.tags_header + tags

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
            category_list = self.text_data[fsid].get("category", [])
            if category_list:
                broad_category = category_list[0]
                token = FREESOUND_CATEGORY_MAP.get(broad_category, "[SOUND_EFFECTS]")
                text = f"{token} {text}"

        return text


@gin.configurable
class FreesoundTextAudioDataModule(L.LightningDataModule):
    """AudioDataModule for the Discogs dataset."""

    def __init__(
        self,
        batch_size: int,
        data_dir: Path,
        filelist_train: Path,
        filelist_val: Path,
        text_file: Path,
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
        self.text_file = Path(text_file)

        self.num_workers = num_workers

        self.description_prob = description_prob
        self.n_text_views = n_text_views
        self.n_audio_views = n_audio_views
        self.audio_noise_std = audio_noise_std
        self.audio_gain_range = audio_gain_range

        self.is_rank_0 = dist.get_rank() == 0 if dist.is_initialized() else 0

        self.text_data = dict()
        with open(self.text_file, "r") as f:
            for line in tqdm(f, desc="Loading text data", disable=not self.is_rank_0):
                entry = json.loads(line)
                for k, v in entry.items():
                    if k not in self.text_data:
                        self.text_data[k] = v

    def setup(self, stage: str):
        self.dataset_train = FreesoundTextAudioDataset(
            self.data_dir,
            filelist=self.filelist_train,
            text_data=self.text_data,
            description_prob=self.description_prob,
            n_text_views=self.n_text_views,
            n_audio_views=self.n_audio_views,
            audio_noise_std=self.audio_noise_std,
            audio_gain_range=self.audio_gain_range,
        )
        self.dataset_val = FreesoundTextAudioDataset(
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
