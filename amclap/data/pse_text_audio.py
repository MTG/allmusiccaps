import random
import re
import warnings
from pathlib import Path
from typing import Union

import gin
import lightning.pytorch as L
import torch
from torch.utils.data import DataLoader

from ..clap_modules.audio_augmentations import waveform_augment

from .data_utils import AudioDataset, collate_with_skip


PSE_CATEGORY_MAP = {
    # MUSIC
    "Musical": "[MUSIC]",
    # INSTRUMENT_SAMPLES
    "Bells": "[INSTRUMENT_SAMPLES]",
    "Horns": "[INSTRUMENT_SAMPLES]",
    "Whistles": "[INSTRUMENT_SAMPLES]",
    # SPEECH / VOICE
    "Voices": "[SPEECH]",
    "Human": "[SPEECH]",
    # SOUNDSCAPES - Natural/ambient sounds
    "Air": "[SOUNDSCAPES]",
    "Ambience": "[SOUNDSCAPES]",
    "Animals": "[SOUNDSCAPES]",
    "Birds": "[SOUNDSCAPES]",
    "Creatures": "[SOUNDSCAPES]",
    "Crowds": "[SOUNDSCAPES]",
    "Dirt & Sand": "[SOUNDSCAPES]",
    "Fire": "[SOUNDSCAPES]",
    "Geothermal": "[SOUNDSCAPES]",
    "Ice": "[SOUNDSCAPES]",
    "Liquid & Mud": "[SOUNDSCAPES]",
    "Natural Disaster": "[SOUNDSCAPES]",
    "Rain": "[SOUNDSCAPES]",
    "Snow": "[SOUNDSCAPES]",
    "Vegetation": "[SOUNDSCAPES]",
    "Water": "[SOUNDSCAPES]",
    "Weather": "[SOUNDSCAPES]",
    "Wind": "[SOUNDSCAPES]",
    "Wings": "[SOUNDSCAPES]",
    # All remaining categories default to [SOUND_EFFECTS]
}


@gin.configurable
class PSETextAudioDataset(AudioDataset):
    """Text-audio dataset for PSE.
    It expects a jsonl file with description and tags for each freesound id.
    """

    def __init__(
        self,
        data_dir: Path,
        filelist: Path,
        num_frames: int,
        frame_offset: Union[int, str] = "random",
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

                audio_views = []
                for _ in range(self.n_audio_views):
                    audio = self.load_audio(file_path, frame_offset=self.frame_offset)
                    assert not torch.isnan(audio).any()
                    audio = waveform_augment(
                        audio, self.audio_noise_std, self.audio_gain_range
                    )
                    audio_views.append(audio)

                text_views = [
                    self.get_text_metadata(path) for _ in range(self.n_text_views)
                ]
                m = None
                return [audio_views, text_views, m]
            except Exception:
                attempt += 1
                idx = random.randint(0, len(self.filelist) - 1)
        return [[None] * self.n_audio_views, [None] * self.n_text_views, None]

    def get_text_metadata(self, path: Path) -> str | None:
        """Get text metadta from  a given PSE path"""

        # Get the category and subcategory from the path
        cat_level_1 = path.parent.parent.name
        cat_level_2 = path.parent.name
        tags = f"{cat_level_1}, {cat_level_2}"

        # All PSE sounds follow the name strucuture <class-id>_description <number>_PSE_<other_info>
        # This discription is short but more detailed than the taxonomy on the folder strucuture.
        stem = path.stem
        description = stem.split("_")[1]
        description = " ".join(description.split(" ")[:-1]).strip()

        # Sometimes <number> contains several numbers separated by spaces
        # Remove numbers that stand alone
        description = re.sub(r"\b\d+\b", "", description)

        data = [tags, description]
        random.shuffle(data)

        tags = ", ".join(data)

        tags = tags.strip()

        text = self.tags_header + tags

        # Prepend audio type token with dropout
        if (
            self.use_audio_type_token
            and random.random() >= self.audio_type_token_dropout
        ):
            token = PSE_CATEGORY_MAP.get(cat_level_1, "[SOUND_EFFECTS]")
            text = f"{token} {text}"

        return text


@gin.configurable
class PSETextAudioDataModule(L.LightningDataModule):
    """AudioDataModule for the PSE dataset."""

    def __init__(
        self,
        batch_size: int,
        data_dir: Path,
        filelist_train: Path,
        filelist_val: Path,
        num_workers: int,
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
        self.n_text_views = n_text_views
        self.n_audio_views = n_audio_views
        self.audio_noise_std = audio_noise_std
        self.audio_gain_range = audio_gain_range

    def setup(self, stage: str):
        self.dataset_train = PSETextAudioDataset(
            self.data_dir,
            filelist=self.filelist_train,
            n_text_views=self.n_text_views,
            n_audio_views=self.n_audio_views,
            audio_noise_std=self.audio_noise_std,
            audio_gain_range=self.audio_gain_range,
        )
        self.dataset_val = PSETextAudioDataset(
            self.data_dir,
            filelist=self.filelist_val,
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
