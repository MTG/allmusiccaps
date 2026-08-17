import json
import random
from pathlib import Path
from typing import Tuple, Union

import gin
import lightning.pytorch as L
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_utils import AudioDataset, collate_with_skip
from ..clap_modules.audio_augmentations import waveform_augment


@gin.configurable
class DiscotubeTextAudioCleanDataset(AudioDataset):
    """Text-audio dataset for discotube.
    It expects a jsonl file with a list of sentences for each youtube id.
    `max_sentences` controls the maximum number of sentences to sample from each release.
    The number of sentences is randomly sampled from 0 to `max_sentences`.
    """

    def __init__(
        self,
        data_dir: Path,
        filelist: Path,
        text_data: dict,
        max_sentences: int,
        num_frames: int,
        frame_offset: Union[int, str] = "random",
        allmusic_meta: dict | None = None,
        allmusic_keys: list[str] | None = None,
        input_audio_is_embedding: bool = False,
        folder_structure_type: str = "standard",
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
        )  # type: ignore

        self.text_data = text_data
        self.max_sentences = max_sentences

        self.allmusic_meta = allmusic_meta
        self.allmusic_keys = allmusic_keys

        self.input_audio_is_embedding = input_audio_is_embedding

        self.folder_structure_type = folder_structure_type

        self.use_audio_type_token = use_audio_type_token
        self.audio_type_token_dropout = audio_type_token_dropout
        self.n_text_views = n_text_views
        self.n_audio_views = n_audio_views
        self.audio_noise_std = audio_noise_std
        self.audio_gain_range = audio_gain_range

    def __len__(self):
        return len(self.filelist)

    @staticmethod
    def get_audio_path(youtube_id: str) -> Path:
        return Path(youtube_id[:2], youtube_id).with_suffix(".mp4")

    def load_embedding(self, file_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        rep = torch.load(file_path)

        # rep is [M, D] and we want N frames of D-dim features
        if rep.size(0) < self.num_frames:
            rep_padded = torch.zeros(self.num_frames, rep.size(1))
            rep_padded[: rep.size(0), :] = rep

            mask = [True] * rep.size(0) + [False] * (self.num_frames - rep.size(0))

            return rep_padded, torch.tensor(mask, dtype=torch.bool)

        if self.frame_offset == "random":
            start_idx = random.randint(0, rep.size(0) - self.num_frames)
        elif isinstance(self.frame_offset, int):
            start_idx = self.frame_offset
        else:
            raise ValueError("Invalid frame_offset")

        rep = rep[start_idx : start_idx + self.num_frames, :]

        # compute mask for the attention
        mask = torch.ones(self.num_frames, dtype=torch.bool)

        return rep, mask

    @staticmethod
    def get_bsc_path(base: Path, rel_path: Path) -> Path:
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

    def __getitem__(self, idx):
        max_attempts = 10
        attempt = 0
        while attempt < max_attempts:
            try:
                file_path = self.filelist[idx]
                file_path = Path(file_path)
                id_yt = file_path.stem

                if self.folder_structure_type == "bsc":
                    file_path = self.get_bsc_path(self.data_dir, file_path)
                elif self.folder_structure_type == "standard":
                    file_path = self.data_dir / file_path
                else:
                    raise ValueError(
                        f"Invalid folder_structure_type {self.folder_structure_type}"
                    )

                # load audio views
                if self.input_audio_is_embedding:
                    x, m = self.load_embedding(file_path)
                    audio_views = [x] * self.n_audio_views
                else:
                    audio_views = []
                    for _ in range(self.n_audio_views):
                        x = self.load_audio(file_path, frame_offset=self.frame_offset)
                        x = waveform_augment(
                            x,
                            noise_std=self.audio_noise_std,
                            gain_range=self.audio_gain_range,
                        )
                        audio_views.append(x)
                    # NOTE: From raw audio we normally use segments that are much
                    # smaller than num_frames so we don't need a mask
                    m = None

                # generate text views
                text_views = [
                    self.get_text_metadata(id_yt) for _ in range(self.n_text_views)
                ]

                return [audio_views, text_views, m]
            except Exception:
                attempt += 1
                # Try a different random index
                idx = random.randint(0, len(self.filelist) - 1)
        # If all attempts fail, return error sentinel
        return [
            [None] * self.n_audio_views,
            [None] * self.n_text_views,
            None,
        ]

    def get_text_metadata(self, id_yt: str) -> str | None:
        """Text preprocessing"""

        release_sentences = self.text_data[id_yt]
        release_sentence_keys = list(release_sentences.keys())

        # random sample senteces from one of the releaes
        key = random.choice(release_sentence_keys)
        sentences = release_sentences[key]

        # get a random number of  sentences up to max_sentences
        max_sentences = min(self.max_sentences, len(sentences))
        n_sentences = random.choice(range(max_sentences)) + 1

        # sample
        sentences = random.sample(sentences, n_sentences)

        sentences = [s.strip() for s in sentences if s.strip() != ""]
        text = "\n".join(sentences)

        if self.allmusic_keys is not None and self.allmusic_meta is not None:
            track_meta = self.allmusic_meta.get(id_yt, {})
            tags = []
            for k in self.allmusic_keys:
                tags.extend(track_meta.get(k, []))

            if text != "":
                text += "\n"

            text += ", ".join(tags)

        if text == "":
            raise ValueError("DiscoTube: No sentences available")

        # Prepend audio type token with dropout
        if (
            self.use_audio_type_token
            and random.random() >= self.audio_type_token_dropout
        ):
            text = f"[MUSIC] {text}"

        return text


@gin.configurable
class DiscotubeTextAudioCleanDataModule(L.LightningDataModule):
    """AudioDataModule for the Discogs dataset."""

    def __init__(
        self,
        batch_size: int,
        data_dir: Path,
        filelist_train: Path,
        filelist_val: Path,
        text_file: Path,
        num_workers: int,
        max_sentences: int,
        allmusic_meta: Path | None = None,
        allmusic_keys: list[str] | None = None,
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
        self.max_sentences = max_sentences
        self.n_text_views = n_text_views
        self.n_audio_views = n_audio_views
        self.audio_noise_std = audio_noise_std
        self.audio_gain_range = audio_gain_range

        self.num_workers = num_workers

        self.is_rank_0 = dist.get_rank() == 0 if dist.is_initialized() else 0

        self.text_data = dict()
        with open(self.text_file, "r") as f:
            for line in tqdm(f, desc="Loading text data", disable=not self.is_rank_0):
                entry = json.loads(line)
                for k, v in entry.items():
                    if k not in self.text_data:
                        self.text_data[k] = v

        self.allmusic_meta = None
        if allmusic_meta is not None:
            with open(allmusic_meta, "r") as f:
                self.allmusic_meta = json.load(f)
        self.allmusic_keys = allmusic_keys

    def setup(self, stage: str):
        self.dataset_train = DiscotubeTextAudioCleanDataset(
            self.data_dir,
            filelist=self.filelist_train,
            text_data=self.text_data,
            max_sentences=self.max_sentences,
            allmusic_meta=self.allmusic_meta,
            allmusic_keys=self.allmusic_keys,
            n_text_views=self.n_text_views,
            n_audio_views=self.n_audio_views,
            audio_noise_std=self.audio_noise_std,
            audio_gain_range=self.audio_gain_range,
        )
        self.dataset_val = DiscotubeTextAudioCleanDataset(
            self.data_dir,
            filelist=self.filelist_val,
            text_data=self.text_data,
            max_sentences=self.max_sentences,
            allmusic_meta=self.allmusic_meta,
            allmusic_keys=self.allmusic_keys,
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
