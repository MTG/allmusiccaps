import os

import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from torch.utils.data import Dataset
from essentia.standard import MonoLoader


def int16_to_float32(x):
    return (x / 32767.0).astype(torch.float32)


def float32_to_int16(x):
    x = ((x - x.min()) / (x.max() - x.min())) * (2) - 1.0  # peak -1, 1 normalize
    return (x * 32767.0).astype(torch.int16)


class SongDescriber(Dataset):
    def __init__(
        self,
        data_dir,
        split,
        caption_type,
        audio_loader="ffmpeg",
        sr=22050,
        duration=120,
        audio_enc=".mp3",
    ):
        self.dataset_name = "song_describer"
        # self.data_dir = os.path.join(data_dir, "song_describer")
        self.data_dir = data_dir
        self.split = split
        self.audio_loader = audio_loader
        self.audio_enc = audio_enc
        self.caption_type = caption_type
        self.sr = sr
        self.n_samples = int(sr * duration)
        # self.dataset = load_dataset("music-temp/song-describer-dataset")
        # self.dataset = load_dataset("renumics/song-describer-dataset")
        self.dataset = load_dataset("seungheondoh/eval-song_describer")
        self.get_split()
        self.get_columns()

    def get_split(self):
        dataset = self.dataset["original"]
        df_anno = pd.DataFrame(dataset)
        self.annotations = df_anno[df_anno["is_valid_subset"]]
        # self.annotations = df_anno

        self.tid_to_path = {
            track_id: path
            for track_id, path in zip(
                self.annotations["track_id"].to_list(),
                self.annotations["path"].to_list(),
            )
        }
        self.tid_to_idx = {
            tid: idx for idx, tid in enumerate(self.annotations["track_id"])
        }

    def get_columns(self):
        self.id_col = "track_id"
        self.tag_col = "aspect_list"
        self.caption_col = "caption"
        # metadata
        self.title_col = None
        self.artist_col = "artist_id"
        self.album_col = "album_id"
        self.year_col = None
        # mid-level data
        self.key_col = None
        self.tempo_col = None
        self.chord_col = None

    @staticmethod
    def load_audio(path, sr=24000, mono=True):
        # import torchaudio
        # from torchaudio.transforms import Resample

        # audio, orig_sr = torchaudio.load(path)

        # Downmix to mono if necessary
        # if audio.shape[0] > 1 and mono:
        #     audio = torch.mean(audio, dim=0, keepdim=True)

        # Resample if necessary
        # if orig_sr != sr:
        #     audio = audio.float()
        #     audio = Resample(orig_freq=orig_sr, new_freq=sr)(audio)

        try:
            audio = MonoLoader(filename=str(path), sampleRate=sr, resampleQuality=4)()
        except Exception as e:
            print(f"Error loading {path}: {e}")
            audio = np.zeros(sr * 30)  # 10 seconds of silence
        audio = torch.from_numpy(audio).float()
        audio.unsqueeze_(0)

        return audio

    def _load_audio(self, fname):
        if self.audio_enc == ".npy":  # for fast audio loading
            audio_path = os.path.join(self.data_dir, "npy", fname + self.audio_enc)
            audio = np.load(audio_path, mmap_mode="r")
            audio = int16_to_float32(audio)
        else:
            audio_path = os.path.join(
                self.data_dir, "audio", str(fname) + self.audio_enc
            )
            audio = self.load_audio(
                path=audio_path,
                sr=self.sr,
                mono=True,
            )

        # to mono
        if len(audio.shape) == 2:
            audio.squeeze_(0)

        # pad if needed
        if audio.shape[-1] < self.n_samples:
            pad = torch.zeros(self.n_samples)
            pad[: audio.shape[-1]] = audio
            audio = pad

            audio.unsqueeze_(0)  # for batch dimension

        elif audio.shape[-1] > self.n_samples:
            # wrap into batches
            n_chunks = int(np.round(audio.shape[-1] / self.n_samples))
            n_samples = n_chunks * self.n_samples

            # trim
            if n_samples < audio.shape[-1]:
                audio = audio[: n_chunks * self.n_samples]
            # or pad
            else:
                pad = torch.zeros(n_samples)
                pad[: audio.shape[-1]] = audio
                audio = pad

            audio = audio.view(n_chunks, self.n_samples)
        else:
            # if equal, just add batch dimension
            audio.unsqueeze_(0)  # for batch dimension

        return audio

    def get_audio(self, tid):
        idx = self.tid_to_idx[tid]
        item = self.annotations.iloc[idx]["track_id"]
        item = str(item)
        rel_path = f"{item[-2:]}/{item}"
        return self._load_audio(rel_path)

    def __getitem__(self, index):
        item = self.annotations.iloc[index]
        fname = item["track_id"]
        text = item["caption"]
        audio_tensor = self._load_audio(fname)
        return fname, text, audio_tensor

    def __len__(self):
        return len(self.annotations)
