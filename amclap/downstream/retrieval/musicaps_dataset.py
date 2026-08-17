import os
import random

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


class MusicCaps(Dataset):
    def __init__(
        self,
        data_dir,
        split,
        caption_type,
        audio_loader="ffmpeg",
        sr=22050,
        duration=10,
        audio_enc=".npy",
    ):
        self.dataset_name = "music_caps"
        # self.data_dir = os.path.join(data_dir, "music_caps")
        self.data_dir = data_dir
        self.split = split
        self.audio_loader = audio_loader
        self.audio_enc = audio_enc
        self.caption_type = caption_type
        self.sr = sr
        self.n_samples = int(sr * duration)
        self.dataset = load_dataset("seungheondoh/LP-MusicCaps-MC")
        self.get_split()
        self.get_columns()

    def get_columns(self):
        self.id_col = "ytid"
        self.tag_col = "aspect_list"
        self.caption_col = "caption_ground_truth"
        # metadata
        self.title_col = None
        self.artist_col = None
        self.album_col = None
        self.year_col = None
        # mid-level data
        self.key_col = None
        self.tempo_col = None
        self.chord_col = None

    def get_split(self):
        self.fl = self.dataset[self.split]
        self.annotations = pd.DataFrame(self.fl)

        # locate where is_balanced_subset is True
        # self.annotations = self.annotations[self.annotations["is_balanced_subset"]]

        self.ytid_to_idx = {
            ytid: idx for idx, ytid in enumerate(self.annotations["ytid"])
        }

    @staticmethod
    def load_audio(path, sr=24000, mono=True):
        # import torchaudio
        # from torchaudio.transforms import Resample
        #
        # audio, orig_sr = torchaudio.load(path)
        #
        # # Downmix to mono if necessary
        # if audio.shape[0] > 1 and mono:
        #     audio = torch.mean(audio, dim=0, keepdim=True)
        #
        # # Resample if necessary
        # if orig_sr != sr:
        #     audio = audio.float()
        #     interm = 16000
        #     audio = Resample(orig_freq=orig_sr, new_freq=interm)(audio)
        #     audio = Resample(orig_freq=interm, new_freq=sr)(audio)
        #     audio = audio.half()

        try:
            audio = MonoLoader(filename=str(path), sampleRate=sr, resampleQuality=4)()
        except Exception as e:
            print(f"Error loading {path}: {e}")
            audio = np.zeros(sr * 10)  # 10 seconds of silence
        audio = torch.from_numpy(audio).float()
        audio.unsqueeze_(0)

        return audio

    def _load_audio(self, fname):
        if self.audio_enc == ".npy":  # for fast audio loading
            audio_path = os.path.join(self.data_dir, "npy", fname + self.audio_enc)
            audio = np.load(audio_path, mmap_mode="r")
            audio = int16_to_float32(audio)
        else:
            audio_path = os.path.join(self.data_dir, "audio", fname + self.audio_enc)
            audio = self.load_audio(
                path=audio_path,
                sr=self.sr,
                mono=True,
            )

        # random_idx = random.randint(0, audio.shape[-1] - self.n_samples)
        # audio_tensor = torch.from_numpy(
        #     np.array(audio[random_idx : random_idx + self.n_samples]).astype("float32")
        # )
        # audio = int16_to_float32(float32_to_int16(audio))

        return audio

    def load_tag(self, item):
        tag_list = item["aspect_list"]
        k = random.choice(range(1, len(tag_list) + 1))
        sampled_tag_list = random.sample(tag_list, k)
        text = ", ".join(sampled_tag_list)
        return text

    def load_caption(self, item):
        return item["caption_ground_truth"]

    def load_text(self, item):
        text_pool = []
        if "tag" in self.caption_type:
            text_pool.append(self.load_tag(item))
        if "caption" in self.caption_type:
            text_pool.append(self.load_caption(item))
        random.shuffle(text_pool)
        k = random.choice(range(1, len(text_pool) + 1))
        sampled_text_pool = random.sample(text_pool, k)
        text = ". ".join(sampled_text_pool)
        return text

    def get_audio(self, ytid):
        # idx = self.ytid_to_idx[ytid]
        # item = self.annotations.iloc[idx]

        # item_parsed = item["fname"].split("-")
        # start_time = item_parsed[-2][1:]

        # filename = f"{ytid}_{start_time}"
        return self._load_audio(ytid)

    def __getitem__(self, index):
        item = self.fl[index]
        fname = item["fname"]
        text = self.load_text(item)
        audio_tensor = self._load_audio(fname)
        return fname, text, audio_tensor

    def __len__(self):
        return len(self.fl)
