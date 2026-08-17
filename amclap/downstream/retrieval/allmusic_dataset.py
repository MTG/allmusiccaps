import os
import random
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm


class AllMusic(Dataset):
    def __init__(
        self,
        data_dir,
        split,
        caption_type,
        audio_loader="ffmpeg",
        sr=22050,
        duration=10,
        audio_enc=".npy",
        n_segments=48,  # number of audio segments for the frozen appraoch
    ):
        self.dataset_name = "all_music"
        self.data_dir = Path(data_dir)
        self.split = split
        self.audio_loader = audio_loader
        self.audio_enc = audio_enc
        self.caption_type = caption_type
        self.sr = sr

        # filelist = "/data/shared/<user>/data/discotube/metadata/mmap_ids_val"
        filelist = "/mnt/projects/mtg_text_audio/audio_embs/omar-rq-multifeature-25hz-fsq/discotube/ids_val"
        # text_file = "/data/shared/<user>/data/discotube/metadata/Qwen_Qwen2.5-32B__chatgpt_v2__t0.5__1.1.jsonl"
        text_file = (
            "/path/to/discotube/metadata/Qwen_Qwen2.5-32B__chatgpt_v2__t0.5__1.1.jsonl"
        )

        with open(filelist, "r") as f:
            self.fl = [line.rstrip() for line in f.readlines()][:2000]

        ids = [Path(i).stem for i in self.fl]
        ids_set = set(ids)

        self.text = dict()
        ids_list = []
        captions_list = []
        with open(text_file, "r") as f:
            for line in tqdm(f, desc="Loading text data"):
                entry = json.loads(line)
                for k, v in entry.items():
                    if k in ids_set:
                        try:
                            sentences = []
                            for s in v.values():
                                sentences.extend(s)
                            self.text[k] = " ".join(sentences)
                            ids_list.append(k)
                            captions_list.append(self.text[k])
                        except Exception as e:
                            print(f"Error processing entry {k}: {e}")

        print(f"Loaded {len(self.text)} text entries out of {len(ids)} ids")

        self.id_col = "ytid"
        self.tag_col = "aspect_list"
        self.caption_col = "caption_ground_truth"

        data = {self.id_col: ids_list, self.caption_col: captions_list}

        self.annotations = pd.DataFrame(data)
        print(self.annotations.head())

    @staticmethod
    def load_audio(path, sr=24000, mono=True):
        import torchaudio
        from torchaudio.transforms import Resample

        audio, orig_sr = torchaudio.load(path)

        # Downmix to mono if necessary
        if audio.shape[0] > 1 and mono:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # Resample if necessary
        if orig_sr != sr:
            audio = audio.float()
            interm = 16000
            audio = Resample(orig_freq=orig_sr, new_freq=interm)(audio)
            resample = Resample(orig_freq=interm, new_freq=sr)
            audio = resample(audio)
            audio = audio.half()

        return audio

    def load_audio_mmap(self, file_path: Path):
        mmap_sr = 16000
        mmap = np.memmap(file_path, dtype="float16", mode="r")
        audio = np.array(mmap).astype(np.float32)
        if self.sr != mmap_sr:
            audio = torchaudio.functional.resample(
                torch.from_numpy(audio), orig_freq=mmap_sr, new_freq=self.sr
            ).numpy()
        audio = torch.from_numpy(audio).float()
        return audio

    def _load_audio(self, fname):
        if self.audio_enc == ".npy":  # for fast audio loading
            audio_path = os.path.join(self.data_dir, "npy", fname + self.audio_enc)
            audio = np.load(audio_path, mmap_mode="r")
            audio = np.asarray(audio).astype(np.float32) / 32768.0
            audio = torch.from_numpy(audio).float()

        elif self.audio_enc == ".mmap":  # for fast audio loading
            audio_path = self.data_dir / fname[:2] / (fname + self.audio_enc)
            audio = self.load_audio_mmap(audio_path)
            audio = torch.from_numpy(audio).float()

        elif self.audio_enc == ".pt":  # for fast audio loading
            audio_path = self.data_dir / fname[:2] / (fname + self.audio_enc)
            audio = torch.load(audio_path)

        else:
            audio_path = os.path.join(self.data_dir, "audio", fname + self.audio_enc)
            audio = self.load_audio(
                path=audio_path,
                sr=self.sr,
                mono=True,
            )
        if len(audio.shape) == 2:
            audio = audio.squeeze(0)

        audio_len = audio.shape[0]
        self.n_samples = 48
        if audio_len < self.n_samples:
            pass
            # pad = torch.zeros(self.n_samples)
            # pad[:audio_len] = audio
            # audio = pad
        elif audio_len > self.n_samples:
            # get self.n_samples from the center
            start = (audio.shape[0] - self.n_samples) // 2
            audio = audio[start : start + self.n_samples]

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
        return self._load_audio(ytid)

    def __getitem__(self, index):
        path = self.fl[index]
        id = Path(path).stem

        text = self.load_text(id)
        audio_tensor = self._load_audio(path)
        return id, text, audio_tensor

    def __len__(self):
        return len(self.fl)
