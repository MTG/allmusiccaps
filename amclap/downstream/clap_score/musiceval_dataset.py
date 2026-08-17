"""MusicEval dataset loader.

Expects the extracted `MusicEval-full/` tree (BAAI/MusicEval, ICASSP 2025):

    MusicEval-full/
        wav/audiomos2025-track1-S###_P###.wav
        sets/total_mos_list.txt      # filename,mos_overall,mos_alignment (no header)
        prompt_info.txt              # id\ttext (header row)
        system_mos/system_mos_*.csv  # per-system aggregate MOS (not used here)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from essentia.standard import MonoLoader
from torch.utils.data import Dataset

FNAME_RE = re.compile(r"audiomos2025-track1-(S\d+)_(P\d+)\.wav$")


class MusicEval(Dataset):
    """(prompt, generated_audio, human_MOS) triples for scoring-model evaluation.

    Mirrors the loader shape of `retrieval.musicaps_dataset.MusicCaps`. Each item
    returns (fname, prompt, audio_tensor) plus the two MOS columns and the
    system id as attributes accessible via `df_row(index)`.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "total",
        sr: int = 24000,
        duration: float = 10.0,
    ) -> None:
        self.dataset_name = "musiceval"
        self.data_dir = Path(data_dir)
        self.sr = sr
        self.duration = float(duration)

        mos_file = self.data_dir / "sets" / f"{split}_mos_list.txt"
        if not mos_file.is_file():
            raise FileNotFoundError(f"MOS file not found: {mos_file}")
        self.mos_df = pd.read_csv(
            mos_file,
            header=None,
            names=["fname", "mos_overall", "mos_alignment"],
        )

        prompt_file = self.data_dir / "prompt_info.txt"
        prompts = pd.read_csv(prompt_file, sep="\t")
        self.prompt_map = dict(zip(prompts["id"], prompts["text"]))

        rows = []
        for _, row in self.mos_df.iterrows():
            m = FNAME_RE.search(str(row["fname"]))
            if not m:
                continue
            system_id, prompt_id = m.group(1), m.group(2)
            prompt = self.prompt_map.get(prompt_id)
            if prompt is None:
                continue
            rows.append(
                {
                    "fname": row["fname"],
                    "system": system_id,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "mos_overall": float(row["mos_overall"]),
                    "mos_alignment": float(row["mos_alignment"]),
                }
            )
        self.df = pd.DataFrame(rows)
        if len(self.df) == 0:
            raise RuntimeError(
                f"No (audio, prompt, mos) triples loaded from {data_dir}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def _load_audio(self, fname: str) -> torch.Tensor:
        path = self.data_dir / "wav" / fname
        try:
            audio = MonoLoader(
                filename=str(path), sampleRate=self.sr, resampleQuality=4
            )()
        except Exception as e:
            print(f"Error loading {path}: {e}")
            audio = np.zeros(int(self.sr * self.duration), dtype=np.float32)

        audio = torch.from_numpy(audio).float()
        n_seg = int(round(self.sr * self.duration))
        if n_seg > 0:
            length = audio.shape[-1]
            if length < n_seg:
                pad = torch.zeros(n_seg - length, dtype=audio.dtype)
                audio = torch.cat([audio, pad], dim=-1)
                length = audio.shape[-1]
            n_full = length // n_seg
            audio = audio[: n_full * n_seg].reshape(n_full, n_seg)
        else:
            audio = audio.unsqueeze(0)
        return audio

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        audio = self._load_audio(row["fname"])
        return row["fname"], row["prompt"], audio

    def row(self, index: int) -> dict:
        return self.df.iloc[index].to_dict()
