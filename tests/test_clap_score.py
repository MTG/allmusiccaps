"""Unit tests for the CLAP-score evaluation pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
MUSICEVAL_DIR = REPO_ROOT / "data" / "musiceval" / "MusicEval-full"


def test_cosine_in_unit_range():
    rng = np.random.default_rng(0)
    a = torch.from_numpy(rng.standard_normal((1, 128)).astype("float32"))
    b = torch.from_numpy(rng.standard_normal((1, 128)).astype("float32"))
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    s = (a @ b.T).item()
    assert -1.0 - 1e-6 <= s <= 1.0 + 1e-6


def test_segment_aggregation_is_mean_invariant_to_shuffle():
    torch.manual_seed(0)
    x = torch.randn(5, 128)
    perm = torch.randperm(5)
    assert torch.allclose(x.mean(0), x[perm].mean(0), atol=1e-6)


@pytest.mark.skipif(
    not MUSICEVAL_DIR.is_dir(),
    reason="MusicEval corpus not present at data/musiceval/MusicEval-full/",
)
def test_musiceval_loader_shapes():
    from amclap.downstream.clap_score.musiceval_dataset import MusicEval

    ds = MusicEval(data_dir=MUSICEVAL_DIR, split="total", sr=24000, duration=10.0)
    assert len(ds) > 0
    fname, prompt, audio = ds[0]
    assert fname.endswith(".wav")
    assert isinstance(prompt, str) and len(prompt) > 0
    assert audio.ndim == 2
    assert audio.shape[-1] == 24000 * 10

    row = ds.row(0)
    assert row["system"].startswith("S")
    assert row["prompt_id"].startswith("P")
    assert 1.0 <= row["mos_overall"] <= 5.0
    assert 1.0 <= row["mos_alignment"] <= 5.0
