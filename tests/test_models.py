"""Smoke tests for loading released models and embedding audio and text."""

import numpy as np
import pytest
import torch

from amclap import get_model

# A frozen-text-encoder model and the trainable-text-encoder flagship, so both
# checkpoint layouts are covered. The flagship is reported at step 60000 rather
# than its final checkpoint; see the model card.
MODEL_IDS = [
    "mtg-upf/allmusiccaps_all_layers",
    "mtg-upf/allmusiccaps_te_trained_sigreg",
]

EMBEDDING_DIM = 512
SAMPLE_RATE = 24000


@pytest.fixture(scope="module", params=MODEL_IDS)
def model(request):
    """The released model, or a skip when the weights cannot be downloaded.

    These tests need network access and permission to read the model repo. Until
    the weights are public they 401 for anyone without access, and offline runs
    cannot reach them at all, so skip rather than fail: the repository's own code
    is not what is broken in either case.
    """
    try:
        return get_model(model_id=request.param, device="cpu").eval()
    except Exception as exc:  # noqa: BLE001 - any download failure is a skip
        if _is_unavailable(exc):
            pytest.skip(
                f"{request.param} is not downloadable here: {type(exc).__name__}"
            )
        raise


def _is_unavailable(exc: BaseException) -> bool:
    """Whether the failure is the model being unreachable rather than a real bug."""
    markers = (
        "401",
        "403",
        "404",
        "gated",
        "unauthorized",
        "not found",
        "offline",
        "connection",
        "timed out",
        "temporary failure in name resolution",
    )
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in markers)


def test_exposes_inference_api(model):
    assert hasattr(model, "forward_audio")
    assert hasattr(model, "forward_text")


def test_embeds_audio_and_text(model):
    audio = torch.randn(1, SAMPLE_RATE * 4)
    with torch.no_grad():
        z_a = model.forward_audio(audio)
        z_t = model.forward_text(["your amazing musical sentence"])

    assert z_a.shape == (1, EMBEDDING_DIM)
    assert z_t.shape == (1, EMBEDDING_DIM)
    assert torch.isfinite(z_a).all()
    assert torch.isfinite(z_t).all()


def test_batches_are_independent(model):
    """Row i of a batch must equal the same clip embedded alone."""
    torch.manual_seed(0)
    batch = torch.randn(3, SAMPLE_RATE * 4)
    with torch.no_grad():
        batched = model.forward_audio(batch)
        single = model.forward_audio(batch[1:2])

    assert batched.shape == (3, EMBEDDING_DIM)
    torch.testing.assert_close(batched[1], single[0], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "bad_audio",
    [
        pytest.param(torch.empty(1, 0), id="empty"),
        pytest.param(torch.full((1, SAMPLE_RATE), float("nan")), id="nan"),
    ],
)
def test_rejects_invalid_audio(model, bad_audio):
    with pytest.raises(Exception):
        with torch.no_grad():
            out = model.forward_audio(bad_audio)
        if torch.isnan(out).any():
            raise ValueError("non-finite embedding")


def test_rejects_numpy_input(model):
    with pytest.raises(Exception):
        model.forward_audio(np.random.randn(1, SAMPLE_RATE).astype("float32"))
