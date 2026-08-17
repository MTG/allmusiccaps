"""Unit tests for pure metric functions in src/layerwise_metrics.py."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from amclap.layerwise_metrics import (
    LayerWiseMetricsCallback,
    _anisotropy_spectral,
    _compute_knn_indices,
    _effective_rank_and_entropy,
    _gaussianity_ep,
    _ii_return_imbalance,
    _mlid_mom,
    _resolve_num_samples,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gaussian_data():
    torch.manual_seed(0)
    return torch.randn(100, 16)


@pytest.fixture
def rank1_data():
    torch.manual_seed(1)
    v = torch.randn(100)
    w = torch.randn(16)
    return torch.outer(v, w)


@pytest.fixture
def clustered_data():
    torch.manual_seed(2)
    c1 = torch.randn(50, 16) * 0.05 + torch.tensor([10.0] + [0.0] * 15)
    c2 = torch.randn(50, 16) * 0.05 - torch.tensor([10.0] + [0.0] * 15)
    return torch.cat([c1, c2], dim=0)


@pytest.fixture
def identity_knn():
    """Hand-crafted kNN index array where space 1 == space 2 (N=5, maxk=2)."""
    indices = np.array(
        [
            [0, 1, 2],
            [1, 0, 2],
            [2, 3, 1],
            [3, 2, 4],
            [4, 3, 2],
        ],
        dtype=np.int64,
    )
    return indices


# ---------------------------------------------------------------------------
# _mlid_mom
# ---------------------------------------------------------------------------


def test_mlid_mom_returns_positive(gaussian_data):
    mlid, frechet_var = _mlid_mom(gaussian_data, k=10)
    assert mlid > 0
    assert frechet_var >= 0


def test_mlid_mom_types(gaussian_data):
    mlid, frechet_var = _mlid_mom(gaussian_data, k=10)
    assert isinstance(mlid, float)
    assert isinstance(frechet_var, float)


def test_mlid_mom_rank1_lower_than_gaussian(gaussian_data, rank1_data):
    mlid_gauss, _ = _mlid_mom(gaussian_data, k=10)
    mlid_rank1, _ = _mlid_mom(rank1_data, k=10)
    assert mlid_rank1 < mlid_gauss


def test_mlid_mom_small_n_no_crash():
    """k should be clamped to N-1 without crashing."""
    X = torch.randn(5, 8)
    mlid, frechet_var = _mlid_mom(X, k=100)
    assert mlid > 0
    assert frechet_var >= 0


# ---------------------------------------------------------------------------
# _effective_rank_and_entropy
# ---------------------------------------------------------------------------


def test_effective_rank_and_entropy_valid_ranges(gaussian_data):
    eff_rank, norm_entropy = _effective_rank_and_entropy(gaussian_data)
    assert eff_rank >= 1.0
    assert 0.0 <= norm_entropy <= 1.0


def test_effective_rank_and_entropy_types(gaussian_data):
    eff_rank, norm_entropy = _effective_rank_and_entropy(gaussian_data)
    assert isinstance(eff_rank, float)
    assert isinstance(norm_entropy, float)


def test_effective_rank_rank1(rank1_data):
    eff_rank, norm_entropy = _effective_rank_and_entropy(rank1_data)
    assert eff_rank == pytest.approx(1.0, abs=0.5)
    assert norm_entropy == pytest.approx(0.0, abs=0.1)


def test_effective_rank_gaussian_high(gaussian_data):
    D = gaussian_data.shape[1]
    eff_rank, norm_entropy = _effective_rank_and_entropy(gaussian_data)
    # Full-rank isotropic data should have high effective rank relative to D
    assert eff_rank > D * 0.5
    assert norm_entropy > 0.7


# ---------------------------------------------------------------------------
# _anisotropy_spectral
# ---------------------------------------------------------------------------


def test_anisotropy_spectral_geq_one(gaussian_data):
    aniso = _anisotropy_spectral(gaussian_data)
    assert aniso >= 1.0


def test_anisotropy_spectral_type(gaussian_data):
    assert isinstance(_anisotropy_spectral(gaussian_data), float)


def test_anisotropy_spectral_rank1_high(rank1_data):
    aniso = _anisotropy_spectral(rank1_data)
    assert aniso > 10.0


def test_anisotropy_spectral_gaussian_low(gaussian_data):
    aniso = _anisotropy_spectral(gaussian_data)
    # Isotropic data should have anisotropy close to 1 (within a loose bound)
    assert aniso < 5.0


# ---------------------------------------------------------------------------
# _gaussianity_ep
# ---------------------------------------------------------------------------


def test_gaussianity_ep_positive(gaussian_data):
    ep = _gaussianity_ep(gaussian_data, num_directions=64, seed=0)
    print(f"EP: {ep}")
    assert ep > 0


def test_gaussianity_ep_type(gaussian_data):
    assert isinstance(_gaussianity_ep(gaussian_data, num_directions=64, seed=0), float)


def test_gaussianity_ep_deterministic(gaussian_data):
    ep1 = _gaussianity_ep(gaussian_data, num_directions=64, seed=42)
    ep2 = _gaussianity_ep(gaussian_data, num_directions=64, seed=42)
    assert ep1 == pytest.approx(ep2, abs=1e-6)


def test_gaussianity_ep_gaussian_lower_than_uniform():
    torch.manual_seed(3)
    gaussian = torch.randn(200, 16)
    uniform = torch.rand(200, 16) * 6 - 3  # uniform in [-3, 3]
    ep_gauss = _gaussianity_ep(gaussian, num_directions=128, seed=0)
    ep_uniform = _gaussianity_ep(uniform, num_directions=128, seed=0)
    assert ep_gauss < ep_uniform


def test_gaussianity_ep_large_n_subsampling_branch():
    """N > 2048 exercises the subsampling branch without crashing."""
    torch.manual_seed(4)
    X = torch.randn(2100, 8)
    ep = _gaussianity_ep(X, num_directions=32, seed=0)
    assert ep > 0


# ---------------------------------------------------------------------------
# _compute_knn_indices
# ---------------------------------------------------------------------------


def test_compute_knn_indices_shape():
    X = torch.randn(20, 8)
    indices = _compute_knn_indices(X, maxk=5)
    assert indices.shape == (20, 6)


def test_compute_knn_indices_self_at_col0():
    X = torch.randn(30, 8)
    indices = _compute_knn_indices(X, maxk=4)
    for i in range(30):
        assert indices[i, 0] == i


def test_compute_knn_indices_valid_range():
    N = 25
    X = torch.randn(N, 8)
    indices = _compute_knn_indices(X, maxk=4)
    assert indices.min() >= 0
    assert indices.max() < N


def test_compute_knn_indices_exact_order():
    """3 collinear points: check that nearest neighbour is correct."""
    X = torch.tensor([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
    indices = _compute_knn_indices(X, maxk=2)
    # Point 0: self=0, nearest=1, farthest=2
    assert indices[0, 0] == 0
    assert indices[0, 1] == 1
    assert indices[0, 2] == 2
    # Point 2: self=2, nearest=1, farthest=0
    assert indices[2, 0] == 2
    assert indices[2, 1] == 1
    assert indices[2, 2] == 0


# ---------------------------------------------------------------------------
# _ii_return_imbalance
# ---------------------------------------------------------------------------


def test_ii_return_imbalance_range(identity_knn):
    rng = np.random.default_rng(0)
    imbalance = _ii_return_imbalance(identity_knn, identity_knn, rng, k=1)
    assert 0.0 <= imbalance <= 2.0


def test_ii_return_imbalance_identical_spaces(identity_knn):
    """Identical spaces → imbalance ≤ 1 (better than random, which gives ~1)."""
    rng = np.random.default_rng(0)
    imbalance = _ii_return_imbalance(identity_knn, identity_knn, rng, k=1)
    # The rank of the k-NN in the same ordering is low (≤ k), so imbalance < 1
    assert imbalance < 1.0


def test_ii_return_imbalance_nonnegative():
    torch.manual_seed(5)
    X1 = torch.randn(40, 8)
    X2 = torch.randn(40, 8)
    idx1 = _compute_knn_indices(X1, maxk=5)
    idx2 = _compute_knn_indices(X2, maxk=5)
    rng = np.random.default_rng(1)
    imbalance = _ii_return_imbalance(idx1, idx2, rng, k=1)
    assert imbalance >= 0.0


# ---------------------------------------------------------------------------
# _resolve_num_samples
# ---------------------------------------------------------------------------


def _make_mock_dl(dataset_size: int):
    dl = MagicMock()
    dl.dataset = MagicMock()
    dl.dataset.__len__ = MagicMock(return_value=dataset_size)
    return dl


def test_resolve_num_samples_fraction():
    dl = _make_mock_dl(200)
    result = _resolve_num_samples(0.5, dl)
    assert result == 100


def test_resolve_num_samples_full_fraction():
    dl = _make_mock_dl(300)
    result = _resolve_num_samples(1.0, dl)
    assert result == 300


def test_resolve_num_samples_int_passthrough():
    dl = _make_mock_dl(200)
    result = _resolve_num_samples(42, dl)
    assert result == 42


def test_resolve_num_samples_small_fraction_clamp():
    """Very small fraction should return at least 1."""
    dl = _make_mock_dl(10)
    result = _resolve_num_samples(0.01, dl)
    assert result >= 1


def test_resolve_num_samples_returns_int():
    dl = _make_mock_dl(100)
    assert isinstance(_resolve_num_samples(0.3, dl), int)
    assert isinstance(_resolve_num_samples(10, dl), int)


# ---------------------------------------------------------------------------
# _collect_representations — multiview batch format
# ---------------------------------------------------------------------------


class TestCollectRepresentationsMultiviewBatch:
    """Verify _collect_representations unpacks the (audio_list, text_list, mask) format."""

    @staticmethod
    def _make_pl_module(embed_dim=16, proj_size=8, n_layers=2):
        """Build a minimal mock pl_module with a fake audio/text encoder."""

        # Fake inner model that returns deterministic embeddings
        inner = MagicMock()
        inner.parameters = MagicMock(
            return_value=iter([torch.nn.Parameter(torch.empty(1))])
        )
        inner.extract_embeddings = MagicMock(
            side_effect=lambda audio, layers: torch.randn(
                len(layers), audio.shape[0], 5, embed_dim
            )
        )
        inner.training = False
        inner.eval = MagicMock()
        inner.train = MagicMock()

        audio_encoder = MagicMock()
        audio_encoder.model = inner
        audio_encoder.training = False
        audio_encoder.eval = MagicMock()
        audio_encoder.train = MagicMock()

        # Simple linear projector
        proj_a = torch.nn.Linear(embed_dim, proj_size)

        # Fake text encoder
        text_encoder = MagicMock()
        text_encoder.model.encode = MagicMock(
            side_effect=lambda texts, **kw: torch.randn(len(texts), embed_dim)
        )
        proj_t = torch.nn.Linear(embed_dim, proj_size)

        pl_module = MagicMock()
        pl_module.audio_encoder = audio_encoder
        pl_module.text_encoder = text_encoder
        pl_module.proj_a = proj_a
        pl_module.proj_t = proj_t
        pl_module.device = torch.device("cpu")

        return pl_module

    @staticmethod
    def _make_multiview_dl(
        n_batches=2, batch_size=4, audio_len=100, n_audio_views=2, n_text_views=2
    ):
        """Build a fake dataloader yielding (audio_list, text_list, mask) batches."""
        batches = []
        for _ in range(n_batches):
            audio_list = [
                torch.randn(batch_size, audio_len) for _ in range(n_audio_views)
            ]
            text_list = [
                [f"text_{i}" for i in range(batch_size)] for _ in range(n_text_views)
            ]
            mask = None
            batches.append((audio_list, text_list, mask))
        return batches

    def test_unpacks_first_audio_view(self):
        """_collect_representations should use batch[0][0], not batch[0]."""
        pl_module = self._make_pl_module()
        dl = self._make_multiview_dl(n_batches=1, batch_size=4, n_audio_views=2)

        cb = LayerWiseMetricsCallback(
            num_samples=4, layers={0, 1}, info_imbalance=False
        )
        audio_reps, _ = cb._collect_representations(pl_module, dl)

        assert audio_reps is not None
        # Should have collected from layer 0, 1, and the projected layer (2)
        assert 0 in audio_reps
        assert 1 in audio_reps
        assert audio_reps[0].shape[0] == 4

    def test_unpacks_first_text_view(self):
        """With info_imbalance=True, should use batch[1][0] for text."""
        pl_module = self._make_pl_module()
        dl = self._make_multiview_dl(n_batches=1, batch_size=4, n_text_views=3)

        cb = LayerWiseMetricsCallback(num_samples=4, layers={0}, info_imbalance=True)
        audio_reps, text_reps = cb._collect_representations(pl_module, dl)

        assert audio_reps is not None
        assert text_reps is not None
        assert "text" in text_reps
        assert text_reps["text"].shape[0] == 4

        # Verify encode was called with the first text view (list of 4 strings)
        call_args = pl_module.text_encoder.model.encode.call_args
        texts_passed = call_args[0][0]
        assert len(texts_passed) == 4
        assert all(isinstance(t, str) for t in texts_passed)

    def test_single_view_batch_works(self):
        """Single-view batches (lists of length 1) should also work."""
        pl_module = self._make_pl_module()
        dl = self._make_multiview_dl(
            n_batches=1, batch_size=4, n_audio_views=1, n_text_views=1
        )

        cb = LayerWiseMetricsCallback(num_samples=4, layers={0}, info_imbalance=True)
        audio_reps, text_reps = cb._collect_representations(pl_module, dl)

        assert audio_reps is not None
        assert text_reps is not None
