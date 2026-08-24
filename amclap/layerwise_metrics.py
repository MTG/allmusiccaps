"""Layer-wise geometric metrics callback for AMCLAP.

Computes six geometric metrics over intermediate layer representations:
  - mLID (geometric mean + Fréchet variance)
  - Effective rank (RankMe)
  - Normalised entropy
  - Spectral anisotropy
  - Epps–Pulley gaussianity
  - Information imbalance (optional)

optionally computes information imbalance between each audio layer and the
text tower (requires info_imbalance=True).

designed for OMARRQ audio encoders whose extract_embeddings() method returns
per-layer tensors of shape (n_layers, B, T, D).
"""

import logging
from collections import defaultdict

import gin
import numpy as np
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_only
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure metric functions (no architecture coupling)
# ---------------------------------------------------------------------------


def _mlid_mom(X: torch.Tensor, k: int = 64) -> tuple[float, float]:
    """Compute mean Local Intrinsic Dimensionality (mLID) and Fréchet variance.

    Args:
        X: (N, D) float tensor of representations.
        k: number of nearest neighbours.

    Returns:
        (mlid, frechet_var) as Python floats.
    """
    X = X.float()
    N = X.shape[0]
    k = min(k, N - 1)

    # Pairwise distances
    dists = torch.cdist(X, X)  # (N, N)
    # Exclude self-distance: sort and take top k+1, drop col 0 (self)
    knn_dists, _ = dists.topk(k + 1, dim=1, largest=False)
    knn_dists = knn_dists[:, 1:]  # (N, k) — skip self

    # Avoid log(0)
    eps = 1e-8
    knn_dists = knn_dists.clamp(min=eps)
    r_k = knn_dists[:, -1]  # distance to k-th neighbour

    # LID estimate per sample: k / sum(log(r_k / r_i))
    log_ratios = torch.log(r_k.unsqueeze(1) / knn_dists)  # (N, k)
    lid_per_sample = k / log_ratios.sum(dim=1).clamp(min=eps)  # (N,)

    mlid = float(lid_per_sample.log().mean().exp().item())  # geometric mean
    frechet_var = float(lid_per_sample.var().item())

    return mlid, frechet_var


def _effective_rank_and_entropy(X: torch.Tensor) -> tuple[float, float]:
    """Compute effective rank (RankMe) and normalised entropy of singular values.

    Args:
        X: (N, D) float tensor of representations.

    Returns:
        (eff_rank, norm_entropy) as Python floats.
    """
    X = X.float()
    # Centre
    X = X - X.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(X, full_matrices=False)

    p = S / S.sum().clamp(min=1e-8)
    p = p.clamp(min=1e-12)

    entropy = -(p * p.log()).sum().item()
    eff_rank = float(np.exp(entropy))
    norm_entropy = float(entropy / np.log(len(p)))

    return eff_rank, norm_entropy


def _anisotropy_spectral(X: torch.Tensor) -> float:
    """Compute spectral anisotropy: ratio of largest to mean singular value.

    Args:
        X: (N, D) float tensor of representations.

    Returns:
        Anisotropy as a Python float.
    """
    X = X.float()
    X = X - X.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(X, full_matrices=False)
    mean_sv = S.mean().clamp(min=1e-8)
    return float((S[0] / mean_sv).item())


def _gaussianity_ep(X: torch.Tensor, num_directions: int = 256, seed: int = 0) -> float:
    """Epps-Pulley gaussianity via random projections (LeJEPA)."""
    N, D = X.shape
    if N < 2:
        return float("nan")

    X_c = X - X.mean(dim=0)

    gen = torch.Generator(device=X.device).manual_seed(seed)
    A = torch.randn(num_directions, D, device=X.device, generator=gen)
    A = F.normalize(A, dim=1)

    U = X_c @ A.T  # (N, M)

    t = torch.linspace(-5.0, 5.0, 17, device=X.device)  # (T,)

    # ECF per direction: ecf(t) = (1/N) sum_j exp(i*t*u_j)
    # U: (N, M), t: (T,) -> ut: (N, M, T)
    ut = U.unsqueeze(2) * t.unsqueeze(0).unsqueeze(0)
    ecf = torch.complex(torch.cos(ut), torch.sin(ut)).mean(dim=0)  # (M, T)

    phi = torch.exp(-0.5 * t**2)  # (T,) target CF of N(0,1)
    diff_sq = (ecf.real - phi) ** 2 + ecf.imag**2  # (M, T)
    integrand = diff_sq * phi  # w(t) = phi(t)

    ep_per_dir = N * torch.trapezoid(integrand, t, dim=1)  # (M,)

    result = (ep_per_dir.mean() / N).item()  # per-sample, N-independent
    return result if np.isfinite(result) else float("nan")


# ---------------------------------------------------------------------------
# Vendored from dadapy._utils.metric_comparisons (Apache-2.0)
# https://github.com/sissa-data-science/DADApy
# ---------------------------------------------------------------------------


def _ii_return_ranks(
    dist_indices_1: np.ndarray,
    dist_indices_2: np.ndarray,
    rng: np.random.Generator,
    k: int = 1,
) -> np.ndarray:
    """Ranks of distance-1 k-NN in the distance-2 ordering. (N, k) int array."""
    N = dist_indices_1.shape[0]
    maxk_2 = dist_indices_2.shape[1]
    conditional_ranks = np.zeros((N, k))
    for i in range(N):
        idx_k_d1 = dist_indices_1[i, 1 : k + 1]
        wr = [np.where(idx_k_d1[j] == dist_indices_2[i])[0] for j in range(k)]
        for j in range(k):
            # Draw a scalar, not a length-1 array: numpy >=2.3 refuses to assign
            # a sequence into a single element.
            conditional_ranks[i, j] = (
                rng.integers(low=maxk_2, high=N) if len(wr[j]) == 0 else wr[j][0]
            )
    return conditional_ranks


def _ii_return_imbalance(
    dist_indices_1: np.ndarray,
    dist_indices_2: np.ndarray,
    rng: np.random.Generator,
    k: int = 1,
) -> float:
    """Information imbalance from distance-1 space to distance-2 space."""
    N = dist_indices_1.shape[0]
    ranks = _ii_return_ranks(dist_indices_1, dist_indices_2, rng=rng, k=k)
    return float(np.mean(ranks) / (N / 2.0))


def _compute_knn_indices(X: torch.Tensor, maxk: int) -> np.ndarray:
    """Compute (N, maxk+1) int array of kNN indices (self at col 0) from (N, D) tensor."""
    X = X.float()
    dists = torch.cdist(X, X)  # (N, N)
    indices = dists.argsort(dim=1)[:, : maxk + 1].cpu().numpy()  # (N, maxk+1)
    return indices  # row i: [i, nn1, nn2, ...] — self at col 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_num_samples(num_samples: int | float, dataloader) -> int:
    """Resolve num_samples to an integer count.

    If num_samples is a float in (0, 1], it is interpreted as a fraction of the
    total number of samples in the dataloader. If it is an int, it is used directly.
    """
    if isinstance(num_samples, float):
        dataset_size = len(dataloader.dataset)
        return max(1, int(num_samples * dataset_size))
    return num_samples


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


@gin.configurable
class LayerWiseMetricsCallback(pl.Callback):
    """Compute and log layer-wise geometric and information-imbalance metrics.

    Activated via gin:
        train.layerwise_metrics = True

    When info_imbalance=True, also computes information imbalance between each
    audio layer and the text tower in the same collection pass.
    """

    def __init__(
        self,
        num_samples: int | float = 0.2,
        lid_k: int = 64,
        num_gauss_directions: int = 1024,
        layers: set[int] = set(range(12)),
        info_imbalance: bool = False,
        ii_k: int = 1,
        ii_maxk: int = 100,
        ii_seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.lid_k = lid_k
        self.num_gauss_directions = num_gauss_directions
        self.layers = layers
        self.info_imbalance = info_imbalance
        self.ii_k = ii_k
        self.ii_maxk = ii_maxk
        self.ii_seed = ii_seed

    # ------------------------------------------------------------------
    # Representation collection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _collect_representations(
        self, pl_module: pl.LightningModule, val_dl
    ) -> tuple[dict[int, torch.Tensor] | None, dict[str, torch.Tensor] | None]:
        """Collect per-layer audio reps and (optionally) text reps in one pass.

        Returns:
            audio_reps: dict mapping layer_idx -> (N, D) tensor, or None on failure.
            text_reps:  dict with keys "text" and optionally "proj_t", or None when
                        info_imbalance=False or text encoder unavailable.
        """
        audio_encoder = getattr(pl_module, "audio_encoder", None)
        if audio_encoder is None:
            logger.warning("LayerWiseMetricsCallback: pl_module has no audio_encoder.")
            return None, None

        inner_model = getattr(audio_encoder, "model", None)
        if inner_model is None or not hasattr(inner_model, "extract_embeddings"):
            logger.warning(
                "LayerWiseMetricsCallback: audio_encoder.model has no "
                "extract_embeddings(). Skipping."
            )
            return None, None

        device = pl_module.device
        layer_outputs: dict[int, list[torch.Tensor]] = defaultdict(list)
        text_outputs: list[torch.Tensor] = []
        text_proj_outputs: list[torch.Tensor] = []
        collected = 0
        was_training = audio_encoder.training
        audio_encoder.eval()

        num_samples = _resolve_num_samples(self.num_samples, val_dl)
        sorted_layers = sorted(self.layers)
        print(
            f"\n[LayerWiseMetrics] Collecting representations "
            f"(target={num_samples} samples, layers={sorted_layers}"
            f"{', +text' if self.info_imbalance else ''})..."
        )
        pbar = tqdm(
            val_dl, desc="[LayerWiseMetrics] batches", leave=False, unit="batch"
        )
        for batch in pbar:
            audio = batch[0][0].to(device)  # first audio view (B, T)
            audio = audio.to(next(inner_model.parameters()).dtype)

            # (n_layers, B, T', D)
            layer_embeds = inner_model.extract_embeddings(audio, layers=self.layers)
            n_layers = layer_embeds.shape[0]

            for pos, layer_idx in enumerate(sorted_layers):
                x = layer_embeds[pos]  # (B, T', D)
                if x.dim() == 3:
                    x = x.mean(dim=1)  # (B, D) — mean pool over time
                layer_outputs[layer_idx].append(x.cpu())

            # Also capture projected audio embedding as an extra "layer"
            proj_a = getattr(pl_module, "proj_a", None)
            if proj_a is not None:
                last_layer = layer_embeds[-1]  # (B, T', D)
                if last_layer.dim() == 3:
                    last_layer = last_layer.mean(dim=1)  # (B, D)
                layer_outputs[n_layers].append(proj_a(last_layer).cpu())

            # Collect text reps in the same pass when needed
            if self.info_imbalance:
                texts = batch[1][0]  # first text view
                x_t = pl_module.text_encoder.model.encode(
                    texts, convert_to_tensor=True, device=device
                ).clone()
                text_outputs.append(x_t.cpu())
                proj_t = getattr(pl_module, "proj_t", None)
                if proj_t is not None:
                    text_proj_outputs.append(proj_t(x_t).cpu())

            collected += audio.shape[0]
            pbar.set_postfix(collected=collected)
            if collected >= num_samples:
                break

        pbar.close()
        n_total = len(layer_outputs)
        print(
            f"[LayerWiseMetrics] Collected {collected} samples "
            f"across {n_layers} model layers + {n_total - n_layers} extra ({n_total} total)."
        )

        if was_training:
            audio_encoder.train()

        audio_reps = {
            i: torch.cat(v, dim=0)[:num_samples] for i, v in layer_outputs.items()
        }

        text_reps = None
        if self.info_imbalance and text_outputs:
            text_reps = {"text": torch.cat(text_outputs)[:num_samples]}
            if text_proj_outputs:
                text_reps["proj_t"] = torch.cat(text_proj_outputs)[:num_samples]

        return audio_reps, text_reps

    # ------------------------------------------------------------------
    # Metric computation and logging
    # ------------------------------------------------------------------

    @rank_zero_only
    def _run_layerwise_metrics(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        val_dl = trainer.val_dataloaders
        if val_dl is None:
            logger.warning("LayerWiseMetricsCallback: no val_dataloaders available.")
            return

        # val_dataloaders can be a list or a single dataloader
        if isinstance(val_dl, (list, tuple)):
            val_dl = val_dl[0]

        audio_reps, text_reps = self._collect_representations(pl_module, val_dl)
        if audio_reps is None:
            return

        epoch = trainer.current_epoch
        metric_rows: list[dict] = []

        n_layers = len(audio_reps)
        print(
            f"\n[LayerWiseMetrics] Computing metrics for {n_layers} layers (epoch {epoch})..."
        )
        layer_pbar = tqdm(
            sorted(audio_reps.items()),
            desc="[LayerWiseMetrics] layers",
            leave=False,
            unit="layer",
        )
        for layer_idx, X in layer_pbar:
            layer_pbar.set_description(f"[LayerWiseMetrics] layer {layer_idx}")
            X = X.float()

            mlid, frechet_var = _mlid_mom(X, k=self.lid_k)
            eff_rank, norm_entropy = _effective_rank_and_entropy(X)
            anisotropy = _anisotropy_spectral(X)
            gaussianity = _gaussianity_ep(X, num_directions=self.num_gauss_directions)

            metrics = {
                "mlid": mlid,
                "frechet_var": frechet_var,
                "eff_rank": eff_rank,
                "norm_entropy": norm_entropy,
                "anisotropy": anisotropy,
                "gaussianity": gaussianity,
            }

            # Log scalar per-layer metrics to Lightning (visible in TensorBoard etc.)
            for name, val in metrics.items():
                pl_module.log(
                    f"layer_metrics/{name}_layer_{layer_idx}",
                    val,
                    on_step=False,
                    on_epoch=True,
                    rank_zero_only=True,
                )

            # Collect row for wandb table
            row = {"epoch": epoch, "layer": layer_idx}
            row.update(metrics)
            metric_rows.append(row)

        layer_pbar.close()
        print(f"[LayerWiseMetrics] Done. Logging {len(metric_rows)} layer metric rows.")
        self._log_geom_to_wandb(trainer, metric_rows)

        if self.info_imbalance and text_reps is not None:
            self._run_info_imbalance(trainer, pl_module, audio_reps, text_reps)

    def _run_info_imbalance(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        audio_reps: dict[int, torch.Tensor],
        text_cols: dict[str, torch.Tensor],
    ) -> None:
        n_collected = len(next(iter(audio_reps.values())))
        maxk = min(self.ii_maxk, n_collected - 1)

        print(
            f"[InfoImbalance] Computing kNN indices for text columns {list(text_cols.keys())}..."
        )
        text_indices = {
            name: _compute_knn_indices(X, maxk) for name, X in text_cols.items()
        }

        n_audio_layers = len(audio_reps)
        print(
            f"[InfoImbalance] Computing information imbalance for "
            f"{n_audio_layers} audio layers x {len(text_cols)} text column(s)..."
        )
        rows = []
        layer_pbar = tqdm(
            sorted(audio_reps.items()),
            desc="[InfoImbalance] layers",
            leave=False,
            unit="layer",
        )
        for audio_layer_idx, X_audio in layer_pbar:
            layer_pbar.set_description(f"[InfoImbalance] layer {audio_layer_idx}")
            audio_idx = _compute_knn_indices(X_audio, maxk)
            for text_col_name, t_idx in text_indices.items():
                ss = np.random.SeedSequence(self.ii_seed + audio_layer_idx)
                rng_ab, rng_ba = [np.random.default_rng(s) for s in ss.spawn(2)]
                ii_ab = _ii_return_imbalance(
                    audio_idx, t_idx, rng_ab, k=self.ii_k
                )  # audio → text
                ii_ba = _ii_return_imbalance(
                    t_idx, audio_idx, rng_ba, k=self.ii_k
                )  # text → audio
                rows.append(
                    {
                        "step": trainer.global_step,
                        "audio_layer": audio_layer_idx,
                        "text_col": text_col_name,
                        "II_audio_to_text": ii_ab,
                        "II_text_to_audio": ii_ba,
                    }
                )
                pl_module.log(
                    f"info_imbalance/audio{audio_layer_idx}_to_{text_col_name}",
                    ii_ab,
                    rank_zero_only=True,
                )
                pl_module.log(
                    f"info_imbalance/{text_col_name}_to_audio{audio_layer_idx}",
                    ii_ba,
                    rank_zero_only=True,
                )

        layer_pbar.close()
        print(f"[InfoImbalance] Done. Logging {len(rows)} imbalance rows.")
        self._log_ii_to_wandb(trainer, rows)

    def _log_geom_to_wandb(
        self,
        trainer: pl.Trainer,
        metric_rows: list[dict],
    ) -> None:
        """Log geometric metrics to wandb if the logger is available."""
        try:
            import wandb
        except ImportError:
            return

        if wandb.run is None:
            return

        # Scalar per-layer metrics
        for row in metric_rows:
            layer_idx = row["layer"]
            for name in (
                "mlid",
                "frechet_var",
                "eff_rank",
                "norm_entropy",
                "anisotropy",
                "gaussianity",
            ):
                wandb.log(
                    {f"layer_metrics/{name}_layer_{layer_idx}": row[name]},
                    step=trainer.global_step,
                )

    def _log_ii_to_wandb(self, trainer: pl.Trainer, rows: list[dict]) -> None:
        """Log information imbalance metrics to wandb."""
        try:
            import wandb
        except ImportError:
            return

        if wandb.run is None:
            return

        for row in rows:
            audio_layer = row["audio_layer"]
            text_col = row["text_col"]
            wandb.log(
                {
                    f"info_imbalance/audio{audio_layer}_to_{text_col}": row[
                        "II_audio_to_text"
                    ],
                    f"info_imbalance/{text_col}_to_audio{audio_layer}": row[
                        "II_text_to_audio"
                    ],
                },
                step=trainer.global_step,
            )

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._run_layerwise_metrics(trainer, pl_module)
