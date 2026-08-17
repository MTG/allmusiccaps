import math
import gin

from collections import defaultdict

import torch
import lightning.pytorch as L
from torch import nn
from torch.distributions.laplace import Laplace
import torch.nn.functional as F


from ..contrastive_losses import MultiPositiveInfoNCELoss


def _sigma_gn(p: float) -> float:
    """Scale sigma so that GN_p(0, sigma) has unit variance (mode_of_sigma='sigma_GN')."""
    return (math.gamma(1 / p) ** 0.5) / ((p ** (1 / p)) * (math.gamma(3 / p) ** 0.5))


def _sample_gn(
    B: int,
    D: int,
    p_norm: float,
    mu: float,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample from GN_p(mu, sigma). Optimised for p=1 (Laplace) and p=2 (Gaussian)."""
    if p_norm == 1.0:
        loc_t = torch.tensor(mu, device=device, dtype=dtype)
        scale_t = torch.tensor(sigma, device=device, dtype=dtype)
        return Laplace(loc_t, scale_t).sample((B, D))
    elif p_norm == 2.0:
        return mu + sigma * torch.randn(B, D, device=device, dtype=dtype)
    else:
        sign = torch.empty(B, D, device=device, dtype=dtype).bernoulli_(0.5) * 2 - 1
        g = (
            torch.distributions.Gamma(1.0 / p_norm, 1.0)
            .sample((B, D))
            .to(device=device, dtype=dtype)
        )
        return mu + sigma * sign * (p_norm * g).pow(1.0 / p_norm)


@gin.configurable
class LpJEPA(L.LightningModule):
    """
    LpJEPA model.
    Replaces LeJEPA SigReg with Sliced 2-Wasserstein distance against a
    Rectified Generalized Gaussian (RGG) target distribution.

    Architecture mirrors LeJEPA exactly (projectors, invariance loss, view
    mixing, attention pooling) so that the regularisation loss is the only
    difference between the two models.
    """

    def __init__(
        self,
        text_encoder: nn.Module,
        audio_encoder: nn.Module,
        train_audio_encoder: bool,
        train_text_encoder: bool,
        proj_size: int,
        lr: float,
        weight_decay: float,
        seed: int,
        aggregation_type: str = "mean",
        n_pool_att_heads: int = 8,
        embedding_prompt: str | None = "",
        lambd: float = 0.02,
        inv_loss: str = "dino",
        temp: float = 0.1,
        mu: float = 0.0,
        p_norm: float = 1.0,
        n_slices: int = 8192,
        target_distribution: str = "rectified_lp_distribution",
        sw_mode: str = "independent",
        mix_reg_views: bool = False,
        n_text_views: int = 1,
    ):
        super(LpJEPA, self).__init__()

        self.seed = seed
        self.train_audio_encoder = train_audio_encoder
        self.train_text_encoder = train_text_encoder
        self.lr = lr
        self.weight_decay = weight_decay
        self.proj_size = proj_size
        self.aggregation_type = aggregation_type
        self.n_pool_att_heads = n_pool_att_heads
        self.embedding_prompt = embedding_prompt
        self.lambd = lambd
        self.temp = temp
        self.mu = mu
        self.p_norm = p_norm
        self.n_slices = n_slices
        self.target_distribution = target_distribution
        self.mix_reg_views = mix_reg_views

        if sw_mode not in ("independent", "joint"):
            raise ValueError(
                f"Unknown sw_mode: {sw_mode!r}. Use 'independent' or 'joint'."
            )
        self.sw_mode = sw_mode

        if inv_loss not in ("dino", "cosine", "info_nce"):
            raise ValueError(
                f"Unknown inv_loss: {inv_loss!r}. Use 'dino', 'cosine', or 'info_nce'."
            )
        self.inv_loss = inv_loss

        # Multiview parameters
        self.n_text_views = n_text_views

        # sigma_GN: normalise GN_p(0, sigma) to unit variance before rectification
        self.sigma = _sigma_gn(p_norm)

        print(
            f"LpJEPA: p_norm={p_norm}, mu={mu}, sigma={self.sigma:.6f} "
            f"(sigma_GN, unit-var before ReLU), target={target_distribution}"
        )

        self.predict_data = defaultdict(list)

        self.text_encoder = text_encoder()
        for param in self.text_encoder.model.parameters():
            param.requires_grad = self.train_text_encoder
        if self.train_text_encoder:
            self.text_encoder.train()
        else:
            self.text_encoder.eval()

        self.audio_encoder = audio_encoder()
        for _, param in self.audio_encoder.named_parameters():
            param.requires_grad = self.train_audio_encoder
        if self.train_audio_encoder:
            self.audio_encoder.train()
        else:
            self.audio_encoder.eval()

        # Determine audio encoder output size
        if self.audio_encoder.__class__.__name__ == "Transformer":
            self.a_z_size = self.audio_encoder.head_dim
        elif hasattr(self.audio_encoder, "embed_dim"):
            self.a_z_size = self.audio_encoder.embed_dim
        else:
            raise ValueError("Unknown audio encoder type")

        # Projectors (same as LeJEPA: Linear + BatchNorm)
        self.proj_a = nn.Sequential(
            nn.Linear(self.a_z_size, self.proj_size),
            nn.BatchNorm1d(self.proj_size),
        )

        if self.aggregation_type == "attention_pooler":
            sr = self.audio_encoder.sr
            patch_size = self.audio_encoder.patch_size
            self.max_timestamps = int(self.max_len_s * sr / patch_size[1])
            self.proj_att_query = nn.Linear(self.max_timestamps, 1)
            self.att_pooler = nn.MultiheadAttention(
                embed_dim=self.a_z_size,
                num_heads=self.n_pool_att_heads,
                batch_first=True,
            )

        dummy_text = ""
        self.t_z_size = self.text_encoder.model.encode(dummy_text).shape[0]
        self.proj_t = nn.Sequential(
            nn.Linear(self.t_z_size, self.proj_size),
            nn.BatchNorm1d(self.proj_size),
        )

        # Choose invariance loss
        if self.inv_loss == "dino":
            self.loss_inv = self.loss_dino
        elif self.inv_loss == "cosine":
            self.loss_inv = self.loss_cosine
        elif self.inv_loss == "info_nce":
            self._info_nce = MultiPositiveInfoNCELoss(temp=self.temp)
            self.loss_inv = self._info_nce

    def _sample_target(
        self, B: int, D: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        raw = _sample_gn(B, D, self.p_norm, self.mu, self.sigma, device, dtype)
        if self.target_distribution == "rectified_lp_distribution":
            return torch.relu(raw)
        elif self.target_distribution == "lp_distribution":
            return raw
        else:
            raise ValueError(f"Unknown target_distribution: {self.target_distribution}")

    def _sliced_wasserstein(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Sliced 2-Wasserstein distance between z and target samples y."""
        D = z.shape[1]
        A = torch.randn(self.n_slices, D, device=z.device, dtype=z.dtype)
        A = A / A.norm(p=2, dim=1, keepdim=True)  # unit rows
        zp_sorted, _ = (z @ A.T).sort(dim=0)
        yp_sorted, _ = (y @ A.T).sort(dim=0)
        return (zp_sorted - yp_sorted).pow(2).mean()

    def _rdmreg_loss(self, views: list[torch.Tensor]) -> torch.Tensor:
        """Compute SW regularization loss for a list of view tensors."""
        if self.sw_mode == "joint":
            z_joint = torch.cat(views, dim=0)  # (V*B, D)
            B_joint = z_joint.shape[0]
            D = z_joint.shape[1]
            y_joint = self._sample_target(B_joint, D, z_joint.device, z_joint.dtype)
            return self._sliced_wasserstein(z_joint, y_joint)
        else:
            # Independent: SW applied separately per view, averaged
            loss = torch.tensor(0.0, device=views[0].device, dtype=views[0].dtype)
            for z_v in views:
                B, D = z_v.shape
                y_v = self._sample_target(B, D, z_v.device, z_v.dtype)
                loss = loss + self._sliced_wasserstein(z_v, y_v)
            return loss / len(views)

    def _sanitize_attn_mask(self, attn_mask):
        if isinstance(attn_mask, list) and len(attn_mask) > 0:
            if attn_mask[0] is None:
                return None
        return attn_mask

    def forward_audio(self, audio, attn_mask=None):
        if isinstance(attn_mask, list):
            if attn_mask[0] is None:
                attn_mask = None

        if attn_mask is not None:
            x_a = self.audio_encoder(audio, attn_mask)
        else:
            x_a = self.audio_encoder(audio)

        if self.aggregation_type == "none":
            assert x_a.dim() == 2, (
                "When using 'none' aggregation, the audio encoder must output fixed-size embeddings."
            )
        elif self.aggregation_type == "mean":
            x_a = x_a.mean(dim=1)
        elif self.aggregation_type == "attention_pooler":
            q_emb = torch.swapaxes(x_a, -2, -1)
            q_emb = self.proj_att_query(q_emb)
            q_emb = torch.swapaxes(q_emb, -2, -1)
            x_a, _ = self.att_pooler(q_emb, x_a, x_a)
            x_a.squeeze_(dim=1)
        else:
            raise ValueError(f"Unknown aggregation type: {self.aggregation_type}")

        return self.proj_a(x_a)

    def forward_text(self, text):
        if self.embedding_prompt:
            text = [self.embedding_prompt + t for t in text]

        if self.train_text_encoder:
            inputs = self.text_encoder.model.tokenize(text)
            # sentence-transformers >=5.1 can return non-tensor entries here
            # (e.g. the original strings), so only move the tensors.
            inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            outputs = self.text_encoder.model(inputs)
            if isinstance(outputs, dict):
                x_t = outputs["sentence_embedding"]
            else:
                x_t = outputs
        else:
            x_t = self.text_encoder.model.encode(
                text, convert_to_tensor=True, device=self.device
            )
            x_t = x_t.clone()
        return self.proj_t(x_t)

    def forward(self, batch):
        audio_views, text_views, m = batch

        if self.training:
            z_ts = [self.forward_text(tv) for tv in text_views]
            z_as = [self.forward_audio(av, m) for av in audio_views]
        else:
            # Use single view per modality during validation for balanced loss
            z_ts = [self.forward_text(text_views[0])]
            z_as = [self.forward_audio(audio_views[0], m)]

        return z_as, z_ts

    def _build_reg_views(self, views):
        """Build views for regularization loss from a list of view tensors."""
        z = torch.stack(views, dim=0)  # (V, B, D)

        if self.mix_reg_views and len(views) == 2:
            # Legacy 2-view mixing
            z_a, z_t = views[0], views[1]
            B = z_a.size(0)
            half = B // 2
            view_1 = torch.cat([z_a[:half], z_t[half:]], dim=0)
            view_2 = torch.cat([z_t[:half], z_a[half:]], dim=0)
            return torch.stack([view_1, view_2], dim=0)
        elif self.mix_reg_views and len(views) > 2:
            # For V > 2, randomly permute view assignments per sample
            V, B, D = z.shape
            z_mixed = torch.empty_like(z)
            for b in range(B):
                perm = torch.randperm(V, device=z.device)
                z_mixed[:, b, :] = z[perm, b, :]
            return z_mixed
        else:
            return z

    def loss_dino(self, z: torch.Tensor):
        """
        Calculates the DINO loss for a given tensor.
        This implementation does not distinguish between local and global views.

        Args:
            z (torch.Tensor): The input tensor with shape [V, B, D],
                where B is the batch size, V the number of views, and D is the feature dimension.

        Returns:
            torch.Tensor: The computed DINO loss value.
        """
        return (z.mean(0) - z).square().mean()

    def loss_cosine(self, z: torch.Tensor, eps=1e-8):
        """
        Calculates the cosine similarity loss.
        To support multiple views, z is expected to have shape [V, B, E].
        The loss works by computing the mean representation across views for each sample,
        then calculating the cosine similarity between each view's representation and the mean.
        The final loss is the average of (1 - cosine similarity) across all views and samples

        Args:
            z (torch.Tensor): The input tensor with shape [V, B, E].
            eps (float): A small value to avoid division by zero during normalization. Default is 1e-8.

        Returns:
            torch.Tensor: The average cosine similarity loss value.
        """
        z_norm = F.normalize(z, dim=-1, eps=eps)  # [V, B, E]

        z_m = z_norm.mean(dim=0, keepdim=True)  # [B, 1, E]
        z_m = F.normalize(z_m, dim=-1, eps=eps)

        # compute 1 - cos for each view-sample, average
        cos_sim = (z_norm * z_m).sum(dim=-1)  # [V, B]

        return (1.0 - cos_sim).mean()  # average over views and batch

    def training_step(self, batch, batch_idx):
        z_as, z_ts = self.forward(batch)
        views = z_as + z_ts
        B = views[0].shape[0]

        # stack views on the first dimension
        z = torch.stack(views, dim=0)  # (V, B, D)
        z_reg = self._build_reg_views(views)

        loss_sw = self._rdmreg_loss(list(z_reg))
        loss_inv = self.loss_inv(z)

        loss = self.lambd * loss_sw + (1 - self.lambd) * loss_inv

        self.log("train_loss_sw", loss_sw, prog_bar=True, batch_size=B)
        self.log("train_loss_inv", loss_inv, prog_bar=True, batch_size=B)
        self.log("train_loss", loss, prog_bar=True, batch_size=B)

        return loss

    def validation_step(self, batch, batch_idx):
        z_as, z_ts = self.forward(batch)
        views = z_as + z_ts
        B = views[0].shape[0]

        # stack views on the first dimension
        z = torch.stack(views, dim=0)  # (V, B, D)
        z_reg = self._build_reg_views(views)

        loss_sw = self._rdmreg_loss(list(z_reg))
        loss_inv = self.loss_inv(z)

        loss = self.lambd * loss_sw + (1 - self.lambd) * loss_inv

        self.log("val_loss_sw", loss_sw, prog_bar=True, batch_size=B, sync_dist=True)
        self.log("val_loss_inv", loss_inv, prog_bar=True, batch_size=B, sync_dist=True)
        self.log("val_loss", loss, prog_bar=True, batch_size=B, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, filenames, segment_idxs, n_segments_batch = batch

        embeddings = self.forward_audio(x).detach().cpu()

        for i in range(len(filenames)):
            self.predict_data[filenames[i]].append(embeddings[i, :])

    def on_before_optimizer_step(self, optimizer):
        if self.global_step % self.trainer.log_every_n_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), float("inf"))
            self.log("grad_norm", grad_norm, prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
