import gin

from collections import defaultdict

import torch
import lightning.pytorch as L
from torch import nn
import torch.nn.functional as F


from ..contrastive_losses import MultiPositiveInfoNCELoss, MultiPositiveSigmoidLoss


def _import_lejepa(required: bool = True):
    """Import the optional `lejepa` package, used only to build the SigReg loss.

    `lejepa` has no PyPI release, and PyPI rejects direct (git) references in
    every dependency field, so it cannot ship in an extra either: it has to be
    installed by hand. Inference on published weights never needs it: the
    exported checkpoints carry no SigReg tensors, so the loss is constructed
    but never evaluated.

    Returns None when the package is missing and `required` is False, which lets
    a published model load for inference without the training extra installed.
    """
    try:
        import lejepa
    except ImportError as exc:  # pragma: no cover - depends on install extras
        if not required:
            return None
        raise ImportError(
            "The SigReg loss requires the optional `lejepa` package, which "
            "has no PyPI release and must be installed from git:\n"
            '  pip install "lejepa @ git+https://github.com/rbalestr-lab/lejepa.git"'
        ) from exc
    return lejepa


@gin.configurable
class LeJEPA(L.LightningModule):
    """
    Contrastive Language-Audio Pretraining (CLAP) model.
    inspired in

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
        n_points=17,
        n_slices: int = 256,
        mix_sigreg_views: bool = False,
        n_text_views: int = 1,
    ):
        super(LeJEPA, self).__init__()

        # global variables
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

        self.predict_data = defaultdict(list)

        self.text_encoder = text_encoder()
        self.inv_loss = inv_loss

        self.temp = temp
        self.n_points = n_points
        self.n_slices = n_slices
        self.mix_sigreg_views = mix_sigreg_views

        self.n_text_views = n_text_views

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

        # aux projection layers
        if self.audio_encoder.__class__.__name__ == "Transformer":
            self.a_z_size = self.audio_encoder.head_dim
        elif self.audio_encoder.__class__.__name__ == "OMARRQ":
            self.a_z_size = self.audio_encoder.embed_dim
        else:
            raise ValueError("Unknown audio encoder type")

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

        # Build the SigReg loss (Epps-Pulley univariate test, sliced to
        # multivariate). Optional: without the `lejepa` extra this stays None so
        # published weights still load for inference, and `training_step` raises
        # only if training actually needs it.
        lejepa_imp = _import_lejepa(required=False)
        if lejepa_imp is None:
            self.loss_sigreg = None
        else:
            univariate_test = lejepa_imp.univariate.EppsPulley(n_points=self.n_points)
            self.loss_sigreg = lejepa_imp.multivariate.SlicingUnivariateTest(
                univariate_test=univariate_test, num_slices=self.n_slices
            )

        # choose invariance loss
        if self.inv_loss == "dino":
            self.loss_inv = self.loss_dino
        elif self.inv_loss == "cosine":
            self.loss_inv = self.loss_cosine
        elif self.inv_loss == "info_nce":
            self._info_nce = MultiPositiveInfoNCELoss(temp=self.temp)
            self.loss_inv = self._info_nce
        elif self.inv_loss == "sigmoid":
            self._sigmoid = MultiPositiveSigmoidLoss()
            self.loss_inv = self._sigmoid
        else:
            raise ValueError("Unknown inv_loss type")

    def _sigreg(self, z):
        """Apply the SigReg loss, failing clearly if the extra is not installed."""
        if self.loss_sigreg is None:
            _import_lejepa(required=True)  # raises with install instructions
        return self.loss_sigreg(z)

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
            x_a = self.audio_encoder(audio, attn_mask)  # (B, T, D) -> (B, T', D)
        else:
            x_a = self.audio_encoder(audio)  # (B, T, D) -> (B, T', D)

        if self.aggregation_type == "none":
            assert x_a.dim() == 2, (
                "When using 'none' aggregation, the audio encoder must output fixed-size embeddings."
            )
            pass

        elif self.aggregation_type == "mean":
            x_a = x_a.mean(dim=1)  # (B, 768)
        elif self.aggregation_type == "attention_pooler":
            # Do self attention aggregation
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
            # Use forward pass to get gradients when training
            # encode() doesn't propagate gradients, so we use the model directly
            inputs = self.text_encoder.model.tokenize(text)
            # sentence-transformers >=5.1 can return non-tensor entries here
            # (e.g. the original strings), so only move the tensors.
            inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            outputs = self.text_encoder.model(inputs)
            # Extract sentence embeddings from the model output
            # sentence_transformers returns a dict with 'sentence_embedding' key
            if isinstance(outputs, dict):
                x_t = outputs["sentence_embedding"]
            else:
                # Fallback: if output is a tensor directly
                x_t = outputs
        else:
            # Use encode() for efficiency when not training (no gradients needed)
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

    def _build_sigreg_views(self, views):
        """Build views for SigReg loss from a list of view tensors."""
        z = torch.stack(views, dim=0)  # (V, B, D)

        if self.mix_sigreg_views and len(views) == 2:
            # Legacy 2-view mixing
            z_a, z_t = views[0], views[1]
            B = z_a.size(0)
            half = B // 2
            view_1 = torch.cat([z_a[:half], z_t[half:]], dim=0)
            view_2 = torch.cat([z_t[:half], z_a[half:]], dim=0)
            return torch.stack([view_1, view_2], dim=0)
        elif self.mix_sigreg_views and len(views) > 2:
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
        z_sigreg = self._build_sigreg_views(views)

        # LeJEPA loss is applied independently per view
        loss_inv = self.loss_inv(z)

        if self.lambd > 0:
            loss_sigreg = self._sigreg(z_sigreg)
            loss = self.lambd * loss_sigreg + (1 - self.lambd) * loss_inv
        else:
            loss_sigreg = torch.tensor(0.0, device=z.device)
            loss = loss_inv

        self.log("train_loss_sigreg", loss_sigreg, prog_bar=True, batch_size=B)
        self.log("train_loss_inv", loss_inv, prog_bar=True, batch_size=B)
        self.log("train_loss", loss, prog_bar=True, batch_size=B)

        return loss

    def validation_step(self, batch, batch_idx):
        z_as, z_ts = self.forward(batch)
        views = z_as + z_ts
        B = views[0].shape[0]

        # stack views on the first dimension
        z = torch.stack(views, dim=0)  # (V, B, D)
        z_sigreg = self._build_sigreg_views(views)

        loss_inv = self.loss_inv(z)

        if self.lambd > 0:
            loss_sigreg = self._sigreg(z_sigreg)
            loss = self.lambd * loss_sigreg + (1 - self.lambd) * loss_inv
        else:
            loss_sigreg = torch.tensor(0.0, device=z.device)
            loss = loss_inv

        self.log(
            "val_loss_sigreg", loss_sigreg, prog_bar=True, batch_size=B, sync_dist=True
        )
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
