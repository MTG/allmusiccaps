import gin

from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as L


from ..contrastive_losses import DDPSigmoidLoss, InfoNCELoss, MultimodalInfoNCELoss


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
class ClapJepa(L.LightningModule):
    """
    Combined CLAP + LeJEPA module.
    Blends contrastive invariance (CLAP) and regularization (SigReg) losses:
        loss = lamb * contrastive_loss + (1 - lamb) * jepa_loss
    """

    max_len_s = 30  # s

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
        # contrastive / invariance loss params
        temp: float = 0.1,
        inv_loss_type: str = "info_nce_multimodal",
        # JEPA loss params
        lambd: float | tuple[float, float] = 0.02,
        n_points: int = 17,
        n_slices: int = 256,
        # blend param
        mix_sigreg_views: bool = False,
    ):
        super(ClapJepa, self).__init__()

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

        # contrastive params
        self.temp = temp

        # JEPA params
        self._lambd_dynamic = isinstance(lambd, (tuple, list))
        if self._lambd_dynamic:
            lambd_tuple = tuple(lambd)  # type: ignore[arg-type]
            assert len(lambd_tuple) == 2, "lambd tuple must have exactly 2 floats"
            self._lambd_start = float(lambd_tuple[0])
            self._lambd_end = float(lambd_tuple[1])
            self.lambd: float = self._lambd_start
        else:
            self.lambd = float(lambd)  # type: ignore[arg-type]
        self.n_points = n_points
        self.n_slices = n_slices

        # blend param
        self.mix_sigreg_views = mix_sigreg_views

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

        # aux projection layers
        if self.audio_encoder.__class__.__name__ == "Transformer":
            self.a_z_size = self.audio_encoder.head_dim
        elif hasattr(self.audio_encoder, "embed_dim"):
            self.a_z_size = self.audio_encoder.embed_dim
        else:
            raise ValueError("Unknown audio encoder type")

        self.proj_a = nn.Linear(self.a_z_size, self.proj_size)

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
        self.proj_t = nn.Linear(self.t_z_size, self.proj_size)

        # JEPA sigreg loss. Optional: without the `lejepa` extra this stays None
        # so published weights still load for inference, and `training_step`
        # raises only if training actually needs it.
        lejepa_imp = _import_lejepa(required=False)
        if lejepa_imp is None:
            self.loss_sigreg = None
        else:
            univariate_test = lejepa_imp.univariate.EppsPulley(n_points=self.n_points)
            self.loss_sigreg = lejepa_imp.multivariate.SlicingUnivariateTest(
                univariate_test=univariate_test, num_slices=self.n_slices
            )

        # invariance loss
        self.inv_loss_type = inv_loss_type
        if self.inv_loss_type == "dino":
            self.loss_inv = self._loss_dino
        elif self.inv_loss_type == "cosine":
            self.loss_inv = self._loss_cosine
        elif self.inv_loss_type == "ddp_sigmoid_loss":
            self._loss_inv_module = DDPSigmoidLoss(temp=self.temp)
            self.loss_inv = self._loss_inv_bimodal
        elif self.inv_loss_type == "info_nce":
            self._loss_inv_module = InfoNCELoss(temp=self.temp)
            self.loss_inv = self._loss_inv_bimodal
        elif self.inv_loss_type == "info_nce_multimodal":
            self._loss_inv_module = MultimodalInfoNCELoss(temp=self.temp)
            self.loss_inv = self._loss_inv_bimodal
        else:
            raise ValueError(f"Unknown inv_loss_type: {self.inv_loss_type}")

    def _sigreg(self, z):
        """Apply the SigReg loss, failing clearly if the extra is not installed."""
        if self.loss_sigreg is None:
            _import_lejepa(required=True)  # raises with install instructions
        return self.loss_sigreg(z)

    def forward_audio(self, audio, attn_mask=None):
        if isinstance(attn_mask, list) and len(attn_mask) > 0:
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
            pass

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

        if self.train_text_encoder and self.training:
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

        z_a = self.forward_audio(audio_views[0], m)
        z_t = self.forward_text(text_views[0])

        return z_a, z_t

    def _loss_dino(self, z_a: torch.Tensor, z_t: torch.Tensor):
        z = torch.stack([z_a, z_t], dim=0)  # (V, B, D)
        return (z.mean(0) - z).square().mean()

    def _loss_cosine(self, z_a: torch.Tensor, z_t: torch.Tensor, eps=1e-8):
        z = torch.stack([z_a, z_t], dim=0)  # (V, B, D)
        z_norm = F.normalize(z, dim=-1, eps=eps)

        z_m = z_norm.mean(dim=0, keepdim=True)
        z_m = F.normalize(z_m, dim=-1, eps=eps)

        cos_sim = (z_norm * z_m).sum(dim=-1)

        return (1.0 - cos_sim).mean()

    def _loss_inv_bimodal(self, z_a: torch.Tensor, z_t: torch.Tensor):
        return self._loss_inv_module(z_a, z_t)

    def _get_lambd(self) -> float:
        if not self._lambd_dynamic:
            return float(self.lambd)
        trainer = self.trainer
        current_step = trainer.global_step
        max_steps = trainer.max_steps
        if max_steps is None or max_steps <= 0:
            return self._lambd_start
        progress = min(current_step / max_steps, 1.0)
        return self._lambd_start + progress * (self._lambd_end - self._lambd_start)

    def _compute_losses(self, z_a, z_t):
        # invariance loss
        loss_inv = self.loss_inv(z_a, z_t)

        # sigreg loss
        if self.mix_sigreg_views:
            B = z_a.size(0)
            half = B // 2
            view_1 = torch.cat([z_a[:half], z_t[half:]], dim=0)  # (B, D)
            view_2 = torch.cat([z_t[:half], z_a[half:]], dim=0)  # (B, D)
            z = torch.stack([view_1, view_2], dim=0)  # (2, B, D)
        else:
            z = torch.stack([z_a, z_t], dim=0)  # (V, B, D)
        loss_sigreg = self._sigreg(z)

        # combined loss
        lambd = self._get_lambd()
        loss = lambd * loss_sigreg + (1 - lambd) * loss_inv

        return loss, loss_inv, loss_sigreg, lambd

    def training_step(self, batch, batch_idx):
        z_a, z_t = self.forward(batch)
        B = z_a.shape[0]

        loss, loss_inv, loss_sigreg, lambd = self._compute_losses(z_a, z_t)

        self.log("train_loss_sigreg", loss_sigreg, prog_bar=True, batch_size=B)
        self.log("train_loss_inv", loss_inv, prog_bar=True, batch_size=B)
        self.log("train_loss", loss, prog_bar=True, batch_size=B)
        if self._lambd_dynamic:
            self.log("lambd", lambd, prog_bar=True, batch_size=B)
        if self.mix_sigreg_views:
            self.log("mix_sigreg_views", True, prog_bar=False, batch_size=B)

        return loss

    def validation_step(self, batch, batch_idx):
        z_a, z_t = self.forward(batch)
        B = z_a.shape[0]

        loss, loss_inv, loss_sigreg, _ = self._compute_losses(z_a, z_t)

        self.log(
            "val_loss_sigreg",
            loss_sigreg,
            prog_bar=True,
            batch_size=B,
            sync_dist=True,
        )
        self.log("val_loss_inv", loss_inv, prog_bar=True, batch_size=B, sync_dist=True)
        self.log("val_loss", loss, prog_bar=True, batch_size=B, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, filenames, segment_idxs, n_segments_batch = batch

        embeddings = self.forward_audio(x).detach().cpu()

        for i in range(len(filenames)):
            self.predict_data[filenames[i]].append(embeddings[i, :])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
