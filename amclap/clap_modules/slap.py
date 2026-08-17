import copy
import gin

from collections import defaultdict

import torch
import torch.nn.functional as F
import lightning.pytorch as L
from torch import nn


@gin.configurable
class SLAP(L.LightningModule):
    """
    Siamese Language-Audio Pretraining (SLAP) model.
    BYOL-style audio-text model that eliminates negative samples.
    Uses EMA-updated target encoders per modality plus a predictor MLP.
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
        predictor_hidden_size: int = 4096,
        lambd: float = 0.5,
        ema_tau: float = 0.95,
        embedding_prompt: str | None = "",
    ):
        super(SLAP, self).__init__()

        self.seed = seed
        self.train_audio_encoder = train_audio_encoder
        self.train_text_encoder = train_text_encoder
        self.lr = lr
        self.weight_decay = weight_decay
        self.proj_size = proj_size
        self.aggregation_type = aggregation_type
        self.lambd = lambd
        self.ema_tau = ema_tau
        self.embedding_prompt = embedding_prompt
        self.predict_data = defaultdict(list)

        # Context text encoder
        self.text_encoder = text_encoder()
        for param in self.text_encoder.model.parameters():
            param.requires_grad = self.train_text_encoder
        if self.train_text_encoder:
            self.text_encoder.train()
        else:
            self.text_encoder.eval()

        # Context audio encoder
        self.audio_encoder = audio_encoder()
        for _, param in self.audio_encoder.named_parameters():
            param.requires_grad = self.train_audio_encoder
        if self.train_audio_encoder:
            self.audio_encoder.train()
        else:
            self.audio_encoder.eval()

        # Audio encoder output size
        if self.audio_encoder.__class__.__name__ == "Transformer":
            self.a_z_size = self.audio_encoder.head_dim
        elif hasattr(self.audio_encoder, "embed_dim"):
            self.a_z_size = self.audio_encoder.embed_dim
        else:
            raise ValueError("Unknown audio encoder type")

        # Text encoder output size
        dummy_text = ""
        self.t_z_size = self.text_encoder.model.encode(dummy_text).shape[0]

        # Online projectors (context path, receives gradients)
        self.proj_a = nn.Linear(self.a_z_size, self.proj_size)
        self.proj_t = nn.Linear(self.t_z_size, self.proj_size)

        # Predictors
        self.pred_a = self._build_predictor(proj_size, predictor_hidden_size)
        self.pred_t = self._build_predictor(proj_size, predictor_hidden_size)

        # Target encoders (EMA copies, no gradients, always in eval mode)
        self.audio_encoder_target = copy.deepcopy(self.audio_encoder)
        for p in self.audio_encoder_target.parameters():
            p.requires_grad = False
        self.audio_encoder_target.eval()

        self.text_encoder_target = copy.deepcopy(self.text_encoder)
        for p in self.text_encoder_target.parameters():
            p.requires_grad = False
        self.text_encoder_target.eval()

        # Target projectors (EMA copies of online projectors, no gradients)
        self.proj_a_target = copy.deepcopy(self.proj_a)
        for p in self.proj_a_target.parameters():
            p.requires_grad = False

        self.proj_t_target = copy.deepcopy(self.proj_t)
        for p in self.proj_t_target.parameters():
            p.requires_grad = False

    def _build_predictor(self, proj_size: int, hidden_size: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(proj_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, proj_size),
        )

    def _encode_audio(self, encoder, audio, attn_mask=None):
        if isinstance(attn_mask, list) and len(attn_mask) > 0:
            if attn_mask[0] is None:
                attn_mask = None

        if attn_mask is not None:
            x_a = encoder(audio, attn_mask)
        else:
            x_a = encoder(audio)

        if self.aggregation_type == "none":
            assert x_a.dim() == 2, (
                "When using 'none' aggregation, the audio encoder must output fixed-size embeddings."
            )
        elif self.aggregation_type == "mean":
            x_a = x_a.mean(dim=1)
        else:
            raise ValueError(f"Unknown aggregation type: {self.aggregation_type}")

        return x_a

    def forward_audio(self, audio, attn_mask=None):
        x_a = self._encode_audio(self.audio_encoder, audio, attn_mask)
        return self.proj_a(x_a)

    def forward_audio_target(self, audio, attn_mask=None):
        x_a = self._encode_audio(self.audio_encoder_target, audio, attn_mask)
        return self.proj_a_target(x_a)

    def _encode_text(self, encoder, text, train_encoder):
        if self.embedding_prompt:
            text = [self.embedding_prompt + t for t in text]

        if train_encoder and self.training:
            inputs = encoder.model.tokenize(text)
            # sentence-transformers >=5.1 can return non-tensor entries here
            # (e.g. the original strings), so only move the tensors.
            inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            outputs = encoder.model(inputs)
            if isinstance(outputs, dict):
                x_t = outputs["sentence_embedding"]
            else:
                x_t = outputs
        else:
            x_t = encoder.model.encode(text, convert_to_tensor=True, device=self.device)
            x_t = x_t.clone()

        return x_t

    def forward_text(self, text):
        x_t = self._encode_text(self.text_encoder, text, self.train_text_encoder)
        return self.proj_t(x_t)

    def forward_text_target(self, text):
        x_t = self._encode_text(self.text_encoder_target, text, False)
        return self.proj_t_target(x_t)

    def _cosine_loss(self, q: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        q = F.normalize(q, dim=-1)
        z = F.normalize(z_target.detach(), dim=-1)
        return (1 - (q * z).sum(dim=-1)).mean()

    def forward(self, batch):
        audio_views, text_views, m = batch

        # Use first audio/text view for online, second for target (if available)
        a_online = audio_views[0]
        a_target = audio_views[1] if len(audio_views) > 1 else audio_views[0]
        t_online = text_views[0]
        t_target = text_views[1] if len(text_views) > 1 else text_views[0]

        z_a_ctx = self.forward_audio(a_online, m)
        z_t_ctx = self.forward_text(t_online)
        q_a = self.pred_a(z_a_ctx)
        q_t = self.pred_t(z_t_ctx)

        with torch.no_grad():
            z_a_tgt = self.forward_audio_target(a_target, m)
            z_t_tgt = self.forward_text_target(t_target)

        return q_a, q_t, z_a_tgt, z_t_tgt

    def _compute_loss(self, q_a, q_t, z_a_tgt, z_t_tgt):
        loss_at = self._cosine_loss(q_a, z_t_tgt)
        loss_ta = self._cosine_loss(q_t, z_a_tgt)
        loss_a = self._cosine_loss(q_a, z_a_tgt)
        loss_t = self._cosine_loss(q_t, z_t_tgt)

        loss = self.lambd * (loss_at + loss_ta) + (1 - self.lambd) * (loss_a + loss_t)
        return loss, loss_at, loss_ta, loss_a, loss_t

    def training_step(self, batch, batch_idx):
        q_a, q_t, z_a_tgt, z_t_tgt = self.forward(batch)
        B = batch[0][0].shape[0]

        loss, loss_at, loss_ta, loss_a, loss_t = self._compute_loss(
            q_a, q_t, z_a_tgt, z_t_tgt
        )

        self.log("train_loss", loss, prog_bar=True, batch_size=B)
        self.log("train_loss_at", loss_at, prog_bar=False, batch_size=B)
        self.log("train_loss_ta", loss_ta, prog_bar=False, batch_size=B)
        self.log("train_loss_a", loss_a, prog_bar=False, batch_size=B)
        self.log("train_loss_t", loss_t, prog_bar=False, batch_size=B)

        return loss

    def validation_step(self, batch, batch_idx):
        q_a, q_t, z_a_tgt, z_t_tgt = self.forward(batch)
        B = batch[0][0].shape[0]

        loss, loss_at, loss_ta, loss_a, loss_t = self._compute_loss(
            q_a, q_t, z_a_tgt, z_t_tgt
        )

        self.log("val_loss", loss, prog_bar=True, batch_size=B, sync_dist=True)
        self.log("val_loss_at", loss_at, prog_bar=False, batch_size=B, sync_dist=True)
        self.log("val_loss_ta", loss_ta, prog_bar=False, batch_size=B, sync_dist=True)
        self.log("val_loss_a", loss_a, prog_bar=False, batch_size=B, sync_dist=True)
        self.log("val_loss_t", loss_t, prog_bar=False, batch_size=B, sync_dist=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, filenames, segment_idxs, n_segments_batch = batch

        embeddings = self.forward_audio(x).detach().cpu()

        for i in range(len(filenames)):
            self.predict_data[filenames[i]].append(embeddings[i, :])

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._ema_update(self.audio_encoder, self.audio_encoder_target)
        self._ema_update(self.text_encoder, self.text_encoder_target)
        self._ema_update(self.proj_a, self.proj_a_target)
        self._ema_update(self.proj_t, self.proj_t_target)

    def _ema_update(self, online: nn.Module, target: nn.Module) -> None:
        for p_o, p_t in zip(online.parameters(), target.parameters()):
            p_t.data.mul_(self.ema_tau).add_((1 - self.ema_tau) * p_o.data)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
