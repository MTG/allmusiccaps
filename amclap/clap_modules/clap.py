import gin

from collections import defaultdict

import torch
from torch import nn
import lightning.pytorch as L


from ..contrastive_losses import (
    DDPSigmoidLoss,
    ChunkedDDPSigmoidLoss,
    InfoNCELoss,
    MultimodalInfoNCELoss,
)


@gin.configurable
class CLAP(L.LightningModule):
    """
    Contrastive Language-Audio Pretraining (CLAP) model.
    inspired in

    """

    max_len_s = 30  # s

    def __init__(
        self,
        audio_encoder: nn.Module,
        train_audio_encoder: bool,
        proj_size: int,
        temp: float,
        lr: float,
        weight_decay: float,
        seed: int,
        text_encoder: nn.Module | None = None,
        train_text_encoder: bool = False,
        t_z_size: int | None = None,
        aggregation_type: str = "mean",
        n_pool_att_heads: int = 8,
        embedding_prompt: str | None = "",
        loss_type: str = "info_nce_multimodal",
        tokenizers_parallelism: bool = False,  # deprecated
    ):
        super(CLAP, self).__init__()

        # global variables
        self.seed = seed

        self.train_audio_encoder = train_audio_encoder
        self.train_text_encoder = train_text_encoder

        self.lr = lr
        self.weight_decay = weight_decay
        self.proj_size = proj_size
        self.temp = temp

        self.aggregation_type = aggregation_type
        self.n_pool_att_heads = n_pool_att_heads

        self.embedding_prompt = embedding_prompt

        self.predict_data = defaultdict(list)

        # Text encoder (optional — None when using pre-computed embeddings)
        if text_encoder is not None:
            self.text_encoder = text_encoder()

            for param in self.text_encoder.model.parameters():
                param.requires_grad = self.train_text_encoder

            if self.train_text_encoder:
                self.text_encoder.train()
            else:
                self.text_encoder.eval()
        else:
            self.text_encoder = None

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
            # OMARRQ, MERT, HTSAT all expose embed_dim
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

        if self.text_encoder is not None:
            dummy_text = ""
            self.t_z_size = self.text_encoder.model.encode(dummy_text).shape[0]
        elif t_z_size is not None:
            self.t_z_size = t_z_size
        else:
            raise ValueError("Either text_encoder or t_z_size must be provided")
        self.proj_t = nn.Linear(self.t_z_size, self.proj_size)

        self.loss_type = loss_type
        if self.loss_type == "ddp_sigmoid_loss":
            self.loss = DDPSigmoidLoss()
        elif self.loss_type == "chunked_ddp_sigmoid_loss":
            self.loss = ChunkedDDPSigmoidLoss()
        elif self.loss_type == "info_nce":
            self.loss = InfoNCELoss(temp=self.temp)
        elif self.loss_type == "info_nce_multimodal":
            self.loss = MultimodalInfoNCELoss(temp=self.temp)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def forward_audio(self, audio, attn_mask=None):
        if isinstance(attn_mask, list) and len(attn_mask) > 0:
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
        # Pre-computed embeddings (e.g., symbolic embeddings) — skip text encoder
        if isinstance(text, torch.Tensor):
            return self.proj_t(text.to(self.device))

        if self.text_encoder is None:
            raise ValueError(
                "text_encoder is None but forward_text received strings. "
                "Provide a text_encoder or pass pre-computed embedding tensors."
            )

        if self.embedding_prompt:
            text = [self.embedding_prompt + t for t in text]

        if self.train_text_encoder and self.training:
            # Use forward pass to get gradients when training
            # encode() doesn't propagate gradients, so we use the model directly
            # TODO: This implementation assumes SentenceTransformer-like interface,
            # we should abstract this to the model container class
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
            # Use encode() for efficiency when not training (no gradients needed)
            x_t = self.text_encoder.model.encode(
                text, convert_to_tensor=True, device=self.device
            )
            x_t = x_t.clone()

        return self.proj_t(x_t)

    def forward(self, batch):
        audio_views, text_views, m = batch

        z_a = self.forward_audio(audio_views[0], m)
        z_t = self.forward_text(text_views[0])

        loss = self.loss(z_a, z_t)

        return z_a, z_t, loss

    def training_step(self, batch, batch_idx):
        _, _, loss = self.forward(batch)
        B = batch[0][0].shape[0]

        self.log("train_loss", loss, prog_bar=True, batch_size=B)
        return loss

    def validation_step(self, batch, batch_idx):
        _, _, loss = self.forward(batch)
        B = batch[0][0].shape[0]

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
