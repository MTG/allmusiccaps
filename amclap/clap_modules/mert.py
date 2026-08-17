"""MERT audio encoder wrapper for CLAP training."""

import gin
import torch
from torch.nn import Module
from transformers import AutoModel, AutoConfig


@gin.configurable
class MERT(Module):
    """MERT audio encoder wrapper.

    MERT is a HuBERT-based music understanding model.
    - Input: 24kHz audio waveform (for MERT-v1)
    - Output: Sequence of embeddings (B, T, 768)

    Reference: https://huggingface.co/m-a-p/MERT-v1-95M
    """

    def __init__(
        self,
        model_id: str = "m-a-p/MERT-v1-95M",
        local_files_only: bool = False,
    ):
        super(MERT, self).__init__()

        # Load config and add missing attribute for compatibility
        config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        # Fix compatibility issue with transformers >= 4.48
        if not hasattr(config, "conv_pos_batch_norm"):
            config.conv_pos_batch_norm = False

        self.model = AutoModel.from_pretrained(
            model_id,
            config=config,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

        # MERT-v1-95M has 768-dim embeddings
        self.embed_dim = 768

        # MERT-v1 expects 24kHz input
        self.sr = 24000

        # Patch size for compatibility with CLAP module
        # MERT-v1 produces 75 frames per second (320 samples per frame at 24kHz)
        self.patch_size = (1, 320)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """Forward pass through MERT.

        Args:
            input_values: Audio waveform tensor of shape (B, T) at 24kHz

        Returns:
            Embeddings of shape (B, T', 768) where T' = T // 320
        """
        outputs = self.model(input_values, output_hidden_states=True)

        # Average over all hidden layers (13 layers)
        # hidden_states is a tuple of (B, T', 768) tensors
        all_hidden_states = torch.stack(outputs.hidden_states, dim=0)
        embeddings = all_hidden_states.mean(dim=0)

        return embeddings
