"""MAEST audio encoder wrapper for CLAP training."""

import gin
import torch
from torch.nn import Module
from transformers import ASTModel, AutoFeatureExtractor


@gin.configurable
class MAEST(Module):
    """MAEST audio encoder wrapper.

    MAEST is an Audio Spectrogram Transformer trained on Discogs music data.
    - Input: 16kHz audio waveform (converted to mel spectrogram internally)
    - Output: Sequence of embeddings (B, T', 768)

    Reference: https://huggingface.co/mtg-upf/discogs-maest-30s-pw-129e-519l
    """

    def __init__(
        self,
        model_id: str = "mtg-upf/discogs-maest-30s-pw-129e-519l",
        local_files_only: bool = False,
    ):
        super(MAEST, self).__init__()

        # Load custom feature extractor (MAESTFeatureExtractor)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

        # Load base AST model (without classification head) to get hidden states
        self.model = ASTModel.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )

        # MAEST has 768-dim embeddings
        self.embed_dim = 768

        # MAEST expects 16kHz input
        self.sr = 16000

        # Patch size for compatibility with CLAP attention pooling
        # MAEST uses hop_length=256 at 16kHz
        self.patch_size = (1, 256)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """Forward pass through MAEST.

        Args:
            input_values: Audio waveform tensor of shape (B, T) at 16kHz

        Returns:
            Embeddings of shape (B, T', 768)
        """
        # Convert waveforms to mel spectrograms via feature extractor
        device = input_values.device
        audio_list = [a.cpu().numpy() for a in input_values]

        inputs = self.feature_extractor(
            audio_list,
            sampling_rate=self.sr,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get all hidden states
        outputs = self.model(**inputs, output_hidden_states=True)

        # Average over all hidden layers
        all_hidden_states = torch.stack(outputs.hidden_states, dim=0)
        embeddings = all_hidden_states.mean(dim=0)  # (B, T', 768)

        return embeddings
