"""HTS-AT audio encoder wrapper for CLAP training."""

import gin
import torch
from torch.nn import Module
from transformers import ClapModel, ClapProcessor


@gin.configurable
class HTSAT(Module):
    """HTS-AT audio encoder from LAION CLAP.

    Uses the audio encoder from the LAION CLAP model (unfused variant).
    - Input: 48kHz audio waveform
    - Output: Fixed-size embedding (B, 512)

    Reference: https://huggingface.co/laion/clap-htsat-unfused
    """

    def __init__(
        self,
        model_id: str = "laion/clap-htsat-unfused",
        local_files_only: bool = False,
    ):
        super(HTSAT, self).__init__()

        # Load full CLAP model
        self.clap_model = ClapModel.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )

        # Load processor for audio preprocessing
        self.processor = ClapProcessor.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )

        # HTS-AT produces 512-dim embeddings after projection
        self.embed_dim = 512

        # HTS-AT expects 48kHz input
        self.sr = 48000

        # Patch size placeholder (HTS-AT uses spectrograms internally)
        # With hop_length=480 at 48kHz, we get 100 frames/second
        self.patch_size = (1, 480)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """Forward pass through HTS-AT audio encoder.

        Args:
            input_values: Audio waveform tensor of shape (B, T) at 48kHz

        Returns:
            Embeddings of shape (B, 1, 512) - fixed size per audio clip
        """
        # Process audio through the processor to get spectrogram features
        # The processor expects numpy arrays or list of arrays
        device = input_values.device
        audio_list = [a.cpu().numpy() for a in input_values]

        inputs = self.processor(
            audios=audio_list,
            return_tensors="pt",
            sampling_rate=self.sr,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get audio features using the full model's get_audio_features method
        embeddings = self.clap_model.get_audio_features(**inputs)

        # HTS-AT returns fixed-size embeddings, but CLAP module expects
        # sequence output. We add a dummy time dimension.
        # Shape: (B, 512) -> (B, 1, 512)
        embeddings = embeddings.unsqueeze(1)

        return embeddings
