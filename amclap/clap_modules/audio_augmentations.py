"""Waveform-domain audio augmentations for multiview training."""

import random

import torch


def waveform_augment(
    audio: torch.Tensor,
    noise_std: float = 0.005,
    gain_range: tuple[float, float] = (0.8, 1.2),
) -> torch.Tensor:
    """Apply cheap waveform augmentations: Gaussian noise + gain jitter.

    Args:
        audio: 1D tensor [T].
        noise_std: Std of additive Gaussian noise. 0 disables noise.
        gain_range: (min, max) for uniform random gain multiplier.

    Returns:
        Augmented 1D tensor [T] (new tensor, not in-place).
    """
    audio = audio.clone()

    if noise_std > 0:
        audio = audio + noise_std * torch.randn_like(audio)

    gain_min, gain_max = gain_range
    if gain_min != gain_max:
        gain = random.uniform(gain_min, gain_max)
        audio = audio * gain

    return audio
