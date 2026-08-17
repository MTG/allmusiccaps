"""Tests for waveform-domain audio augmentations."""

import torch

from amclap.clap_modules.audio_augmentations import waveform_augment


class TestWaveformAugment:
    def test_shape_preserved(self):
        audio = torch.randn(24000)
        out = waveform_augment(audio, noise_std=0.005, gain_range=(0.8, 1.2))
        assert out.shape == audio.shape

    def test_not_in_place(self):
        audio = torch.randn(24000)
        audio_copy = audio.clone()
        _ = waveform_augment(audio, noise_std=0.005, gain_range=(0.8, 1.2))
        assert torch.equal(audio, audio_copy)

    def test_noise_changes_signal(self):
        torch.manual_seed(42)
        audio = torch.ones(24000)
        out = waveform_augment(audio, noise_std=0.01, gain_range=(1.0, 1.0))
        assert not torch.equal(audio, out)

    def test_no_noise_when_disabled(self):
        audio = torch.ones(24000)
        out = waveform_augment(audio, noise_std=0.0, gain_range=(1.0, 1.0))
        assert torch.equal(audio, out)

    def test_gain_applied(self):
        torch.manual_seed(42)
        audio = torch.ones(24000)
        out = waveform_augment(audio, noise_std=0.0, gain_range=(0.8, 1.2))
        # Gain should change the signal (uniform random != 1.0)
        assert not torch.equal(audio, out)
        # All values should be scaled by the same factor
        ratios = out / audio
        assert torch.allclose(ratios, ratios[0].expand_as(ratios))

    def test_gain_range_varies(self):
        torch.manual_seed(42)
        audio = torch.ones(24000)
        out1 = waveform_augment(audio, noise_std=0.0, gain_range=(0.5, 1.5))
        out2 = waveform_augment(audio, noise_std=0.0, gain_range=(0.5, 1.5))
        # With different random seeds, gains should differ
        assert not torch.equal(out1, out2)

    def test_half_precision(self):
        audio = torch.randn(24000).half()
        out = waveform_augment(audio, noise_std=0.005, gain_range=(0.8, 1.2))
        assert out.dtype == torch.float16
        assert out.shape == audio.shape
