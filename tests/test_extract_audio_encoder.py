from pathlib import Path

import pytest
import torch
from omar_rq import get_model

from scripts.extract_audio_encoder import (
    extract_audio_encoder_weights,
    extract_gin_params,
    generate_omar_rq_config,
)


CLAP_MODEL_ID = "mtg-upf/clap_omarrq_mp_small_music"


@pytest.fixture
def clap_ckpt_and_config():
    """Download and return paths to a CLAP checkpoint and config."""
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(repo_id=CLAP_MODEL_ID, filename="config.gin")
    ckpt_path = hf_hub_download(repo_id=CLAP_MODEL_ID, filename="model.ckpt")
    return Path(ckpt_path), Path(config_path)


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "extracted_model"


def test_extract_weights_from_flat_checkpoint(clap_ckpt_and_config):
    """Test extraction from a checkpoint with flat keys (net.*, embedding_layer.*)."""
    ckpt_path, _ = clap_ckpt_and_config
    state_dict = torch.load(ckpt_path, map_location="cpu")

    audio_weights = extract_audio_encoder_weights(state_dict)

    net_keys = [k for k in audio_weights if k.startswith("net.")]
    emb_keys = [k for k in audio_weights if k.startswith("embedding_layer.")]

    assert len(net_keys) > 0, "Should have net weights"
    assert len(emb_keys) > 0, "Should have embedding_layer weights"

    # No CLAP-specific keys should be present
    assert not any(k.startswith("proj_") for k in audio_weights)
    assert not any(k.startswith("text_encoder") for k in audio_weights)


def test_extract_weights_from_prefixed_checkpoint(clap_ckpt_and_config):
    """Test extraction from a checkpoint with audio_encoder.model.* prefix."""
    ckpt_path, _ = clap_ckpt_and_config
    original = torch.load(ckpt_path, map_location="cpu")

    # Simulate a Lightning CLAP checkpoint with prefixed keys
    prefixed = {}
    for k, v in original.items():
        if k.startswith(("net.", "embedding_layer.")):
            prefixed[f"audio_encoder.model.{k}"] = v
        else:
            prefixed[k] = v

    audio_weights = extract_audio_encoder_weights(prefixed)

    # Keys should have the prefix stripped
    assert all(k.startswith(("net.", "embedding_layer.")) for k in audio_weights)
    assert len(audio_weights) == len(
        [k for k in original if k.startswith(("net.", "embedding_layer."))]
    )


def test_generate_config(clap_ckpt_and_config):
    """Test that the generated config contains the expected parameters."""
    _, config_path = clap_ckpt_and_config
    gin_params = extract_gin_params(config_path)
    config = generate_omar_rq_config(gin_params)

    assert "build_module.ckpt_path" in config
    assert "build_module.net = @nets.conformer.Conformer" in config
    assert "nets.conformer.Conformer.embed_dim" in config
    assert "nets.conformer.Conformer.depth" in config


def test_full_extraction_and_load(clap_ckpt_and_config, output_dir):
    """End-to-end test: extract from CLAP, save, and load with omar_rq.get_model()."""
    ckpt_path, config_path = clap_ckpt_and_config

    # Load and extract
    state_dict = torch.load(ckpt_path, map_location="cpu")
    audio_weights = extract_audio_encoder_weights(state_dict)

    # Generate config
    gin_params = extract_gin_params(config_path)
    omar_config = generate_omar_rq_config(gin_params)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(audio_weights, output_dir / "model.ckpt")
    with open(output_dir / "config.gin", "w") as f:
        f.write(omar_config)

    # Load with omar_rq
    model = get_model(config_file=str(output_dir / "config.gin"))

    assert model is not None
    assert hasattr(model, "extract_embeddings")

    # Test inference
    x = torch.randn(1, 24000 * 4)  # 4 seconds at 24kHz
    embeddings = model.extract_embeddings(x)
    assert embeddings.dim() == 4  # (L, B, T, D)
    assert embeddings.shape[1] == 1  # batch size
    assert embeddings.shape[3] == 512  # embed_dim for small model
