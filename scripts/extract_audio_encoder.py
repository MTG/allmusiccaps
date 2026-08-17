"""Extract the audio encoder weights from a CLAP checkpoint into an omar-rq compatible format.

CLAP checkpoints store audio encoder weights with the prefix `audio_encoder.model.`
(e.g., `audio_encoder.model.net.layers.0.ff1.module.0.weight`).

omar_rq.get_model() expects weights with just `net.` and `embedding_layer.` prefixes
(e.g., `net.layers.0.ff1.module.0.weight`).

This script:
1. Loads a CLAP Lightning checkpoint
2. Extracts and renames the audio encoder weights
3. Generates an omar-rq compatible config.gin
4. Saves both to an output directory ready for omar_rq.get_model()

Usage:
    python scripts/extract_audio_encoder.py \\
        --clap_ckpt path/to/clap/checkpoint.ckpt \\
        --clap_config path/to/clap/config.gin \\
        --output_dir path/to/output/
"""

from argparse import ArgumentParser
from pathlib import Path

import torch


AUDIO_ENCODER_PREFIX = "audio_encoder.model."

# Conformer params that need to be carried over to the omar-rq config.
CONFORMER_PARAMS = [
    "embed_dim",
    "depth",
    "conv_kernel_size",
    "num_heads",
    "mlp_ratio",
    "mlp_residual_factor",
    "dropout",
    "input_dropout",
    "use_deepnorm",
    "alpha_deepnorm",
    "beta_deepnorm",
    "use_rope",
    "num_patches",
]

# MaskingModel params to carry over.
MASKING_MODEL_PARAMS = [
    "codebook_dim",
    "codebook_size",
    "diff_input",
    "lr",
    "mask_prob",
    "mask_seconds",
    "num_codebooks",
    "plot_tokens",
    "seed",
    "weight_decay",
]


def extract_audio_encoder_weights(state_dict: dict) -> dict:
    """Extract audio encoder weights, stripping the CLAP prefix.

    Handles two checkpoint formats:
    - Lightning format: keys prefixed with `audio_encoder.model.`
    - Published flat format: keys already starting with `net.` / `embedding_layer.`
    """
    audio_weights = {}

    has_prefix = any(k.startswith(AUDIO_ENCODER_PREFIX) for k in state_dict)

    for key, value in state_dict.items():
        if has_prefix:
            if key.startswith(AUDIO_ENCODER_PREFIX):
                new_key = key[len(AUDIO_ENCODER_PREFIX) :]
                audio_weights[new_key] = value
        else:
            if key.startswith(("net.", "embedding_layer.")):
                audio_weights[key] = value

    return audio_weights


def parse_gin_value(value_str: str) -> str:
    """Parse a gin config value string, handling line continuations."""
    value_str = value_str.strip()

    # Remove trailing comments
    if " #" in value_str:
        value_str = value_str[: value_str.index(" #")].strip()

    return value_str


def extract_gin_params(config_path: Path) -> dict:
    """Extract relevant parameters from a CLAP gin config file."""
    params = {}

    with open(config_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip comments and empty lines
        if not line or line.startswith("#"):
            i += 1
            continue

        # Handle line continuations
        full_line = line
        while full_line.endswith("\\"):
            i += 1
            full_line = full_line[:-1] + lines[i].strip()

        if "=" in full_line:
            key, value = full_line.split("=", 1)
            key = key.strip()
            value = parse_gin_value(value)
            params[key] = value

        i += 1

    return params


def generate_omar_rq_config(gin_params: dict, ckpt_filename: str = "model.ckpt") -> str:
    """Generate an omar-rq compatible config.gin from extracted CLAP config parameters."""
    lines = []

    lines.append("# Parameters for build_module:")
    lines.append("# " + "=" * 78)
    lines.append(f"build_module.ckpt_path = '{ckpt_filename}'")
    lines.append("build_module.module = @modules.maskingmodel.MaskingModel")
    lines.append("build_module.net = @nets.conformer.Conformer")

    # Check if the original config had multi-view representations
    rep_key = next(
        (k for k in gin_params if "build_module.representation" in k),
        None,
    )
    if rep_key and "[" in gin_params[rep_key]:
        # Multi-view: use waveform as the input representation for omar-rq
        lines.append(
            "build_module.representation = "
            "[@nets.melspectrogram.MelSpectrogram,"
            " @nets.waveform.Waveform,"
            " @nets.cqt.CQT,"
            " @nets.encodec.EnCodec]"
        )
    else:
        lines.append(
            "build_module.representation = @nets.melspectrogram.MelSpectrogram"
        )

    lines.append("")
    lines.append("")
    lines.append("# Parameters for Conformer:")
    lines.append("# " + "=" * 78)

    # Extract Conformer params from various possible key formats
    for param in CONFORMER_PARAMS:
        for prefix in [
            "nets.conformer.Conformer",
            "omar_rq.nets.conformer.Conformer",
        ]:
            key = f"{prefix}.{param}"
            if key in gin_params:
                lines.append(f"nets.conformer.Conformer.{param} = {gin_params[key]}")
                break

    lines.append("")
    lines.append("")
    lines.append("# Parameters for MaskingModel:")
    lines.append("# " + "=" * 78)

    for param in MASKING_MODEL_PARAMS:
        for prefix in [
            "modules.maskingmodel.MaskingModel",
            "omar_rq.modules.maskingmodel.MaskingModel",
        ]:
            key = f"{prefix}.{param}"
            if key in gin_params:
                lines.append(
                    f"modules.maskingmodel.MaskingModel.{param} = {gin_params[key]}"
                )
                break

    # Check for input_representation
    for prefix in [
        "modules.maskingmodel.MaskingModel",
        "omar_rq.modules.maskingmodel.MaskingModel",
    ]:
        key = f"{prefix}.input_representation"
        if key in gin_params:
            value = gin_params[key]
            # Normalize the reference
            value = value.replace("@omar_rq.", "@")
            lines.append(
                f"modules.maskingmodel.MaskingModel.input_representation = {value}"
            )
            break

    # Add Waveform params if present (common for multi-view models)
    waveform_params = {}
    for key, value in gin_params.items():
        if "Waveform." in key:
            param = key.split(".")[-1]
            waveform_params[param] = value

    if waveform_params:
        lines.append("")
        lines.append("")
        lines.append("# Parameters for Waveform:")
        lines.append("# " + "=" * 78)
        for param, value in waveform_params.items():
            lines.append(f"nets.waveform.Waveform.{param} = {value}")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = ArgumentParser(description="Extract audio encoder from CLAP checkpoint")
    parser.add_argument(
        "--clap_ckpt",
        type=Path,
        required=True,
        help="Path to the CLAP Lightning checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--clap_config",
        type=Path,
        required=True,
        help="Path to the CLAP gin config file (.gin)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for the omar-rq model (will contain model.ckpt and config.gin)",
    )
    args = parser.parse_args()

    # Load the CLAP checkpoint
    print(f"Loading CLAP checkpoint from {args.clap_ckpt}")
    checkpoint = torch.load(args.clap_ckpt, map_location="cpu")

    # Handle Lightning checkpoint format
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Extract audio encoder weights
    audio_weights = extract_audio_encoder_weights(state_dict)
    if not audio_weights:
        raise RuntimeError(
            "No audio encoder weights found. Expected keys starting with "
            f"'{AUDIO_ENCODER_PREFIX}' or 'net.'/'embedding_layer.'"
            f"Found keys: {list(state_dict.keys())[:10]}..."  # Show a sample of keys for debugging
        )

    net_keys = [k for k in audio_weights if k.startswith("net.")]
    emb_keys = [k for k in audio_weights if k.startswith("embedding_layer.")]
    print(
        f"Extracted {len(net_keys)} net weights and {len(emb_keys)} embedding_layer weights"
    )

    # Parse the CLAP gin config
    print(f"Parsing CLAP config from {args.clap_config}")
    gin_params = extract_gin_params(args.clap_config)

    # Generate omar-rq config
    omar_config = generate_omar_rq_config(gin_params)

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = args.output_dir / "model.ckpt"
    config_path = args.output_dir / "config.gin"

    torch.save(audio_weights, ckpt_path)
    print(f"Saved audio encoder weights to {ckpt_path}")

    with open(config_path, "w") as f:
        f.write(omar_config)
    print(f"Saved omar-rq config to {config_path}")

    print(f"\nTo load with omar-rq:")
    print(f"  from omar_rq import get_model")
    print(f'  model = get_model(config_file="{config_path}")')


if __name__ == "__main__":
    main()
