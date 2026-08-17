import gin
import re

from pathlib import Path

import torch
import lightning.pytorch as L
from torch import nn
from huggingface_hub import hf_hub_download

from . import clap_modules
from .def_module import def_module, ensure_registered


def _register_referenced_modules(config_str: str) -> None:
    """Register with gin every module the config selects by name.

    Only the modules the published models use are registered at import time
    (see `def_module`), so a config selecting another variant -- LeJEPA,
    for instance -- needs it registered before the config is parsed.
    """
    referenced = [
        name
        for name in clap_modules.MODULES
        if f"@{name}" in config_str or f"{name}." in config_str
    ]
    if referenced:
        ensure_registered(*referenced)


def _average_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Average multiple state dicts element-wise."""
    n = len(state_dicts)
    avg = {}
    for key in state_dicts[0]:
        tensors = [sd[key] for sd in state_dicts]
        if tensors[0].is_floating_point():
            avg[key] = sum(tensors) / n
        else:
            # Non-float tensors (e.g. batch norm num_batches_tracked): take from last
            avg[key] = tensors[-1]
    return avg


def get_model(
    model_id: str | None = None,
    config_file: Path | str | None = None,
    device: str = "cpu",
    weights_only: bool = True,
    ckpt_step: int | None = None,
    avg_last_n: int | None = None,
) -> L.LightningModule:
    """Returns an OMAR-RQ Module from the provided  model_id or config_file.

    Args:
        model_id (str): Hugging Face's Model ID or local path to the model
        config_file (Path): Path to the model config of a trained model.
        device (str): Device to use for the model. Defaults to "cpu".
        quantization_targets (bool): If True, it will create the quantization
            targets for SSL pre-training of the model. Defaults to False.

    Output:
        module: The model from the provided config file.


    Module usage:

    Args:
        audio (torch.Tensor): 2D mono audio tensor (B, T'). Where B is
            the batch size and T' is the number of samples.
        layers (set): Set of layer indices to extract embeddings from.
            By default, it extracts embeddings from the last layer (logits).

    Output:
        torch.Tensor: Extracted embeddings. The output tensor has shape
            (L, B, T, C,) where L = len(layers), B is the batch size, T is
            the number of output timestamps, and C = embedding dimension.


    Example:

    >>> x = torch.randn(1, 16000 * 4).cpu()
    >>>
    >>> model = get_model(config_file, device="cpu")
    >>>
    >>> embeddings = model.extract_embeddings(x, layers=(6))
    >>>
    >>> # use the `eps` field to compute timestamps
    >>> timestamps = torch.arange(embeddings.shape[2]) / model.eps



    >> NOTE: The model's embedding rate depends on the model's configuration.
        For example, the melspectrogram model has an embedding rate of 16ms.
        audio should be a sequence with a sample rate as inditacted in the
        config file and up to 30s.
    """

    if not config_file != model_id:
        raise ValueError("Provide either a model_id or a config_file, not both.")

    if avg_last_n is not None and ckpt_step is not None:
        raise ValueError("Cannot use both avg_last_n and ckpt_step.")

    ckpt_path = None
    avg_ckpt_paths = None
    if model_id:
        config_file = hf_hub_download(repo_id=model_id, filename="config.gin")
        ckpt_path = hf_hub_download(repo_id=model_id, filename="model.ckpt")

    # When no config file is provided, it is assumed that an external
    # gin-config file with all the required fileds has already been parsed.
    # Don't try to moddify the gin configuration nor load a checkpoint.
    if config_file != "":
        # Start from a clean gin state. Bindings are global and persist, so
        # loading a second model in the same process would otherwise inherit the
        # first model's parameters -- including `local_files_only`, which makes
        # the second load fail offline against a cache it never populated.
        with gin.unlock_config():
            gin.clear_config()

        # Read and preprocess the config to resolve ambiguous selectors
        with open(config_file) as f:
            config_str = f.read()

        # Replace ambiguous selectors with fully qualified names
        # Handle both parameter prefixes (Class.) and configurable references (@Class)
        ambiguous_selectors = {
            "AudioDataModule": "data.data_utils.AudioDataModule",
            "AudioDataset": "data.data_utils.AudioDataset",
            "discotube_text_audio": "data.discotube_text_audio.discotube_text_audio",
        }
        for short_name, full_name in ambiguous_selectors.items():
            # Replace parameter prefix style: ClassName.param
            config_str = config_str.replace(f"{short_name}.", f"{full_name}.")
            # Replace configurable reference style: @ClassName
            config_str = config_str.replace(f"@{short_name}", f"@{full_name}")

        # Register any module this config selects before parsing. Parsing uses
        # skip_unknown=True, so an unregistered name would be dropped silently
        # and the model would come back partly unconfigured.
        _register_referenced_modules(config_str)

        # Parse the gin config
        with gin.unlock_config():
            gin.parse_config(config_str, skip_unknown=True)

        def _extract_step(p):
            m = re.search(r"step=(\d+)", p.name)
            return int(m.group(1)) if m else -1

        # look for the ckpt file in folder
        if avg_last_n is not None:
            ckpts = list(Path(config_file).parent.glob("*.ckpt"))
            if not ckpts:
                raise FileNotFoundError(
                    f"No checkpoints found in {Path(config_file).parent}"
                )
            ckpts_sorted = sorted(ckpts, key=_extract_step)
            if len(ckpts_sorted) < avg_last_n:
                raise ValueError(
                    f"Requested avg_last_n={avg_last_n} but only "
                    f"{len(ckpts_sorted)} checkpoints found"
                )
            avg_ckpt_paths = ckpts_sorted[-avg_last_n:]
            ckpt_path = avg_ckpt_paths[-1]  # latest, used to instantiate the model
            steps = [_extract_step(p) for p in avg_ckpt_paths]
            print(f"Averaging last {avg_last_n} checkpoints (steps {steps})")
        elif ckpt_step is not None:
            matches = list(Path(config_file).parent.glob(f"*-step={ckpt_step}.ckpt"))
            if not matches:
                raise FileNotFoundError(
                    f"No checkpoint found for step={ckpt_step} in {Path(config_file).parent}"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"Multiple checkpoints found for step={ckpt_step}: {matches}"
                )
            ckpt_path = matches[0]
        else:
            ckpts = list(Path(config_file).parent.glob("*.ckpt"))
            if not ckpts:
                raise FileNotFoundError(
                    f"No checkpoints found in {Path(config_file).parent}"
                )
            ckpt_path = max(ckpts, key=_extract_step)

        print(f"Loading checkpoint: {ckpt_path}")

    # get classes of interest
    module = def_module(ckpt_path=ckpt_path, weights_only=weights_only)

    # Apply weight averaging if requested
    if avg_ckpt_paths is not None:
        print("Loading state dicts for weight averaging...")
        state_dicts = []
        for p in avg_ckpt_paths:
            ckpt = torch.load(p, map_location="cpu")
            sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            state_dicts.append(sd)
        avg_sd = _average_state_dicts(state_dicts)
        module.load_state_dict(avg_sd, strict=True)
        print(f"Weight averaging applied ({len(state_dicts)} checkpoints)")

    # Set the model to eval mode
    module.eval()

    # Move the model to the device
    module.to(device)

    return module
