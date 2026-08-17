import gin
from argparse import ArgumentParser
from pathlib import Path
import traceback

from lightning.pytorch.utilities import rank_zero_info
import lightning.pytorch as L
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from .cosineannealingscheduler import CosineAnnealingCallback
from .data import DATASETS
from .clap_modules import MODULES
from .nets import NETS
from .def_module import def_module

from .callbacks import GinConfigSaverCallback

for module_name, module in MODULES.items():
    gin.external_configurable(module, module_name)

for data_name, data in DATASETS.items():
    gin.external_configurable(data, data_name)

for net_name, net in NETS.items():
    gin.external_configurable(net, net_name)

torch.set_float32_matmul_precision("medium")


def gin_config_to_readable_dictionary(gin_config: dict):
    """
    Parses the gin configuration to a dictionary. Useful for logging to e.g. W&B

    Copied from https://github.com/google/gin-config/issues/154

    :param gin_config: the gin's config dictionary. Can be obtained by gin.config._OPERATIVE_CONFIG
    :return: the parsed (mainly: cleaned) dictionary
    """
    data = {}
    for key in gin_config.keys():
        name = key[1].split(".")[-1]
        values = gin_config[key]
        for k, v in values.items():
            data[".".join([name, k])] = v

    return data


def load_hf_checkpoint(model: LightningModule, model_id: str) -> None:
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(repo_id=model_id, filename="model.ckpt")

    state_dict = torch.load(ckpt_path, map_location="cpu")

    try:
        model.load_state_dict(state_dict, strict=True)

    except Exception:
        rank_zero_info("Failed to load state dict strictly. Trying to adapt keys...")

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(("net.", "embedding_layer.")):
                new_k = "audio_encoder.model." + k
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)

    rank_zero_info(f"Loaded Hugging Face checkpoint from {model_id}")


@gin.configurable
def train(
    datamodule: L.LightningDataModule,
    params: dict,
    wandb_params: dict,
    config_path: Path,
    ckpt_path: Path | None = None,
    hf_ckpt: str | None = None,
    ckpt_save_every_n_epochs: int = 1,
    save_every_n_train_steps: int = 20000,
    layerwise_metrics: bool = False,
    info_imbalance: bool = False,
) -> None:
    """Train a model using the given module, datamodule and netitecture"""

    # get the lightning wandb logger wrapper and log the config
    wandb_logger = WandbLogger(**wandb_params)

    # create callbacks
    cosine_annealing_callback = CosineAnnealingCallback(total_steps=params["max_steps"])  # type: ignore[attr-defined]
    config_save_callback = GinConfigSaverCallback(config_path)

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=ckpt_save_every_n_epochs,
    )
    # checkpoint_best_callback = ModelCheckpoint(
    #     every_n_epochs=ckpt_save_every_n_epochs,
    #     monitor="val_loss",
    #     mode="min",
    # )
    checkpoint_every_n_steps_callback = ModelCheckpoint(
        every_n_train_steps=save_every_n_train_steps,
        save_top_k=-1,  # Save all checkpoints, not just the best one
    )
    callbacks = [
        cosine_annealing_callback,
        checkpoint_callback,
        # checkpoint_best_callback,
        checkpoint_every_n_steps_callback,
        config_save_callback,
    ]

    if layerwise_metrics or info_imbalance:
        from amclap.layerwise_metrics import LayerWiseMetricsCallback

        callbacks.append(LayerWiseMetricsCallback(info_imbalance=info_imbalance))

    module = def_module()  # type: ignore[call-arg]
    datamodule = datamodule()  # type: ignore[call-arg]

    if hf_ckpt is not None:
        if ckpt_path is not None:
            rank_zero_info(
                "Both model_id and ckpt_path are provided. Ignoring model_id."
            )
        else:
            load_hf_checkpoint(module, hf_ckpt)

    # create the trainer and fit the model
    trainer = Trainer(logger=wandb_logger, callbacks=callbacks, **params)
    # If a checkpoint is provided, load it and continue training

    gin_config_dict = gin_config_to_readable_dictionary(gin.config._OPERATIVE_CONFIG)
    wandb_logger.log_hyperparams(gin_config_dict)

    trainer.fit(model=module, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = ArgumentParser("Train SSL models using gin config")
    parser.add_argument(
        "train_config",
        type=Path,
        help="Path to the gin config file for training.",
    )

    args = parser.parse_args()

    try:
        gin.parse_config_file(args.train_config, skip_unknown=True)

        train(config_path=args.train_config)  # type: ignore[arg-type]

    except Exception:
        traceback.print_exc()
