"""MagnaTagATune autotagging downstream evaluation.

Phase 1: Extract audio embeddings using a pre-trained SSL model.
Phase 2: Train a multi-label fully-connected probe and evaluate with AUROC/MAP metrics.
"""

import argparse
import json
import math
import os
import shutil
import traceback
import warnings
from pathlib import Path

# Add src directory to path for imports

import gin.torch
import lightning.pytorch as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

torch.set_float32_matmul_precision("medium")
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import (
    MultilabelAUROC,
    MultilabelAveragePrecision,
)

from .. import get_model
from ..data import DATASETS
from ..data.audio_embedding_datamodule import AudioEmbeddingDataModule
from embedding_writer import EmbeddingWriter
from ..nets import NETS

for data_name, data in DATASETS.items():
    gin.external_configurable(data, data_name)

for net_name, net in NETS.items():
    gin.external_configurable(net, net_name)


# --- LR Scheduler ---


class CosineAnnealingWithWarmup(_LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_steps, eta_min, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

        assert self.total_steps > self.warmup_steps, (
            f"total_steps: {self.total_steps} must be greater than "
            f"warmup_steps: {self.warmup_steps}"
        )

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            progress = (self.last_epoch - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return [
                self.eta_min
                + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]


class CosineAnnealingCallback(Callback):
    def __init__(self, total_steps, warmup_steps, eta_min):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min

    def on_train_start(self, trainer, pl_module):
        last_epoch = -1
        if trainer.global_step > 0:
            last_epoch = trainer.global_step

        optimizer = trainer.optimizers[0]
        self.scheduler = CosineAnnealingWithWarmup(
            optimizer,
            self.total_steps,
            self.warmup_steps,
            self.eta_min,
            last_epoch=last_epoch,
        )

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        self.scheduler.step()
        pl_module.log("lr", self.scheduler.get_last_lr()[0])


# --- Probe Model ---


class MultiLabelProbe(L.LightningModule):
    def __init__(
        self,
        in_features: int,
        num_labels: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        lr: float,
        class_names: list[str] | None = None,
        plot_dir: Path | None = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.class_names = class_names
        self.plot_dir = Path(plot_dir) if plot_dir is not None else None

        # Build layers: [Dropout -> Linear -> ReLU] x (num_layers-1) + [Dropout -> Linear]
        layers = []
        current_in = in_features
        for i in range(num_layers):
            out_size = num_labels if i == num_layers - 1 else hidden_size
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(current_in, out_size))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
            current_in = out_size
        self.model = nn.Sequential(*layers)

        self.criterion = nn.BCEWithLogitsLoss()

        # Metrics
        self.train_metrics = nn.ModuleDict(
            {
                "train-AUROC-macro": MultilabelAUROC(
                    num_labels=num_labels, average="macro"
                ),
                "train-MAP-macro": MultilabelAveragePrecision(
                    num_labels=num_labels, average="macro"
                ),
            }
        )
        self.val_metrics = nn.ModuleDict(
            {
                "val-AUROC-macro": MultilabelAUROC(
                    num_labels=num_labels, average="macro"
                ),
                "val-MAP-macro": MultilabelAveragePrecision(
                    num_labels=num_labels, average="macro"
                ),
            }
        )
        self.test_metrics = nn.ModuleDict(
            {
                "test-AUROC-macro": MultilabelAUROC(
                    num_labels=num_labels, average="macro"
                ),
                "test-MAP-macro": MultilabelAveragePrecision(
                    num_labels=num_labels, average="macro"
                ),
                "test-MAP-classwise": MultilabelAveragePrecision(
                    num_labels=num_labels, average=None
                ),
            }
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)

        for name, metric in self.train_metrics.items():
            metric.update(logits, y.int())
            self.log(name, metric, on_step=True, prog_bar=True, batch_size=x.shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log("val_loss", loss, prog_bar=True, batch_size=x.shape[0])

        for _, metric in self.val_metrics.items():
            metric.update(logits, y.int())

    def on_validation_epoch_end(self):
        for name, metric in self.val_metrics.items():
            self.log(name, metric, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log("test_loss", loss, prog_bar=True)

        for _, metric in self.test_metrics.items():
            metric.update(logits, y.int())

    def on_test_epoch_end(self):
        for name, metric in self.test_metrics.items():
            if "classwise" in name and self.class_names is not None:
                metric_value = metric.compute().cpu().numpy()
                values = {
                    class_name: float(value)
                    for class_name, value in zip(self.class_names, metric_value)
                }

                if self.plot_dir:
                    self.plot_dir.mkdir(parents=True, exist_ok=True)
                    classwise_path = self.plot_dir / f"{name}.json"
                    with open(classwise_path, "w") as f:
                        json.dump(values, f, indent=4)
                    print(f"Classwise results saved to {classwise_path}")
            else:
                self.log(name, metric, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# --- Dataset & DataModule ---


class MTTEmbeddingDataset(Dataset):
    """Dataset for loading pre-extracted embeddings and multi-label tags."""

    def __init__(
        self,
        embeddings_dir: Path,
        metadata: pd.DataFrame,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.class_names = metadata.columns.to_list()

        self.embeddings = []
        self.labels = []

        for filename, data in metadata.iterrows():
            stem = Path(filename).stem
            emb_path = self.embeddings_dir / stem[:3] / f"{stem}.pt"

            try:
                emb = torch.load(emb_path, weights_only=True)
                # EmbeddingWriter saves as (D, N_chunks); average over chunks -> (D,)
                emb = torch.mean(emb, dim=1)

                self.embeddings.append(emb)
                self.labels.append(torch.tensor(data.tolist()).float())
            except Exception as e:
                warnings.warn(f"Could not load embedding for {filename}: {e}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class MTTDataModule(L.LightningDataModule):
    """DataModule for MagnaTagATune autotagging with pre-extracted embeddings."""

    def __init__(
        self,
        embeddings_dir: Path,
        annotations_file: Path,
        train_split: Path,
        val_split: Path,
        test_split: Path,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.embeddings_dir = Path(embeddings_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Load ground truth
        metadata = pd.read_csv(annotations_file, sep="\t")
        metadata.set_index("mp3_path", inplace=True)
        metadata.drop(columns=["clip_id"], inplace=True)

        print(f"Loading metadata: {len(metadata)} files")

        # Filter to top-50 tags
        counts = metadata.sum()
        top_50_tags = counts.nlargest(n=50).keys()
        metadata = metadata[top_50_tags]

        # Sort columns alphabetically
        metadata = metadata.reindex(sorted(metadata.columns), axis=1)

        # Remove tracks with no tags in top-50
        metadata = metadata.loc[metadata.sum(axis=1) != 0]

        print(f"After keeping top-50 tags: {len(metadata)} files remaining")

        self.metadata = metadata
        self.class_names = metadata.columns.to_list()
        self.n_labels = len(self.class_names)

        # Load splits
        train_filenames = self._load_split(train_split)
        val_filenames = self._load_split(val_split)
        test_filenames = self._load_split(test_split)

        self.metadata_train = metadata.loc[metadata.index.isin(train_filenames)]
        self.metadata_val = metadata.loc[metadata.index.isin(val_filenames)]
        self.metadata_test = metadata.loc[metadata.index.isin(test_filenames)]

        print(
            f"Splits - train: {len(self.metadata_train)}, "
            f"val: {len(self.metadata_val)}, test: {len(self.metadata_test)}"
        )

        # Get embedding dim from first available embedding
        self.embedding_dim = self._get_embedding_dim()

    def _load_split(self, split_path: Path) -> list[str]:
        return [line.split("\t")[1] for line in np.load(split_path)]

    def _get_embedding_dim(self) -> int:
        for filename in self.metadata.index:
            stem = Path(filename).stem
            emb_path = self.embeddings_dir / stem[:3] / f"{stem}.pt"
            if emb_path.exists():
                emb = torch.load(emb_path, weights_only=True)
                return emb.shape[0] if emb.ndim == 2 else emb.shape[-1]
        raise RuntimeError(f"No embeddings found in {self.embeddings_dir}")

    def setup(self, stage=None):
        self.train_dataset = MTTEmbeddingDataset(
            self.embeddings_dir, self.metadata_train
        )
        self.val_dataset = MTTEmbeddingDataset(self.embeddings_dir, self.metadata_val)
        self.test_dataset = MTTEmbeddingDataset(self.embeddings_dir, self.metadata_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


# --- Extraction (Phase 1) ---


@gin.configurable
def predict(
    config_file: Path,
    dataset_name: str,
    embeddings_dir: Path,
    params: dict,
    overlap_ratio: float,
    output_dir: str = None,
    segment_size: float = 10.0,
    new_freq: int = 24000,
    use_audio_type_token: bool = False,
    ckpt_step: int = None,
    force_recompute: bool = False,
    avg_last_n: int = None,
):
    """Extract embeddings from a pre-trained SSL model."""

    ssl_model_id = Path(config_file).parent.parent.name
    embeddings_dir = Path(embeddings_dir) / ssl_model_id / dataset_name

    if force_recompute and embeddings_dir.exists():
        print(f"force_recompute=True: deleting existing embeddings at {embeddings_dir}")
        shutil.rmtree(embeddings_dir)

    if output_dir:
        log_dir = output_dir
    else:
        log_dir = embeddings_dir

    callbacks = [EmbeddingWriter(embeddings_dir)]
    trainer = L.Trainer(callbacks=callbacks, default_root_dir=log_dir, **params)

    module = get_model(
        config_file=config_file,
        device="cuda",
        weights_only=False,
        ckpt_step=ckpt_step,
        avg_last_n=avg_last_n,
    )
    module.eval()
    module.overlap_ratio = overlap_ratio

    print(f"Using segment size: {segment_size}s at {new_freq}Hz")
    print(f"Use audio type token: {use_audio_type_token}")

    data_module = AudioEmbeddingDataModule(
        n_seconds=segment_size, new_freq=new_freq, embeddings_dir=embeddings_dir
    )

    # Skip extraction if all embeddings already exist
    # data_module.setup(stage="predict")
    if len(data_module.dataset) == 0:
        return ssl_model_id

    trainer.predict(module, datamodule=data_module)

    return ssl_model_id


# --- Probe Training (Phase 2) ---


@gin.configurable
def train_probe(
    config_file: Path,
    annotations_file: Path,
    train_split: Path,
    val_split: Path,
    test_split: Path,
    embeddings_dir: Path,
    dataset_name: str,
    hidden_size: int = 512,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 1e-4,
    batch_size: int = 64,
    num_workers: int = 6,
    train_params: dict = None,
    warmup_steps: int = 2000,
    eta_min: float = 1e-7,
    output_dir: str = None,
):
    """Train and evaluate a multi-label probe on extracted embeddings."""

    ssl_model_id = Path(config_file).parent.parent.name
    embeddings_path = Path(embeddings_dir) / ssl_model_id / dataset_name

    save_dir = output_dir if output_dir else str(embeddings_path)
    os.makedirs(save_dir, exist_ok=True)

    # Create datamodule
    datamodule = MTTDataModule(
        embeddings_dir=embeddings_path,
        annotations_file=annotations_file,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Create probe
    probe = MultiLabelProbe(
        in_features=datamodule.embedding_dim,
        num_labels=datamodule.n_labels,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        class_names=datamodule.class_names,
        plot_dir=save_dir,
    )

    # Callbacks
    max_steps = train_params.get("max_steps", 20000) if train_params else 20000
    cosine_callback = CosineAnnealingCallback(
        total_steps=max_steps,
        warmup_steps=warmup_steps,
        eta_min=eta_min,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath=save_dir,
        filename="best-probe",
    )

    if train_params is None:
        train_params = {}

    trainer = L.Trainer(
        callbacks=[cosine_callback, checkpoint_callback],
        default_root_dir=save_dir,
        **train_params,
    )

    # Train
    trainer.fit(probe, datamodule=datamodule)

    # Test with best checkpoint
    test_results = trainer.test(probe, datamodule=datamodule, ckpt_path="best")

    # Save results
    results = {
        "model_id": ssl_model_id,
        "dataset": dataset_name,
        "embedding_dim": datamodule.embedding_dim,
        "n_labels": datamodule.n_labels,
    }
    if test_results:
        results.update(test_results[0])

    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "train_config",
        type=Path,
        help="Path to the model config of a trained model.",
    )
    parser.add_argument(
        "predict_config",
        type=Path,
        help="Path to the config file of the downstream task.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results",
    )
    parser.add_argument(
        "--segment_size",
        type=float,
        default=10.0,
        help="Audio segment size in seconds",
    )
    parser.add_argument(
        "--new_freq",
        type=int,
        default=24000,
        help="Audio sample rate in Hz",
    )
    parser.add_argument(
        "--use_audio_type_token",
        action="store_true",
        help="Add [MUSIC] prefix to text queries",
    )
    parser.add_argument(
        "--ckpt_step",
        type=int,
        default=None,
        help="Load checkpoint at this specific training step",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Delete existing embeddings and recompute from scratch",
    )
    parser.add_argument(
        "--avg_last_n",
        type=int,
        default=None,
        help="Average the last N checkpoints",
    )

    args = parser.parse_args()

    try:
        gin.parse_config_file(args.predict_config, skip_unknown=True)

        # Phase 1: Extract embeddings
        print("=" * 50)
        print("Phase 1: Extracting embeddings")
        print("=" * 50)

        ssl_model_id = predict(
            config_file=args.train_config,
            output_dir=args.output_dir,
            segment_size=args.segment_size,
            new_freq=args.new_freq,
            use_audio_type_token=args.use_audio_type_token,
            ckpt_step=args.ckpt_step,
            force_recompute=args.force_recompute,
            avg_last_n=args.avg_last_n,
        )

        print("Embedding extraction completed!")

        # Phase 2: Train and evaluate probe
        print("=" * 50)
        print("Phase 2: Training probe")
        print("=" * 50)

        train_probe(
            config_file=args.train_config,
            output_dir=args.output_dir,
        )

        print("Probe training and evaluation completed!")

    except Exception:
        traceback.print_exc()
