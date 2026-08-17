import gin.torch
import warnings

from lightning_utilities.core.rank_zero import rank_zero_info
import torch
import lightning.pytorch as L
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from .data_utils import collate_with_skip


@gin.configurable
class ConcatTextAudioDataModule(L.LightningDataModule):
    # Create a combined datamodule that samples from multiple datasets with specified ratios.
    def __init__(
        self,
        datamodules: list,
        ratios: list,
        num_workers: int = 4,
        batch_size: int = 32,
    ):
        super().__init__()
        assert len(datamodules) == len(ratios), (
            "datamodules and ratios must have same length"
        )
        assert all(0.0 <= r <= 1.0 for r in ratios), "ratios must be between 0 and 1"
        assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"

        self.dms = [dm() for dm in datamodules]
        self.ratios = ratios
        self.num_workers = num_workers
        self.batch_size = batch_size

    def setup(self, stage=None):
        for dm in self.dms:
            dm.setup("")

            train_len = len(dm.train_dataloader().dataset)
            val_len = len(dm.val_dataloader().dataset)

            rank_zero_info(
                f"Setup datamodule: {dm.__class__.__name__} with {train_len} train samples and {val_len} val samples"
            )

        train_datasets = [dm.train_dataloader().dataset for dm in self.dms]
        val_datasets = [dm.val_dataloader().dataset for dm in self.dms]

        self.dataset_train = ConcatDataset(train_datasets)
        self.dataset_val = ConcatDataset(val_datasets)

        ns = [len(ds) for ds in train_datasets]
        weights = np.concatenate(
            [np.full(n, ratio / n) for n, ratio in zip(ns, self.ratios)]
        )

        # we should sample with replacement so that all batches are balanced as required
        # https://discuss.pytorch.org/t/sampling-with-replacement/26474/6
        self.sampler_train = WeightedRandomSampler(
            weights, num_samples=sum(ns), replacement=True
        )

        # Validation sampler
        ns_val = [len(ds) for ds in val_datasets]
        weights_val = np.concatenate(
            [np.full(n, ratio / n) for n, ratio in zip(ns_val, self.ratios)]
        )
        self.sampler_val = WeightedRandomSampler(
            weights_val, num_samples=sum(ns_val), replacement=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_with_skip,
            sampler=self.sampler_train,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_with_skip,
            sampler=self.sampler_val,
            drop_last=True,
        )
