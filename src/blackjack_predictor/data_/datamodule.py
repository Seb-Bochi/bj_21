from pathlib import Path

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from blackjack_predictor.data_.dataset import BlackjackDataset


class BlackjackDataModule(LightningDataModule):
    """DataModule for Blackjack win/loss prediction."""

    def __init__(self, data_path: Path, batch_size: int = 32, split_seed: int = 42, num_workers: int = 4) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.split_seed = split_seed
        self.num_workers = num_workers

        # Initialize dataset attributes
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        """Load data and split into train, val, and test sets."""

        full_dataset = BlackjackDataset(self.data_path)

        total_len = len(full_dataset)
        train_size = int(0.8 * total_len)
        val_size = int(0.1 * total_len)
        test_size = total_len - train_size - val_size

        generator = torch.Generator().manual_seed(self.split_seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )

    # ==================== M29: DISTRIBUTED DATA LOADING ====================
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    # =======================================================================
