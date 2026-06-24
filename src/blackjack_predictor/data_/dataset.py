import torch
from torch.utils.data import Dataset


class BlackjackDataset(Dataset):
    """Dataset for Blackjack win/loss prediction loading from a processed .pt file."""

    def __init__(self, processed_file="data/processed/blkjckhands_processed.pt", transform=None):
        super().__init__()
        self.transform = transform

        data = torch.load(processed_file, weights_only=True)

        # We removed the [:, [0, 1, 6]] slice because preprocessing.py
        # already filters the tensor down to the exact 3 features we need!
        self.X = data["X"]
        self.y = data["y"]

    @property
    def labels(self):
        return self.y.tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)

        return x, y
