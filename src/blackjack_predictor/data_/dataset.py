import os
from pathlib import Path
import gdown
import pandas as pd
import torch
from torch.utils.data import Dataset


class BlackjackDataset(Dataset):
    """Dataset for Blackjack win/loss prediction loading and processing from a CSV file."""

    def __init__(self, data_dir="data/raw", transform=None):
        super().__init__()
        self.transform = transform

        data_dir = Path(data_dir)
        csv_path = data_dir / "blkjckhands.csv"

        # Walk up the parent directories to see if the file exists at a higher level (like repo root)
        if not csv_path.exists():
            for parent in data_dir.parents:
                fallback = parent / "blkjckhands.csv"
                if fallback.exists():
                    csv_path = fallback
                    break

        # Trigger Google Drive Download if still completely missing
        if not csv_path.exists():
            gdrive_id = os.getenv("BLACKJACK_GDRIVE_FILE_ID")
            if gdrive_id:
                data_dir.mkdir(parents=True, exist_ok=True)
                url = f"https://drive.google.com/uc?id={gdrive_id}"
                gdown.download(url, str(data_dir / "blkjckhands.csv"), quiet=True, fuzzy=True)
                csv_path = data_dir / "blkjckhands.csv"

        # Explicit check ensuring the correct FileNotFoundError triggers for the test match
        if not csv_path.exists():
            raise FileNotFoundError(f"blkjckhands.csv not found at {csv_path}")

        # Load file
        df = pd.read_csv(csv_path)

        # Filter "Push" outcomes
        if "winloss" in df.columns:
            df = df[df["winloss"] != "Push"]

        # Drop non-feature/target columns to arrive at exactly 13 features
        cols_to_drop = [
            "",
            "PlayerNo",
            "winloss",
            "plybustbeat",
            "dlbustbeat",
            "blkjck",
            "sumofcards",
            "sumofdeal",
            "plwinamt",
        ]
        feature_cols = [c for c in df.columns if c not in cols_to_drop]

        # Binary classification mapping
        label_map = {"Win": 1, "Loss": 0}

        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df["winloss"].map(label_map).values, dtype=torch.long)

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

    def preprocess(self, output_folder):
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        processed_df = pd.DataFrame(self.X.numpy())
        processed_df["label"] = self.y.numpy()
        processed_df.to_csv(output_folder / "processed.csv", index=False)
