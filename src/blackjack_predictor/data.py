from pathlib import Path
import os
from random import Random

import gdown
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset


CSV_FILENAME = "blkjckhands.csv"
GDRIVE_URL_ENV = "BLACKJACK_GDRIVE_URL"
GDRIVE_FILE_ID_ENV = "BLACKJACK_GDRIVE_FILE_ID"
TEST_SIZE = 0.2
SPLIT_SEED = 42


class MyDataset(Dataset):
    """Blackjack dataset for win/loss prediction."""

    def __init__(self, data_path: Path, target_column: str = "winloss") -> None:
        self.data_path = Path(data_path)
        self.target_column = target_column

        csv_path = self._resolve_csv_path()
        self.df = pd.read_csv(csv_path)

        if self.df.columns[0] == "":
            self.df = self.df.drop(self.df.columns[0], axis=1)

        self.feature_columns = [
            "card1",
            "card2",
            "card3",
            "card4",
            "card5",
            "sumofcards",
            "dealcard1",
            "dealcard2",
            "dealcard3",
            "dealcard4",
            "dealcard5",
            "sumofdeal",
            "ply2cardsum",
        ]

        self.df = self.df[self.feature_columns + [self.target_column]].copy()

        self.label_map = {"Loss": 0, "Win": 1}
        self.df = self.df[self.df[self.target_column].isin(self.label_map)].copy()

        self.features = torch.tensor(self.df[self.feature_columns].values, dtype=torch.float32)
        self.labels = torch.tensor(
            self.df[self.target_column].map(self.label_map).values,
            dtype=torch.long,
        )

    def _resolve_csv_path(self) -> Path:
        """Find or download the raw CSV from disk or Google Drive."""

        candidates = [
            self.data_path / CSV_FILENAME,
            self.data_path.parent / CSV_FILENAME,
            self.data_path.parent.parent / CSV_FILENAME,
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return self._download_csv()

    def _download_csv(self) -> Path:
        """Download the CSV from Google Drive when it is not already present."""

        output_path = self.data_path / CSV_FILENAME
        output_path.parent.mkdir(parents=True, exist_ok=True)

        drive_url = os.getenv(GDRIVE_URL_ENV)
        drive_file_id = os.getenv(GDRIVE_FILE_ID_ENV)

        if drive_url:
            gdown.download(drive_url, output=str(output_path), quiet=False, fuzzy=True)
        elif drive_file_id:
            gdown.download(
                f"https://drive.google.com/uc?id={drive_file_id}",
                output=str(output_path),
                quiet=False,
                fuzzy=True,
            )
        else:
            raise FileNotFoundError(
                f"Could not find {CSV_FILENAME} in data/raw or the repository root, "
                f"and no Google Drive source was configured. Set {GDRIVE_URL_ENV} or {GDRIVE_FILE_ID_ENV}."
            )

        if not output_path.exists():
            raise FileNotFoundError(f"Google Drive download did not create {output_path}.")

        return output_path

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]

    def preprocess(self, output_folder: Path) -> None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(output_folder / "processed.csv", index=False)


def split_dataset(dataset: MyDataset, test_size: float = TEST_SIZE, seed: int = SPLIT_SEED) -> tuple[Subset, Subset]:
    """Split a dataset into deterministic train and test subsets."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    if len(dataset) < 2:
        raise ValueError("dataset must contain at least 2 rows to create a train/test split")

    rng = Random(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []

    label_to_indices: dict[int, list[int]] = {}
    for index, label in enumerate(dataset.labels.tolist()):
        label_to_indices.setdefault(int(label), []).append(index)

    for indices in label_to_indices.values():
        shuffled_indices = indices.copy()
        rng.shuffle(shuffled_indices)

        split_point = int(round(len(shuffled_indices) * test_size))
        if len(shuffled_indices) > 1:
            split_point = max(1, min(len(shuffled_indices) - 1, split_point))
        else:
            split_point = 0

        test_indices.extend(shuffled_indices[:split_point])
        train_indices.extend(shuffled_indices[split_point:])

    if not train_indices or not test_indices:
        all_indices = list(range(len(dataset)))
        rng.shuffle(all_indices)
        split_point = max(1, min(len(all_indices) - 1, int(round(len(all_indices) * (1 - test_size)))))
        train_indices = all_indices[:split_point]
        test_indices = all_indices[split_point:]

    return Subset(dataset, train_indices), Subset(dataset, test_indices)



def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    import typer

    typer.run(preprocess)