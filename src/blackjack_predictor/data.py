from pathlib import Path
import os

import gdown
import pandas as pd
import torch
from torch.utils.data import Dataset


CSV_FILENAME = "blkjckhands.csv"
GDRIVE_URL_ENV = "BLACKJACK_GDRIVE_URL"
GDRIVE_FILE_ID_ENV = "BLACKJACK_GDRIVE_FILE_ID"


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



def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    import typer

    typer.run(preprocess)