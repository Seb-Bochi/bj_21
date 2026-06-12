from pathlib import Path

import gdown
import pandas as pd
import pytest
from torch.utils.data import Dataset

from blackjack_predictor.data import MyDataset


def _write_sample_csv(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / "blkjckhands.csv"

    pd.DataFrame(
        [
            [0, "Player1", 7, 10, 0, 0, 0, 17, 10, 8, 0, 0, 0, 18, "nowin", "Loss", "Beat", "Dlwin", 0, 10, 17],
            [1, "Player2", 10, 9, 0, 0, 0, 19, 10, 8, 0, 0, 0, 18, "nowin", "Win", "Plwin", "Beat", 20, 0, 19],
            [2, "Player3", 9, 8, 0, 0, 0, 17, 10, 8, 0, 0, 0, 18, "nowin", "Push", "Beat", "Beat", 0, 0, 17],
        ],
        columns=[
            "",
            "PlayerNo",
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
            "blkjck",
            "winloss",
            "plybustbeat",
            "dlbustbeat",
            "plwinamt",
            "dlwinamt",
            "ply2cardsum",
        ],
    ).to_csv(csv_path, index=False)

    return csv_path


def test_my_dataset_loads_and_filters_push(tmp_path):
    """Test that the dataset loads and removes Push outcomes."""
    _write_sample_csv(tmp_path / "data" / "raw")

    dataset = MyDataset(tmp_path / "data" / "raw")
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2

    features, label = dataset[0]
    assert features.shape == (13,)
    assert label.item() in {0, 1}


def test_my_dataset_finds_csv_in_repo_root(tmp_path):
    """Test that the dataset falls back to the repository root CSV path."""
    _write_sample_csv(tmp_path)

    dataset = MyDataset(tmp_path / "data" / "raw")
    assert len(dataset) == 2


def test_preprocess_writes_processed_file(tmp_path):
    """Test that preprocess writes a CSV to the output folder."""
    _write_sample_csv(tmp_path / "data" / "raw")

    dataset = MyDataset(tmp_path / "data" / "raw")
    output_folder = tmp_path / "data" / "processed"

    dataset.preprocess(output_folder)

    processed_file = output_folder / "processed.csv"
    assert processed_file.exists()
    assert pd.read_csv(processed_file).shape[0] == 2


def test_my_dataset_raises_when_csv_is_missing(tmp_path):
    """Test that a missing CSV raises a helpful error."""
    with pytest.raises(FileNotFoundError, match="blkjckhands.csv"):
        MyDataset(tmp_path / "data" / "raw")


def test_my_dataset_downloads_from_google_drive(tmp_path, monkeypatch):
    """Test that the dataset downloads from Google Drive when the CSV is missing."""

    downloaded_rows = []

    def fake_download(url, output, quiet, fuzzy):
        downloaded_rows.append((url, output, quiet, fuzzy))
        _write_sample_csv(Path(output).parent)
        return output

    monkeypatch.setenv("BLACKJACK_GDRIVE_FILE_ID", "fake-file-id")
    monkeypatch.delenv("BLACKJACK_GDRIVE_URL", raising=False)
    monkeypatch.setattr(gdown, "download", fake_download)

    dataset = MyDataset(tmp_path / "data" / "raw")

    assert len(dataset) == 2
    assert downloaded_rows
    assert "fake-file-id" in downloaded_rows[0][0]
