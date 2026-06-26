from pathlib import Path

import pandas as pd
import pytest
import torch

from blackjack_predictor.data_.preprocessing import preprocess


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


def test_preprocess_filters_push_and_saves_tensor(tmp_path):
    """Verify preprocessing filters Push rows and saves three features."""

    raw_dir = tmp_path / "raw"
    csv_path = _write_sample_csv(raw_dir)
    output_pt = tmp_path / "processed" / "test_processed.pt"

    preprocess(data_path=csv_path, output_path=output_pt)

    assert output_pt.exists(), "Processed .pt file was not created."

    data = torch.load(output_pt, weights_only=True)

    assert len(data["y"]) == 2, "Failed to filter out the 'Push' outcome."

    assert data["X"].shape == (2, 3), f"Expected shape (2, 3), got {data['X'].shape}"

    assert data["y"].tolist() == [0, 1]


def test_preprocess_raises_error_on_missing_csv(tmp_path):
    """Verify preprocessing raises on a missing CSV file."""

    missing_csv = tmp_path / "does_not_exist.csv"
    output_pt = tmp_path / "processed.pt"

    with pytest.raises(FileNotFoundError):
        preprocess(data_path=missing_csv, output_path=output_pt)
