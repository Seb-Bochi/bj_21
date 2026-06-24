import pytest
import torch
from blackjack_predictor.data_.dataset import BlackjackDataset


def test_dataset_loads_pt_file(tmp_path):
    """Test that the dataset correctly loads a processed .pt file."""

    # 1. Create a fake .pt file with 10 rows and 3 columns
    # (Because preprocessing.py already did the filtering!)
    fake_pt_path = tmp_path / "fake_processed.pt"
    fake_X = torch.rand((10, 3))
    fake_y = torch.randint(0, 2, (10,))

    torch.save({"X": fake_X, "y": fake_y}, fake_pt_path)

    # 2. Load it
    dataset = BlackjackDataset(processed_file=fake_pt_path)

    # 3. Assertions
    assert len(dataset) == 10
    x_sample, y_sample = dataset[0]

    # Assert the shape is 3 because the preprocessor ensures it is 3
    assert x_sample.shape[0] == 3
    assert torch.equal(x_sample, fake_X[0])


def test_dataset_missing_file_raises_error(tmp_path):
    """Test that providing a missing file path raises a FileNotFoundError."""
    missing_path = tmp_path / "does_not_exist.pt"

    with pytest.raises(FileNotFoundError):
        BlackjackDataset(processed_file=missing_path)
