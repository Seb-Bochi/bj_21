import pytest
import torch

from blackjack_predictor.data_.dataset import BlackjackDataset


def test_dataset_loads_pt_file(tmp_path):
    """Verify the dataset loads a processed tensor file."""

    fake_pt_path = tmp_path / "fake_processed.pt"
    fake_X = torch.rand((10, 3))
    fake_y = torch.randint(0, 2, (10,))

    torch.save({"X": fake_X, "y": fake_y}, fake_pt_path)

    dataset = BlackjackDataset(processed_file=fake_pt_path)

    assert len(dataset) == 10
    x_sample, y_sample = dataset[0]

    assert x_sample.shape[0] == 3
    assert torch.equal(x_sample, fake_X[0])


def test_dataset_missing_file_raises_error(tmp_path):
    """Verify a missing processed file raises FileNotFoundError."""
    missing_path = tmp_path / "does_not_exist.pt"

    with pytest.raises(FileNotFoundError):
        BlackjackDataset(processed_file=missing_path)
