import torch

from blackjack_predictor.models.ffnn import SimpleFNN


def test_model_forward_shape():
    """Test that the model returns a two-class output."""
    model = SimpleFNN(input_dim=13, hidden_dim=64, output_dim=2)
    output = model(torch.rand(4, 13))

    assert output.shape == (4, 2)
