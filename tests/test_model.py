import torch

from blackjack_predictor.model import Model


def test_model_forward_shape():
	"""Test that the model returns a two-class output."""
	model = Model()
	output = model(torch.rand(4, 13))

	assert output.shape == (4, 2)
