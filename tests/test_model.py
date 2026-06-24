import os
import time
import torch
import pytest
import wandb

from blackjack_predictor.models.ffnn import SimpleFNN


def test_model_forward_shape():
    """Test that the model returns a two-class output."""
    model = SimpleFNN(input_dim=13, hidden_dim=64, output_dim=2)
    output = model(torch.rand(4, 13))

    assert output.shape == (4, 2)


@pytest.fixture
def model_from_registry(tmp_path):
    """Load the model from the W&B registry using the MODEL_NAME env variable."""
    model_name = os.getenv("WANDB_MODEL_NAME")
    if model_name is None:
        pytest.skip("WANDB_MODEL_NAME environment variable not set")

    api_key = os.getenv("WANDB_API_KEY")
    if api_key is None:
        pytest.skip("WANDB_API_KEY environment variable not set")
    wandb.login(key=api_key, relogin=True)
    api = wandb.Api()

    artifact = api.artifact(model_name)
    artifact.download(root=str(tmp_path))
    model_file = next(tmp_path.iterdir())

    model = SimpleFNN(input_dim=13, hidden_dim=128, output_dim=2)
    model.load_state_dict(torch.load(model_file, map_location="cpu", weights_only=True))
    model.eval()
    return model


def test_model_speed(model_from_registry):
    """Test that 100 forward passes on the registry model complete within the time limit."""
    x = torch.rand(1, 13)

    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            model_from_registry(x)
    elapsed = time.time() - start

    assert elapsed < 100, f"Model too slow: {elapsed:.2f}s for 100 forward passes"
