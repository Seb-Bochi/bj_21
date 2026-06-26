from pathlib import Path

import numpy as np
import onnxruntime as rt
import torch

from blackjack_predictor.models.ffnn import SimpleFNN

MODEL_PATH = Path("models/model.pth")
ONNX_MODEL_PATH = Path("models/model.onnx")


def check_onnx_model(
    onnx_model_file: str,
    pytorch_model: torch.nn.Module,
    random_input: torch.Tensor,
    rtol: float = 1e-03,
    atol: float = 1e-05,
) -> None:
    """Assert that ONNX Runtime and PyTorch produce aligned outputs."""

    ort_session = rt.InferenceSession(onnx_model_file, providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: random_input.detach().cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    pytorch_model.eval()
    with torch.no_grad():
        pytorch_outs = pytorch_model(random_input).detach().cpu().numpy()

    assert ort_outs[0].shape == pytorch_outs.shape
    assert np.allclose(ort_outs[0], pytorch_outs, rtol=rtol, atol=atol)


def test_onnx_model_matches_pytorch_model() -> None:
    """Ensure the committed ONNX artifact stays aligned with the PyTorch model."""

    assert MODEL_PATH.exists(), f"Expected PyTorch weights at '{MODEL_PATH}'"
    assert ONNX_MODEL_PATH.exists(), f"Expected ONNX artifact at '{ONNX_MODEL_PATH}'"

    model = SimpleFNN(input_dim=3, hidden_dim=128, output_dim=2)
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state_dict)

    torch.manual_seed(0)
    test_input = torch.randint(low=0, high=12, size=(8, 3), dtype=torch.int64).to(torch.float32)

    check_onnx_model(ONNX_MODEL_PATH.as_posix(), model, test_input)
