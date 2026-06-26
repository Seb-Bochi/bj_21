from __future__ import annotations

from pathlib import Path

import bentoml
import numpy as np
from hydra import compose, initialize_config_dir
from onnxruntime import InferenceSession
from pydantic import BaseModel, Field

from src.blackjack_predictor.helpers.export_onnx import ONNX_MODEL_PATH

CONFIGS_PATH = Path(__file__).resolve().parents[3] / "configs"

try:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS_PATH)):
        cfg = compose(config_name="config")
except Exception as exc:
    raise RuntimeError(f"Failed to load Hydra configuration globally: {exc}") from exc


class InferenceRequest(BaseModel):
    """Input payload for blackjack inference."""

    dealt_card_1: int = Field(..., ge=0, le=11, description="Value of player initial card 1")
    dealt_card_2: int = Field(..., ge=0, le=11, description="Value of player initial card 2")
    dealer_card: int = Field(..., ge=0, le=11, description="Value of dealer face-up card")


class InferenceResponse(BaseModel):
    """Prediction payload returned by the specialized API."""

    loss_probability: float
    win_probability: float
    prediction: bool


def ensure_onnx_artifact_exists(onnx_model_path: Path = ONNX_MODEL_PATH) -> Path:
    """Return the ONNX artifact path when it exists.

    Args:
        onnx_model_path: Path to the ONNX artifact.

    Returns:
        Path to the ONNX artifact.

    Raises:
        FileNotFoundError: If the ONNX artifact is missing.
    """

    if onnx_model_path.exists():
        return onnx_model_path

    raise FileNotFoundError(f"Missing ONNX artifact at '{onnx_model_path}'. Run uv run src/blackjack_predictor/train.py to generate it.")


def create_inference_session() -> InferenceSession:
    """Create an ONNX Runtime session for the specialized API."""

    onnx_model_path = ensure_onnx_artifact_exists()

    try:
        return InferenceSession(
            onnx_model_path.as_posix(),
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize ONNX runtime session from '{onnx_model_path}': {exc}") from exc


def compute_probabilities(logits: np.ndarray) -> list[float]:
    """Convert logits into a normalized probability vector."""

    shifted_logits = logits - np.max(logits)
    exp_scores = np.exp(shifted_logits)
    probabilities = exp_scores / np.sum(exp_scores)
    return probabilities.astype(float).tolist()


@bentoml.service(name="blackjack-specialized-api")
class BlackjackSpecializedService:
    """BentoML service that serves blackjack predictions through ONNX Runtime."""

    def __init__(self) -> None:
        """Initialize the runtime session and input metadata."""

        self.session = create_inference_session()
        self.input_name = self.session.get_inputs()[0].name

    def _predict_logits(self, payload: InferenceRequest) -> tuple[list[int], np.ndarray]:
        """Run ONNX inference and return the input features and logits."""

        input_features = [payload.dealt_card_1, payload.dealt_card_2, payload.dealer_card]
        if len(input_features) != cfg.model_config.input_dim:
            raise ValueError(f"Dimension mismatch. Model requires {cfg.model_config.input_dim} features, " f"received {len(input_features)}.")

        input_array = np.asarray([input_features], dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: input_array})
        logits = np.asarray(outputs[0], dtype=np.float32)
        if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != cfg.model_config.output_dim:
            raise RuntimeError(f"Unexpected ONNX output shape: {tuple(logits.shape)}")

        return input_features, logits[0]

    @bentoml.api(route="/predict")
    async def predict(self, payload: InferenceRequest, /) -> InferenceResponse:
        """Run blackjack inference for a flat JSON request payload."""

        return predict_from_payload(self, payload)


def predict_from_payload(service: BlackjackSpecializedService, payload: InferenceRequest) -> InferenceResponse:
    """Run blackjack inference using the specialized ONNX runtime session."""

    input_features, logits = service._predict_logits(payload)
    probabilities = compute_probabilities(logits)
    predicted_class = int(np.argmax(logits))

    response_data = InferenceResponse(
        loss_probability=probabilities[0],
        win_probability=probabilities[1],
        prediction=bool(predicted_class),
    )

    return response_data
