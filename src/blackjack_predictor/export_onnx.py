from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from blackjack_predictor.models.ffnn import SimpleFNN

ONNX_MODEL_PATH = Path("models/model.onnx")


def export_onnx_artifact(
    cfg: DictConfig,
    model_path: Path,
    onnx_model_path: Path = ONNX_MODEL_PATH,
) -> Path:
    """Export trained PyTorch weights to an ONNX artifact.

    Args:
        cfg: Hydra configuration with model dimensions.
        model_path: Path to the trained PyTorch weights.
        onnx_model_path: Destination path for the ONNX artifact.

    Returns:
        Path to the exported ONNX artifact.

    Raises:
        FileNotFoundError: If the PyTorch weights are missing.
        RuntimeError: If the ONNX export fails.
    """

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model weights missing at '{model_path}'. Please run training first.")

    model = SimpleFNN(
        input_dim=cfg.model_config.input_dim,
        hidden_dim=cfg.model_config.hidden_dim,
        output_dim=cfg.model_config.output_dim,
    )
    state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.zeros((1, cfg.model_config.input_dim), dtype=torch.float32)
    onnx_model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_model_path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
            opset_version=17,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to export ONNX model to '{onnx_model_path}': {exc}") from exc

    return onnx_model_path


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Export the trained model weights to ONNX."""

    model_path = Path(cfg.data_config.model_path)
    onnx_model_path = export_onnx_artifact(cfg=cfg, model_path=model_path)
    print(f"Exported ONNX model to {onnx_model_path}")


if __name__ == "__main__":
    main()
