import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError

import blackjack_predictor.api as api
import blackjack_predictor.api_specialized as api_specialized


class FakeModel:
    """Deterministic model double for predict endpoint tests."""

    def __init__(self, outputs: torch.Tensor) -> None:
        self.outputs = outputs
        self.called = False

    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.called = True
        return self.outputs


class FakeInput:
    """Minimal ONNX input descriptor for test sessions."""

    def __init__(self, name: str) -> None:
        self.name = name


class FakeSession:
    """Deterministic ONNX Runtime double for specialized API tests."""

    def __init__(self, outputs: np.ndarray) -> None:
        self.outputs = outputs
        self.called = False
        self.last_inputs = None

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput("input")]

    def run(self, output_names, inputs):
        self.called = True
        self.last_inputs = inputs
        return [self.outputs]


def read_log_lines(log_file: Path) -> list[dict]:
    """Parse JSON log entries from the temporary production log file."""
    if not log_file.exists():
        return []

    return [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]


def test_predict_returns_probabilities_and_logs_result(monkeypatch, tmp_path) -> None:
    log_file = tmp_path / "production_logs.jsonl"
    fake_model = FakeModel(torch.tensor([[0.2, 1.4]], dtype=torch.float32))

    with TestClient(api.app) as client:
        monkeypatch.setattr(api, "model", fake_model)
        monkeypatch.setattr(api, "LOG_FILE", log_file)
        response = client.post(
            "/predict",
            json={"dealt_card_1": 10, "dealt_card_2": 7, "dealer_card": 6},
        )

    assert response.status_code == 200

    payload = response.json()
    assert set(payload) == {"loss_probability", "win_probability", "prediction"}
    assert payload["prediction"] is True
    assert 0.0 <= payload["loss_probability"] <= 1.0
    assert 0.0 <= payload["win_probability"] <= 1.0
    assert payload["loss_probability"] + payload["win_probability"] == pytest.approx(1.0)
    assert fake_model.called is True

    log_entries = read_log_lines(log_file)
    assert len(log_entries) == 1
    assert log_entries[0]["input"] == [10, 7, 6]
    assert log_entries[0]["prediction"] == 1


def test_predict_rejects_out_of_range_cards_with_422(monkeypatch, tmp_path) -> None:
    log_file = tmp_path / "production_logs.jsonl"
    fake_model = FakeModel(torch.tensor([[0.2, 1.4]], dtype=torch.float32))

    with TestClient(api.app) as client:
        monkeypatch.setattr(api, "model", fake_model)
        monkeypatch.setattr(api, "LOG_FILE", log_file)
        response = client.post(
            "/predict",
            json={"dealt_card_1": -1, "dealt_card_2": 7, "dealer_card": 6},
        )

    assert response.status_code == 422

    detail = response.json()["detail"]
    assert any(error["loc"][-1] == "dealt_card_1" for error in detail)
    assert fake_model.called is False
    assert read_log_lines(log_file) == []


def test_specialized_predict_returns_probabilities_and_logs_result(monkeypatch, tmp_path) -> None:
    fake_session = FakeSession(np.asarray([[0.2, 1.4]], dtype=np.float32))

    monkeypatch.setattr(api_specialized, "create_inference_session", lambda: fake_session)
    service = api_specialized.BlackjackSpecializedService()
    payload = api_specialized.InferenceRequest(dealt_card_1=10, dealt_card_2=7, dealer_card=6)

    response = asyncio.run(service.predict(payload))

    assert response.model_dump().keys() == {"loss_probability", "win_probability", "prediction"}
    assert response.prediction is True
    assert 0.0 <= response.loss_probability <= 1.0
    assert 0.0 <= response.win_probability <= 1.0
    assert response.loss_probability + response.win_probability == pytest.approx(1.0)
    assert fake_session.called is True
    assert fake_session.last_inputs["input"].tolist() == [[10.0, 7.0, 6.0]]


def test_specialized_predict_rejects_out_of_range_cards_with_422(monkeypatch, tmp_path) -> None:
    log_file = tmp_path / "specialized_production_logs.jsonl"
    fake_session = FakeSession(np.asarray([[0.2, 1.4]], dtype=np.float32))

    monkeypatch.setattr(api_specialized, "LOG_FILE", log_file)
    monkeypatch.setattr(api_specialized, "create_inference_session", lambda: fake_session)
    api_specialized.BlackjackSpecializedService()

    with pytest.raises(ValidationError) as exc_info:
        api_specialized.InferenceRequest(dealt_card_1=-1, dealt_card_2=7, dealer_card=6)

    detail = exc_info.value.errors()
    assert any(error["loc"][-1] == "dealt_card_1" for error in detail)
    assert fake_session.called is False
    assert read_log_lines(log_file) == []


def test_prepare_onnx_artifact_exports_when_missing(monkeypatch, tmp_path) -> None:
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"weights")
    onnx_model_path = tmp_path / "model.onnx"
    export_calls = []

    monkeypatch.setattr(api_specialized.torch, "load", lambda *args, **kwargs: {"weights": "state"})

    def fake_export(model, dummy_input, export_path, **kwargs):
        export_calls.append(
            {
                "shape": tuple(dummy_input.shape),
                "path": Path(export_path),
                "kwargs": kwargs,
            }
        )
        Path(export_path).write_bytes(b"onnx")

    monkeypatch.setattr(api_specialized.torch.onnx, "export", fake_export)

    class FakeFNN:
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.state_dict = None
            self.eval_called = False

        def load_state_dict(self, state_dict) -> None:
            self.state_dict = state_dict

        def eval(self) -> None:
            self.eval_called = True

    monkeypatch.setattr(api_specialized, "SimpleFNN", FakeFNN)

    result = api_specialized.prepare_onnx_artifact(model_path=model_path, onnx_model_path=onnx_model_path)

    assert result == onnx_model_path
    assert onnx_model_path.exists()
    assert len(export_calls) == 1
    assert export_calls[0]["shape"] == (1, api_specialized.cfg.model_config.input_dim)
    assert export_calls[0]["path"] == onnx_model_path


def test_prepare_onnx_artifact_skips_export_when_present(monkeypatch, tmp_path) -> None:
    model_path = tmp_path / "model.pth"
    onnx_model_path = tmp_path / "model.onnx"
    onnx_model_path.write_bytes(b"existing")

    def fail_export(*args, **kwargs):
        raise AssertionError("export should not be called")

    monkeypatch.setattr(api_specialized.torch.onnx, "export", fail_export)

    result = api_specialized.prepare_onnx_artifact(model_path=model_path, onnx_model_path=onnx_model_path)

    assert result == onnx_model_path
