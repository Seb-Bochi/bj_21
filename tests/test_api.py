import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

import blackjack_predictor.api as api


class FakeModel:
    """Deterministic model double for predict endpoint tests."""

    def __init__(self, outputs: torch.Tensor) -> None:
        self.outputs = outputs
        self.called = False

    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.called = True
        return self.outputs


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
