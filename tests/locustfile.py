"""Locust load tests for deployed blackjack prediction APIs."""

import os
import random

from locust import HttpUser, between, task

API_TYPE = os.environ.get("API_TYPE", "fastapi").strip().lower()
SUPPORTED_API_TYPES = {"fastapi", "specialized"}

if API_TYPE not in SUPPORTED_API_TYPES:
    raise ValueError(f"Unsupported API_TYPE '{API_TYPE}'. Expected one of: {sorted(SUPPORTED_API_TYPES)}")


def build_predict_payload() -> dict[str, int]:
    """Build a randomized prediction payload."""
    return {
        "dealt_card_1": random.randint(1, 11),
        "dealt_card_2": random.randint(1, 11),
        "dealer_card": random.randint(1, 11),
    }


class BlackjackPredictorUser(HttpUser):
    """Locust user for exercising deployed prediction APIs."""

    wait_time = between(0.1, 1.0)

    @task(4)
    def test_predict_endpoint(self) -> None:
        """Simulate a user requesting a blackjack prediction."""
        with self.client.post("/predict", json=build_predict_payload(), catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Failed with status code: {response.status_code}")

    @task(1)
    def test_health_endpoint(self) -> None:
        """Periodically check the health endpoint when supported."""
        if API_TYPE != "fastapi":
            return
        self.client.get("/health")
