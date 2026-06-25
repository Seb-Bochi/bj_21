import random
from locust import HttpUser, task, between

class BlackjackPredictorUser(HttpUser):
    # Simulate a user waiting between 0.1 and 1 second between requests
    wait_time = between(0.1, 1.0)

    @task(4)
    def test_predict_endpoint(self):
        """Simulate a user requesting a blackjack prediction."""
        payload = {
            "dealt_card_1": random.randint(1, 11),
            "dealt_card_2": random.randint(1, 11),
            "dealer_card": random.randint(1, 11)
        }
        
        # Hit the prediction endpoint
        with self.client.post("/predict", json=payload, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Failed with status code: {response.status_code}")

    @task(1)
    def test_health_endpoint(self):
        """Periodically check the health liveness probe."""
        self.client.get("/health")