"""Streamlit frontend for blackjack predictions."""

import os
from typing import Any

import requests
import streamlit as st

from blackjack_predictor.api import app

__all__ = ["app"]


def get_backend_url() -> str:
    """Return the configured backend URL."""
    return os.environ.get("BACKEND", "http://127.0.0.1:8000").rstrip("/")


def predict(
    player_card_1: int,
    player_card_2: int,
    dealer_card: int,
) -> dict[str, Any]:
    """Request a prediction from the backend."""
    response = requests.post(
        f"{get_backend_url()}/predict",
        json={
            "dealt_card_1": player_card_1,
            "dealt_card_2": player_card_2,
            "dealer_card": dealer_card,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    """Render the blackjack prediction interface."""
    st.title("Blackjack Outcome Predictor")
    st.write("Enter the player's two cards and the dealer's visible card.")
    st.caption(
        "The model estimates win or loss from the player's first two cards "
        "and the dealer's visible card. It does not recommend whether to hit or stand."
    )

    player_card_1 = st.selectbox(
        "Player card 1",
        options=range(1, 12),
        format_func=format_card,
    )
    player_card_2 = st.selectbox(
        "Player card 2",
        options=range(1, 12),
        format_func=format_card,
    )
    dealer_card = st.selectbox(
        "Dealer visible card",
        options=range(1, 12),
        format_func=format_card,
    )

    if st.button("Predict outcome", type="primary"):
        try:
            result = predict(player_card_1, player_card_2, dealer_card)
        except requests.RequestException as error:
            st.error(f"Prediction request failed: {error}")
            return

        prediction = "Win" if result["prediction"] else "Loss"
        st.subheader(f"Predicted outcome: {prediction}")

        col1, col2 = st.columns(2)
        col1.metric("Win probability", f"{result['win_probability']:.1%}")
        col2.metric("Loss probability", f"{result['loss_probability']:.1%}")

        st.bar_chart(
            {
                "Loss": result["loss_probability"],
                "Win": result["win_probability"],
            },
            horizontal=True,
        )


def format_card(value: int) -> str:
    """Return a readable card label."""
    if value == 1:
        return "Ace (1)"
    if value == 11:
        return "Ace (11)"
    if value == 10:
        return "10 / face card"
    return str(value)


if __name__ == "__main__":
    main()
