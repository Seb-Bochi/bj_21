FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PORT=3000

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project
RUN uv pip install hydra-core

COPY src src/
COPY configs configs/
COPY models models/
COPY README.md README.md
COPY LICENSE LICENSE

RUN uv sync --frozen

EXPOSE 3000

CMD ["sh", "-c", "uv run bentoml serve blackjack_predictor.api.api_specialized:BlackjackSpecializedService --host 0.0.0.0 --port ${PORT}"]
