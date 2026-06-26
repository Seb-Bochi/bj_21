FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PORT=8080

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY src src/
COPY configs configs/
COPY README.md README.md
COPY LICENSE LICENSE
COPY models models/

RUN uv sync --frozen

ENV PATH="/.venv/bin:$PATH"
ENV PYTHONPATH="/src"

CMD ["sh", "-c", "uvicorn blackjack_predictor.api.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
