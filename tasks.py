import os
from pathlib import Path

from invoke.context import Context
from invoke.tasks import task

WINDOWS = os.name == "nt"
PROJECT_NAME = "blackjack_predictor"
PYTHON_VERSION = "3.12"


def uv_command() -> str:
    """Return the uv executable path if it exists locally, otherwise fall back to the command name."""

    local_uv = Path.home() / ".local" / "bin" / "uv.exe"
    return str(local_uv) if local_uv.exists() else "uv"


# Project commands
@task
def preprocess_data(ctx):
    """Preprocesses the raw CSV into a 3-column tensor."""
    # We point directly to your new, filtered script
    ctx.run("python src/blackjack_predictor/data_/preprocessing.py")


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"{uv_command()} run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def evaluate(ctx: Context) -> None:
    """Evaluate model accuracy."""
    ctx.run(f"{uv_command()} run src/{PROJECT_NAME}/evaluate.py", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run(f"{uv_command()} run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run(f"{uv_command()} run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def serve_backend(ctx: Context, host: str = "127.0.0.1", port: int = 8000, reload: bool = True) -> None:
    """Serve the sample FastAPI backend."""
    reload_flag = " --reload" if reload else ""
    ctx.run(
        f"{uv_command()} run python -m uvicorn samples.frontend_backend.backend:app " f"--host {host} --port {port}{reload_flag}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def serve_frontend(
    ctx: Context,
    host: str = "127.0.0.1",
    port: int = 8501,
    backend: str = "http://127.0.0.1:8000",
) -> None:
    """Serve the sample Streamlit frontend."""
    ctx.run(
        f"BACKEND={backend} {uv_command()} run streamlit run samples/frontend_backend/frontend.py "
        f"--server.address {host} --server.port {port} --server.headless true --browser.gatherUsageStats false",
        echo=True,
        pty=not WINDOWS,
    )


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS)


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run(f"{uv_command()} run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run(f"{uv_command()} run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
