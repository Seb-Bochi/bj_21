import os
from pathlib import Path

from invoke.context import Context
from invoke.tasks import task

WINDOWS = os.name == "nt"
PROJECT_NAME = "blackjack_predictor"
PYTHON_VERSION = "3.12"
GCP_PROJECT = "dtumlops-499809"
GCP_REGION = "europe-west1"
GCLOUD_DIR = "gcloud/vertex"
CLOUD_BUILD_CONFIG = f"{GCLOUD_DIR}/cloudbuild.yaml"
VERTEX_JOB_CONFIG = f"{GCLOUD_DIR}/blackjack_train_custom_job.yaml"


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


@task
def cloud_build(ctx: Context, project: str = GCP_PROJECT, config: str = CLOUD_BUILD_CONFIG) -> None:
    """Build and push the GCP training/API images with Cloud Build."""

    ctx.run(
        f"gcloud builds submit --project {project} --config {config}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def vertex_train(
    ctx: Context,
    project: str = GCP_PROJECT,
    region: str = GCP_REGION,
    config: str = VERTEX_JOB_CONFIG,
    display_name: str = "blackjack-train",
) -> None:
    """Create a Vertex AI custom training job from the checked-in job config."""

    ctx.run(
        f"gcloud ai custom-jobs create " f"--project {project} " f"--region {region} " f"--display-name {display_name} " f"--config {config}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def cloud_train(
    ctx: Context,
    project: str = GCP_PROJECT,
    region: str = GCP_REGION,
    build_config: str = CLOUD_BUILD_CONFIG,
    vertex_config: str = VERTEX_JOB_CONFIG,
    display_name: str = "blackjack-train",
) -> None:
    """Run Cloud Build and then launch the Vertex AI training job."""

    cloud_build(ctx, project=project, config=build_config)
    vertex_train(ctx, project=project, region=region, config=vertex_config, display_name=display_name)


@task
def vertex_results(
    ctx: Context,
    project: str = GCP_PROJECT,
    region: str = GCP_REGION,
    limit: int = 5,
    job_id: str = "",
    logs: bool = False,
) -> None:
    """List recent Vertex AI jobs, or describe/read logs for a specific job id."""

    if job_id:
        ctx.run(
            f"gcloud ai custom-jobs describe {job_id} --project {project} --region {region}",
            echo=True,
            pty=not WINDOWS,
        )
        if logs:
            ctx.run(
                f'gcloud logging read "resource.type=ml_job AND resource.labels.job_id={job_id}" '
                f"--project {project} --limit 100 --format='value(textPayload)'",
                echo=True,
                pty=not WINDOWS,
            )
        return

    ctx.run(
        f"gcloud ai custom-jobs list --project {project} --region {region} --sort-by=~createTime --limit={limit}",
        echo=True,
        pty=not WINDOWS,
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run(f"{uv_command()} run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run(f"{uv_command()} run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
