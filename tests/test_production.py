import os
from typing import Annotated
import wandb
import typer


def link_model(
    artifact_path: str,
    aliases: Annotated[list[str], typer.Option("--aliases")] = ["staging"],
) -> None:
    """Add aliases to a model artifact in the W&B registry."""
    if artifact_path == "":
        typer.echo("No artifact path provided. Exiting.")
        return

    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key, relogin=True)

    api = wandb.Api()
    _, _, artifact_name_version = artifact_path.split("/")
    artifact_name, _ = artifact_name_version.split(":")

    artifact = api.artifact(artifact_path)
    target_path = f"{os.getenv('WANDB_ENTITY')}/wandb-registry-model/{artifact_name}"
    artifact.link(target_path=target_path, aliases=aliases)
    artifact.save()
    typer.echo(f"Artifact {artifact_path} linked to {aliases}")


if __name__ == "__main__":
    typer.run(link_model)
