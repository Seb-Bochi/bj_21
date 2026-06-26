import os
from typing import Annotated, Optional

import typer
import wandb


def link_model(
    artifact_path: Optional[str] = typer.Argument(default=None),
    aliases: Annotated[list[str], typer.Option("--aliases")] = ["staging"],
) -> None:
    """Add aliases to a model already in the W&B registry.

    If artifact_path is not provided, looks up the latest 'staging' artifact
    from the registry using WANDB_ORG and WANDB_COLLECTION env vars.
    """
    wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=True)
    api = wandb.Api()

    if not artifact_path:
        org = os.getenv("WANDB_ORG")
        collection = os.getenv("WANDB_COLLECTION", "model_worflow")
        registry = os.getenv("WANDB_REGISTRY", "wandb-registry-models")
        artifact_path = f"{org}/{registry}/{collection}:staging"
        typer.echo(f"No artifact path provided, using: {artifact_path}")

    artifact = api.artifact(artifact_path)

    for alias in aliases:
        if alias not in artifact.aliases:
            artifact.aliases.append(alias)
    artifact.save()

    typer.echo(f"Added aliases {aliases} to {artifact_path}")


if __name__ == "__main__":
    typer.run(link_model)
