import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from blackjack_predictor.data_ import BlackjackDataModule
from blackjack_predictor.tasks import PredictionTask
from blackjack_predictor.models.ffnn import SimpleFNN


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """Trains the model using PyTorch Lightning, Hydra, and W&B."""

    processed_data_path = Path(cfg.data_config.processed_path)
    model_path = Path(cfg.data_config.model_path)

    dm = BlackjackDataModule(
        data_path=processed_data_path,
        batch_size=cfg.training_config.batch_size,
        split_seed=cfg.training_config.split_seed,
    )
    dm.setup()

    base_model = SimpleFNN(
        input_dim=cfg.model_config.input_dim,
        hidden_dim=cfg.model_config.hidden_dim,
        output_dim=cfg.model_config.output_dim,
    )

    task = PredictionTask(
        model=base_model,
        lr=cfg.training_config.lr,
    )

    wandb_logger = WandbLogger(
        project="project_dtu_mlops",
        config=OmegaConf.to_container(cfg, resolve=True),
        group=base_model.__class__.__name__,
        name=f"{base_model.__class__.__name__}_train",
        job_type="train",
    )

    # Detect if multiple GPUs are available to enforce explicit DDP distributed strategy
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        selected_strategy = "ddp"
        selected_devices = num_gpus
        print(f"Explicitly initiating Distributed Training via DDP over {num_gpus} GPUs.")
    else:
        selected_strategy = "auto"
        selected_devices = "auto"

    trainer = Trainer(
        max_epochs=cfg.training_config.max_epochs,
        log_every_n_steps=1,
        accelerator="auto",
        devices=selected_devices,
        strategy=selected_strategy,
        logger=wandb_logger,
    )

    if trainer.is_global_zero:
        print(f"Loaded dataset splits: train={len(dm.train_dataset)}", flush=True)
        print("Starting trainer.fit()...", flush=True)

    trainer.fit(task, datamodule=dm)

    if trainer.is_global_zero:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(base_model.state_dict(), model_path)
        print(f"Saved model to {model_path}")

        artifact = wandb.Artifact(
            name="blackjack-model",
            type="model",
            description="Trained SimpleFNN for blackjack win/loss prediction",
        )
        artifact.add_file(str(model_path))
        wandb_logger.experiment.log_artifact(artifact)
        print("Logged model artifact to W&B")


if __name__ == "__main__":
    train()