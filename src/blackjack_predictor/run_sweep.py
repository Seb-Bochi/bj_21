from pathlib import Path

import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from blackjack_predictor.data_ import BlackjackDataModule
from blackjack_predictor.helpers.logger import get_logger
from blackjack_predictor.models.ffnn import SimpleFNN
from blackjack_predictor.tasks import PredictionTask

logger = get_logger(__name__)


def sweep_iteration():
    """A single training run executed by the W&B Sweep Agent."""

    wandb.init()
    config = wandb.config
    logger.info(f"Starting sweep iteration with batch_size={config.batch_size}, hidden_dim={config.hidden_dim}, lr={config.lr}")

    dm = BlackjackDataModule(
        data_path=Path("data/processed/blkjckhands_processed.pt"),
        batch_size=config.batch_size,
        split_seed=42,
    )
    dm.setup()

    model = SimpleFNN(
        input_dim=3,
        hidden_dim=config.hidden_dim,
        output_dim=2,
    )

    task = PredictionTask(
        model=model,
        lr=config.lr,
    )

    wandb_logger = WandbLogger()

    trainer = Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        enable_checkpointing=False,
    )

    logger.info("Starting sweep training run")
    trainer.fit(task, datamodule=dm)


sweep_config = {
    "method": "bayes",
    "name": "blackjack-hyperparameter-sweep",
    "metric": {
        "name": "train_loss",
        "goal": "minimize",
    },
    "parameters": {
        "lr": {"min": 0.0001, "max": 0.01},
        "batch_size": {"values": [32, 64, 128]},
        "hidden_dim": {"values": [32, 64, 128, 256]},
    },
}

if __name__ == "__main__":
    logger.info("Initializing W&B sweep configuration")

    sweep_id = wandb.sweep(sweep_config, project="project_dtu_mlops")

    logger.info(f"Sweep ID generated: {sweep_id}")
    logger.info("Starting sweep agent for 3 iterations")

    wandb.agent(sweep_id, function=sweep_iteration, count=3)

    logger.info("Sweep pipeline complete")
