from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import wandb
from blackjack_predictor.data_ import BlackjackDataModule
from blackjack_predictor.models.ffnn import SimpleFNN
from blackjack_predictor.tasks import PredictionTask


def sweep_iteration():
    """A single training run executed by the W&B Sweep Agent."""

    wandb.init()
    config = wandb.config


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
    print("Initializing W&B Sweep Configuration...")

    sweep_id = wandb.sweep(sweep_config, project="project_dtu_mlops")

    print(f"Sweep ID generated: {sweep_id}")
    print("Starting Sweep Agent (Running 3 iterations for demonstration)...")


    wandb.agent(sweep_id, function=sweep_iteration, count=3)

    print("Sweep pipeline complete!")
