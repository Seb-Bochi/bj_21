import wandb
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from blackjack_predictor.data_ import BlackjackDataModule
from blackjack_predictor.tasks import PredictionTask
from blackjack_predictor.models.ffnn import SimpleFNN


def sweep_iteration():
    """A single training run executed by the W&B Sweep Agent."""
    # Initialize wandb (this pulls the parameters for the current iteration)
    wandb.init()
    config = wandb.config

    # 1. Load Data with the sweeping batch_size
    # Point directly to the exact processed data file
    dm = BlackjackDataModule(
        data_path=Path("data/processed/blkjckhands_processed.pt"),
        batch_size=config.batch_size,
        split_seed=42,
    )
    dm.setup()

    # 2. Build Model with the sweeping hidden_dim
    model = SimpleFNN(
        input_dim=3,
        hidden_dim=config.hidden_dim,
        output_dim=2,
    )

    # 3. Configure Task with the sweeping learning rate
    task = PredictionTask(
        model=model,
        lr=config.lr,
    )

    # 4. Attach WandbLogger explicitly connected to the current sweep run
    wandb_logger = WandbLogger()

    # 5. Trainer configured for fast sweep iterations
    trainer = Trainer(
        max_epochs=3,  # Short run to quickly prove the sweep mechanism works
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        enable_checkpointing=False,  # Saves disk space during multiple sweep runs
    )

    trainer.fit(task, datamodule=dm)


# ==========================================
# SWEEP CONFIGURATION DICTIONARY
# ==========================================
sweep_config = {
    "method": "bayes",  # Uses Bayesian optimization to find the best params
    "name": "blackjack-hyperparameter-sweep",
    "metric": {
        "name": "train_loss",  # Ensure your PredictionTask logs something similar
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
    # Register the sweep with W&B
    sweep_id = wandb.sweep(sweep_config, project="project_dtu_mlops")

    print(f"Sweep ID generated: {sweep_id}")
    print("Starting Sweep Agent (Running 3 iterations for demonstration)...")

    # The agent will automatically call sweep_iteration() 3 times with different parameters
    wandb.agent(sweep_id, function=sweep_iteration, count=3)

    print("Sweep pipeline complete!")
