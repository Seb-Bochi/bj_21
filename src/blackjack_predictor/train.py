import torch
import hydra
from omegaconf import DictConfig
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from blackjack_predictor.data_ import BlackjackDataModule
from blackjack_predictor.tasks import PredictionTask
from blackjack_predictor.models.ffnn import SimpleFNN


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """Trains the model using parameters from Hydra config."""

    model = SimpleFNN(
        input_dim=cfg.model.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        output_dim=cfg.model.output_dim,
    )

    dm = BlackjackDataModule(
        data_path=cfg.data.processed_path,
        batch_size=cfg.training.batch_size,
        split_seed=cfg.training.split_seed,
    )

    task = PredictionTask(
        model=model,
        lr=cfg.training.lr,
        num_classes=cfg.training.num_classes,
    )

    wandb_logger = WandbLogger(
        project="project_dtu_mlops",
        config={
            "batch_size": cfg.training.batch_size,
            "learning_rate": cfg.training.lr,
            "epochs": cfg.training.max_epochs,
        },
        job_type="train",
    )

    trainer = Trainer(max_epochs=cfg.training.max_epochs, logger=wandb_logger)
    trainer.fit(task, datamodule=dm)

    model_path = Path(cfg.data.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    train()
