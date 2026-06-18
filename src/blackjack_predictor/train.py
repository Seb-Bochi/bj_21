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
        input_dim=cfg.model_config.input_dim,
        hidden_dim=cfg.model_config.hidden_dim,
        output_dim=cfg.model_config.output_dim,
    )

    dm = BlackjackDataModule(
        data_path=cfg.data_config.processed_path,
        batch_size=cfg.training_config.batch_size,
        split_seed=cfg.training_config.split_seed,
    )

    task = PredictionTask(
        model=model,
        lr=cfg.training_config.lr,
        num_classes=cfg.training_config.num_classes,
    )

    wandb_logger = WandbLogger(
        project="project_dtu_mlops",
        config={
            "batch_size": cfg.training_config.batch_size,
            "learning_rate": cfg.training_config.lr,
            "epochs": cfg.training_config.max_epochs,
        },
        group=model.__class__.__name__,
        name=f"{model.__class__.__name__}_train",
        job_type="train",
    )   

    trainer = Trainer(max_epochs=cfg.training_config.max_epochs, logger=wandb_logger)
    trainer.fit(task, datamodule=dm)

    model_path = Path(cfg.data_config.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    train()
