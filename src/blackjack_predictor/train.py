from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from blackjack_predictor.data_ import BlackjackDataModule
from blackjack_predictor.data_.preprocessing import preprocess
from blackjack_predictor.helpers.export_onnx import export_onnx_artifact
from blackjack_predictor.helpers.logger import get_logger
from blackjack_predictor.models.ffnn import SimpleFNN
from blackjack_predictor.tasks import PredictionTask

logger = get_logger(__name__)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """Train the model with PyTorch Lightning, Hydra, and W&B."""

    processed_data_path = Path(cfg.data_config.processed_path)
    raw_data_path = Path(cfg.data_config.raw_path)
    model_path = Path(cfg.data_config.model_path)

    logger.info(f"Starting training with data at {processed_data_path} and output model at {model_path}")

    if not processed_data_path.exists():
        logger.info(f"Processed data missing at {processed_data_path}. Creating it from {raw_data_path}")
        preprocess(raw_data_path, processed_data_path, cfg.data_config.target_column)

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

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        selected_strategy = "ddp"
        selected_devices = num_gpus
        logger.info(f"Using DDP across {num_gpus} GPUs")
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
        logger.info(f"Loaded dataset splits: train={len(dm.train_dataset)}")
        logger.info("Starting trainer.fit()")

    trainer.fit(task, datamodule=dm)

    if trainer.is_global_zero:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(base_model.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")

        onnx_model_path = export_onnx_artifact(cfg=cfg, model_path=model_path)
        logger.info(f"Exported ONNX model to {onnx_model_path}")

        artifact = wandb.Artifact(
            name="blackjack-model",
            type="model",
            description="Trained SimpleFNN for blackjack win/loss prediction",
        )
        artifact.add_file(str(model_path))
        wandb_logger.experiment.log_artifact(artifact)
        logger.info("Logged model artifact to W&B")


if __name__ == "__main__":
    train()
