import torch
import hydra
from omegaconf import DictConfig
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from blackjack_predictor.data_ import BlackjackDataModule
from blackjack_predictor.tasks import PredictionTask
from blackjack_predictor.models.ffnn import SimpleFNN

# ==================== HYDRA DECORATOR ====================
@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """Trains the model using PyTorch Lightning, Hydra, and W&B."""
    
    processed_data_path = Path(cfg.data_config.processed_path)
    model_path = Path(cfg.data_config.model_path)

    # 1. Initialize DataModule (M29 Distributed Data Loading happens inside here)
    dm = BlackjackDataModule(
        data_path=processed_data_path, 
        batch_size=cfg.training_config.batch_size,
        split_seed=cfg.training_config.split_seed
    )
    dm.setup()
    
    # 2. Initialize Model
    base_model = SimpleFNN(
        input_dim=cfg.model_config.input_dim,
        hidden_dim=cfg.model_config.hidden_dim,
        output_dim=cfg.model_config.output_dim
    )
    
    task = PredictionTask(
        model=base_model, 
        lr=cfg.training_config.lr
    )

    # 3. Setup Weights & Biases Logger

    wandb_logger = WandbLogger(
        project="project_dtu_mlops",
        config={
            "batch_size": cfg.training_config.batch_size,
            "learning_rate": cfg.training_config.lr,
            "epochs": cfg.training_config.max_epochs, 
        },
        group=base_model.__class__.__name__,
        name=f"{base_model.__class__.__name__}_train",
        job_type="train",
    )



    trainer = Trainer(
        max_epochs=cfg.training_config.max_epochs, 
        log_every_n_steps=50,
        accelerator="auto",   
        devices="auto",        
        strategy="auto",
        logger=wandb_logger
    )

    if trainer.is_global_zero:
        print(f"Loaded dataset splits: train={len(dm.train_dataset)}", flush=True)
        print("Starting trainer.fit()...", flush=True)
        
    # 5. Train
    trainer.fit(task, datamodule=dm)

    # 6. Save Model (Protected by is_global_zero so GPUs don't overwrite each other)
    if trainer.is_global_zero:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(base_model.state_dict(), model_path)
        print(f"Saved model to {model_path}")

if __name__ == "__main__":
    train()