import torch
import hydra
from omegaconf import DictConfig
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger

# Replaced BlackjackDataset with your DataModule
from blackjack_predictor.data_ import BlackjackDataModule
from blackjack_predictor.models.ffnn import SimpleFNN


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def evaluate(cfg: DictConfig) -> float:
    """Load a trained model, evaluate on the test split, and log to W&B."""

    model_path = Path(cfg.data_config.model_path)

    # 1. Use the DataModule to guarantee the exact same splits using the config's seed
    dm = BlackjackDataModule(
        data_path=cfg.data_config.processed_path,
        batch_size=cfg.training_config.batch_size,
        split_seed=cfg.training_config.split_seed,
    )

    # Run the standard Lightning setup and grab the appropriate dataloader.
    # Note: If your DataModule doesn't have a test_dataloader, change this to dm.val_dataloader()
    dm.setup(stage="test")
    loader = dm.test_dataloader()

    # 2. Instantiate the model architecture
    model = SimpleFNN(
        input_dim=cfg.model_config.input_dim,
        hidden_dim=cfg.model_config.hidden_dim,
        output_dim=cfg.model_config.output_dim,
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Could not find trained model at {model_path}. Run training first.")

    # 3. Load the trained weights
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Initialize W&B logger
    wandb_logger = WandbLogger(
        project="project_dtu_mlops",
        config={
            "batch_size": cfg.training_config.batch_size,
            "split_seed": cfg.training_config.split_seed,  # Good to log the seed used!
        },
        group=model.__class__.__name__,
        name=f"{model.__class__.__name__}_eval",
        job_type="eval",
    )

    # 5. Run the evaluation loop on the unseen split
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for states, actions in loader:
            outputs = model(states)
            predicted_labels = outputs.argmax(dim=1)
            correct_predictions += (predicted_labels == actions).sum().item()
            total_predictions += actions.size(0)

    accuracy = correct_predictions / total_predictions if total_predictions else 0.0
    print(f"Test Accuracy: {accuracy:.4f}")

    # 6. Log the metric
    wandb_logger.experiment.log({"eval/accuracy": accuracy})
    wandb_logger.experiment.finish()

    return accuracy


if __name__ == "__main__":
    evaluate()
