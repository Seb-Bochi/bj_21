import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from blackjack_predictor.data_ import BlackjackDataModule
from pathlib import Path
from blackjack_predictor.tasks import PredictionTask


MODEL_PATH = Path("models/model.pth")


def train(base_model: torch.nn.Module) -> None:
    """Trains the given model using PyTorch Lightning."""
    
    processed_data_path = Path("data/processed/blkjckhands_processed.csv")

    dm = BlackjackDataModule(data_path=processed_data_path, batch_size=32)
    task = PredictionTask(model=base_model, lr=0.003)
 
    trainer = Trainer(max_epochs=5)

    # 4. Execute the training run
    trainer.fit(task, datamodule=dm)


    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    from blackjack_predictor.models.ffnn import SimpleFNN
    model = SimpleFNN()
    train(base_model=model)