import pytest
import torch
from pytorch_lightning import LightningDataModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from blackjack_predictor.models.ffnn import SimpleFNN
from blackjack_predictor.tasks import PredictionTask


class DummyBlackjackDataModule(LightningDataModule):
    """Minimal datamodule for exercising the training loop."""

    def __init__(self, batch_size: int = 4):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        x_dummy = torch.rand(20, 13, dtype=torch.float32)
        y_dummy = torch.randint(0, 2, (20,))
        self.dataset = TensorDataset(x_dummy, y_dummy)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


def test_lightning_training_step():
    """Verify the training loop runs and updates model weights."""

    model = SimpleFNN(input_dim=13, hidden_dim=32, output_dim=2)
    task = PredictionTask(model=model, lr=1e-3)
    dm = DummyBlackjackDataModule(batch_size=4)

    initial_weight = model.hidden.weight.clone()

    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        fast_dev_run=2,
        logger=False,
        enable_checkpointing=False,
    )

    try:
        trainer.fit(task, datamodule=dm)
    except Exception as e:
        pytest.fail(f"PyTorch Lightning training loop crashed during execution step: {e}")

    updated_weight = model.hidden.weight
    assert not torch.equal(initial_weight, updated_weight), "Model training failure: Weights did not update after execution steps."
