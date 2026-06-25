import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import Trainer, LightningDataModule
from blackjack_predictor.models.ffnn import SimpleFNN
from blackjack_predictor.tasks import PredictionTask


# A mock minimal DataModule to test the training pipeline without disk dependencies
class DummyBlackjackDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 4):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        # 13 features matching the model's primary dimension, 2 classes for output
        x_dummy = torch.rand(20, 13, dtype=torch.float32)
        y_dummy = torch.randint(0, 2, (20,))
        self.dataset = TensorDataset(x_dummy, y_dummy)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


def test_lightning_training_step():
    """Unit test to ensure that a training step executes successfully,

    loss is computed, and weights can receive backpropagation gradients.
    """
    # 1. Initialize core system modules
    model = SimpleFNN(input_dim=13, hidden_dim=32, output_dim=2)
    task = PredictionTask(model=model, lr=1e-3)
    dm = DummyBlackjackDataModule(batch_size=4)

    # Track initial parameter weight state to verify learning capability
    initial_weight = model.hidden.weight.clone()

    # 2. Configure a fast developer runner to execute isolated sanity check loops
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        fast_dev_run=2,  # Runs exactly 2 batches of training without logging/saving artifact layers
        logger=False,  # Disable wandb requirements during isolated test run loops
        enable_checkpointing=False,
    )

    # 3. Fit pipeline execution
    try:
        trainer.fit(task, datamodule=dm)
    except Exception as e:
        pytest.fail(f"PyTorch Lightning training loop crashed during execution step: {e}")

    # 4. Assert weights modified to ensure optimization took place
    updated_weight = model.hidden.weight
    assert not torch.equal(
        initial_weight, updated_weight
    ), "Model training failure: Weights did not update after execution steps."
