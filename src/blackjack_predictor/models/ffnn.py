from torch import nn
import torch


class SimpleFNN(nn.Module):
    """Feed-forward network for blackjack win/loss prediction."""

    def __init__(self, input_dim: int = 13, hidden_dim: int = 16, output_dim: int = 2):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        x = self.relu(x)
        return self.output(x)


if __name__ == "__main__":
    model = SimpleFNN()
    x = torch.rand(1, 13)
    print(f"Output shape of model: {model(x).shape}")
