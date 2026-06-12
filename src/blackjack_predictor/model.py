from torch import nn
import torch


class Model(nn.Module):
    """Just a dummy model to show how to structure your code"""

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(13, 16)
        self.relu = nn.ReLU()
        self.output = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        x = self.relu(x)
        return self.output(x)


if __name__ == "__main__":
    model = Model()
    x = torch.rand(1, 13)
    print(f"Output shape of model: {model(x).shape}")
