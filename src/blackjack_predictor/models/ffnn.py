import torch
import torch.nn as nn

class SimpleFNN(nn.Module):
    """Feed-forward network for blackjack with an extra hidden layer."""

    def __init__(self, input_dim: int = 13, hidden_dim: int = 128, output_dim: int = 2):
        super().__init__()
        
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, 64)
        self.hidden3 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.output = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        x = self.relu(x)
        
        x = self.hidden2(x)
        x = self.relu(x)
        
        x = self.hidden3(x) 
        x = self.relu(x)
        
        return self.output(x)

if __name__ == "__main__":
    # Test the model with your 13-dimensional input
    model = SimpleFNN(input_dim=13, hidden_dim=128)
    x = torch.rand(1, 13)
    print(f"Output shape of model: {model(x).shape}")