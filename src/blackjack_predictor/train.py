from pathlib import Path

from torch import nn, optim
from torch.utils.data import DataLoader

from blackjack_predictor.data import MyDataset
from blackjack_predictor.model import Model


def train() -> None:
    dataset = MyDataset(Path("data/raw"))
    trainloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003)

    epochs = 5
    for epoch in range(epochs):
        running_loss = 0.0

        for states, actions in trainloader:
            predictions = model(states)
            loss = criterion(predictions, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}: training loss = {running_loss / len(trainloader):.4f}")

if __name__ == "__main__":
    train()