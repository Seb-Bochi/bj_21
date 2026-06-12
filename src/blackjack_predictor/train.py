from blackjack_predictor.model import Model
from blackjack_predictor.data import MyDataset
from torch import nn, optim

def train():
    dataset = MyDataset("data/raw")
    model = Model()
    # add rest of your training code here
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003)

    epochs = 5
    for _ in range(epochs):
        running_loss = 0
        # states consits of [player_total, dealer_upcard, usable_ace]
        # actions consists of [hit, stand]
        for states, actions in trainloader:
            predictions = model(states)

            loss = criterion(predictions, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            runnig_loss += loss.item()
        else:
            print("fTraining loss: {runnig_loss / len(trainloader)}")

if __name__ == "__main__":
    train()
