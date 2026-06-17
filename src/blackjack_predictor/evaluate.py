from pathlib import Path

import torch
from torch.utils.data import DataLoader

from blackjack_predictor.data_ import BlackjackDataset
from blackjack_predictor.models.ffnn import SimpleFNN


MODEL_PATH = Path("models/model.pth")


def evaluate(model_path: Path = MODEL_PATH, data_path: Path = Path("data/processed/blkjckhands_processed.pt")) -> float:
	"""Load a trained model and report accuracy on the available dataset."""

	dataset = BlackjackDataset(data_path)
	loader = DataLoader(dataset, batch_size=32, shuffle=False)

	model = SimpleFNN()
	if not model_path.exists():
		raise FileNotFoundError(
			f"Could not find trained model at {model_path}. Run training first."
		)

	state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
	model.load_state_dict(state_dict)
	model.eval()

	correct_predictions = 0
	total_predictions = 0

	with torch.no_grad():
		for states, actions in loader:
			outputs = model(states)
			predicted_labels = outputs.argmax(dim=1)
			correct_predictions += (predicted_labels == actions).sum().item()
			total_predictions += actions.size(0)

	accuracy = correct_predictions / total_predictions if total_predictions else 0.0
	print(f"Accuracy: {accuracy:.4f}")
	return accuracy


if __name__ == "__main__":
	evaluate()
