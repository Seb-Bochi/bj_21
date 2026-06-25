import time
import torch
import torch.nn.utils.prune as prune
from pathlib import Path
from blackjack_predictor.models.ffnn import SimpleFNN

MODEL_PATH = Path("models/model.pth")


def prune_and_compare(model_path: Path = MODEL_PATH, amount: float = 0.3, n_runs: int = 1000) -> None:
    """Load a saved model, prune it, save it, and compare inference speed.

    NOTE: Pruning is not relevant for this model. SimpleFNN is a 3-layer network with
    at most 128 neurons — already runs in microseconds. Pruning zeros weights but does
    not reduce computation (PyTorch still runs full matrix multiplications). Meaningful
    speedup requires large models (ResNets, transformers) or structured pruning that
    removes entire neurons. Kept here as an empirical demonstration.
    """
    model = SimpleFNN(input_dim=3, hidden_dim=128, output_dim=2)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    parameters_to_prune = (
        (model.hidden, "weight"),
        (model.hidden2, "weight"),
        (model.hidden3, "weight"),
        (model.output, "weight"),
    )
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    for module, _ in parameters_to_prune:
        prune.remove(module, "weight")

    pruned_path = model_path.parent / (model_path.stem + "_pruned.pth")
    torch.save(model.state_dict(), pruned_path)
    print(f"Saved pruned model to {pruned_path}")

    original_model = SimpleFNN(input_dim=3, hidden_dim=128, output_dim=2)
    original_model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    original_model.eval()

    batch_sizes = [1, 32, 256, 1024]

    print(f"\n--- Speed comparison ({n_runs} forward passes per batch size) ---")
    print(f"{'Batch':>6} | {'Original (ms)':>14} | {'Pruned (ms)':>12} | {'Speedup':>8}")
    print("-" * 50)

    with torch.no_grad():
        for batch in batch_sizes:
            x = torch.rand(batch, 3)

            start = time.perf_counter()
            for _ in range(n_runs):
                original_model(x)
            original_time = time.perf_counter() - start

            start = time.perf_counter()
            for _ in range(n_runs):
                model(x)
            pruned_time = time.perf_counter() - start

            print(
                f"{batch:>6} | {original_time / n_runs * 1000:>14.4f} | "
                f"{pruned_time / n_runs * 1000:>12.4f} | {original_time / pruned_time:>7.2f}x"
            )

    print(f"\nSparsity: {amount * 100:.0f}% of weights zeroed")


if __name__ == "__main__":
    prune_and_compare()
