from pathlib import Path

import torch
import typer

from blackjack_predictor.data_.dataset import BlackjackDataset

EXPECTED_SIZE = 815767
EXPECTED_FEATURES = 3


def dataset_statistics(
    data_path: str = "data/processed/blkjckhands_processed.pt",
    report_path: str = "report.md",
) -> None:
    """Compute dataset statistics and write results to a markdown report."""
    dataset = BlackjackDataset(processed_file=data_path)

    num_samples = len(dataset)
    num_features = dataset[0][0].shape[0]

    lines = []
    lines.append("## Dataset Statistics\n")
    lines.append(f"- **Dataset path:** `{data_path}`")
    lines.append(f"- **Number of samples:** {num_samples}")
    lines.append(f"- **Number of features:** {num_features}")

    assert num_samples == EXPECTED_SIZE, f"Dataset size mismatch: expected {EXPECTED_SIZE}, got {num_samples}"
    lines.append(f"- **Size check:** passed ({num_samples} samples)")

    assert (
        num_features == EXPECTED_FEATURES
    ), f"Feature dimension mismatch: expected {EXPECTED_FEATURES}, got {num_features}"
    lines.append(f"- **Feature check:** passed ({num_features} features)")

    labels = torch.tensor(dataset.labels)
    label_distribution = torch.bincount(labels)
    label_names = ["Loss", "Win"]

    lines.append("\n### Label Distribution\n")
    lines.append("| Outcome | Count | % |")
    lines.append("|---------|-------|---|")
    for name, count in zip(label_names, label_distribution):
        lines.append(f"| {name} | {count.item()} | {100 * count.item() / num_samples:.1f}% |")

    X = dataset.X
    feature_names = ["card1", "card2", "dealcard1"]
    lines.append("\n### Feature Statistics\n")
    lines.append("| Feature | Mean | Std | Min | Max |")
    lines.append("|---------|------|-----|-----|-----|")
    for i, name in enumerate(feature_names):
        lines.append(
            f"| {name} | {X[:, i].mean():.3f} | {X[:, i].std():.3f} " f"| {X[:, i].min():.3f} | {X[:, i].max():.3f} |"
        )

    report = "\n".join(lines)
    print(report)
    Path(report_path).write_text(report)
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    typer.run(dataset_statistics)
