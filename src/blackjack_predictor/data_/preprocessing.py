from pathlib import Path
import torch
import pandas as pd


def preprocess(data_path: str, output_folder: Path, target_column: str = "winloss") -> None:
    """Blackjack dataset for win/loss prediction."""

    data_path = Path(data_path)

    df = pd.read_csv(data_path)

    if df.columns[0] == "":
        df = df.drop(df.columns[0], axis=1)

    feature_columns = [
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "sumofcards",
        "dealcard1",
        "dealcard2",
        "dealcard3",
        "dealcard4",
        "dealcard5",
        "sumofdeal",
        "ply2cardsum",
    ]

    df = df[feature_columns + [target_column]].copy()

    label_map = {"Loss": 0, "Win": 1}
    df = df[df[target_column].isin(label_map)].copy()

    X = torch.tensor(df[feature_columns].values, dtype=torch.float32)
    y = torch.tensor(df[target_column].map(label_map).values, dtype=torch.long)

    output_path = Path(output_folder)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"X": X, "y": y}, output_path)
    print(f"Saved {len(y)} samples to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Blackjack dataset.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw/blkjckhands.csv",
        help="Path to the raw CSV data file.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/processed/blkjckhands_processed.pt",
        help="Path to save the processed tensor file.",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="winloss",
        help="Name of the target column.",
    )

    args = parser.parse_args()
    preprocess(args.data_path, args.output_folder, args.target_column)
