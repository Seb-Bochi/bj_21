
from pathlib import Path
import pandas as pd




def preprocess(data_path: str, output_folder: Path, target_column: str = 'winloss') -> None:
    """Blackjack dataset for win/loss prediction."""

    data_path = Path(data_path)
    target_column = target_column

    df = pd.read_csv(data_path)

    # Remove an empty first column if it exists (e.g., leftover index column)
    if df.columns[0] == "":
        df = df.drop(df.columns[0], axis=1)

    feature_columns = [
        "card1", "card2", "card3", "card4", "card5", "sumofcards",
        "dealcard1", "dealcard2", "dealcard3", "dealcard4", "dealcard5", 
        "sumofdeal", "ply2cardsum",
    ]

    # Filter to relevant columns and map labels
    df = df[feature_columns + [target_column]].copy()

    label_map = {"Loss": 0, "Win": 1}
    df = df[df[target_column].isin(label_map)].copy()

    output_folder = Path(output_folder)
    output_folder.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_folder, index=False)

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
        default="data/processed/blkjckhands_processed.csv",
        help="Path to save the processed CSV file.",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="winloss",
        help="Name of the target column.",
    )

    args = parser.parse_args()
    preprocess(args.data_path, args.output_folder, args.target_column)
