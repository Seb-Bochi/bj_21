import argparse
from pathlib import Path
from typing import Final, Union

import pandas as pd
import torch

from blackjack_predictor.logger import get_logger

logger = get_logger(__name__)
FEATURE_COLUMNS: Final[list[str]] = ["card1", "card2", "dealcard1"]
LABEL_MAP: Final[dict[str, int]] = {"Loss": 0, "Win": 1}


def preprocess(data_path: Union[str, Path], output_path: Union[str, Path], target_column: str = "winloss") -> None:
    """Convert the raw blackjack CSV into a processed tensor file."""

    data_p = Path(data_path)
    output_p = Path(output_path)

    logger.info(f"Starting preprocessing: {data_p}")

    df = pd.read_csv(data_p)

    if df.columns[0] == "":
        df = df.drop(df.columns[0], axis=1)

    df_filtered = df[FEATURE_COLUMNS + [target_column]].copy()
    df_filtered = df_filtered[df_filtered[target_column].isin(LABEL_MAP.keys())].copy()

    X = torch.tensor(df_filtered[FEATURE_COLUMNS].values, dtype=torch.float32)
    y = torch.tensor(df_filtered[target_column].map(LABEL_MAP).values, dtype=torch.long)

    output_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"X": X, "y": y}, output_p)

    logger.info(f"Success: Saved {len(y)} samples to {output_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Blackjack dataset.")
    parser.add_argument("--data_path", type=str, default="data/raw/blkjckhands.csv", help="Path to the raw CSV data file.")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/processed/blkjckhands_processed.pt",
        help="Path to save the processed tensor file.",
    )
    parser.add_argument("--target_column", type=str, default="winloss", help="Name of the target column.")

    args = parser.parse_args()
    preprocess(args.data_path, args.output_folder, args.target_column)
