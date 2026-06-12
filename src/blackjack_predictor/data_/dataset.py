import torch
import pandas as pd
from torch.utils.data import Dataset

class BlackjackDataset(Dataset):
    """
    Dataset for Blackjack win/loss prediction loading from a processed CSV.
    """
    def __init__(self, processed_file="data/processed/blackjack_processed.csv", transform=None):
        super().__init__()
        self.transform = transform

        df = pd.read_csv(processed_file)
        
        feature_columns = [col for col in df.columns if col != "winloss"]
        
        label_map = {"Loss": 0, "Win": 1}
        
        self.X = torch.tensor(df[feature_columns].values, dtype=torch.float32)
        self.y = torch.tensor(df["winloss"].map(label_map).values, dtype=torch.long)

    @property
    def labels(self):
        """Returns all labels as a standard Python list"""
        return self.y.tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y