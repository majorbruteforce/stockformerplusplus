"""
Sliding window dataset for time series forecasting.
Creates sequences of fixed length for model input.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        market_features: np.ndarray = None, # <-- NEW
        seq_len: int = 60,
        include_time: bool = False,
    ):
        self.features = torch.FloatTensor(features.astype(np.float32))
        self.targets = torch.FloatTensor(targets.astype(np.float32))
        
        # --- NEW: Initialize market features ---
        if market_features is not None:
            self.market_features = torch.FloatTensor(market_features.astype(np.float32))
        else:
            self.market_features = None
        # ---------------------------------------
        
        self.seq_len = seq_len
        self.include_time = include_time

        self.n_samples = len(self.features) - seq_len
        self.n_features = self.features.shape[1]

        if self.n_samples <= 0:
            raise ValueError(f"Not enough samples: {len(self.features)} < seq_len + 1")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        X = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len]

        if self.include_time:
            time_indices = torch.arange(
                idx, idx + self.seq_len, dtype=torch.float32
            ).unsqueeze(1) / float(self.__len__())
            X = torch.cat([X, time_indices], dim=1)

        # --- NEW: Yield market status vector ---
        if self.market_features is not None:
            m = self.market_features[idx + self.seq_len - 1]
            return X, m, y
        # ---------------------------------------
        return X, y

def create_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    market_features: np.ndarray = None, # <-- NEW
    seq_len: int = 60,
    batch_size: int = 32,
    shuffle: bool = False,
    include_time: bool = False,
) -> Tuple[TimeSeriesDataset, torch.Tensor, torch.Tensor]:
    
    # Pass market_features into the dataset
    dataset = TimeSeriesDataset(features, targets, market_features, seq_len, include_time)

    # Note: dataset[i] now might return 2 or 3 items. Target is always the last item.
    all_targets = torch.stack([dataset[i][-1] for i in range(len(dataset))])

    return dataset, all_targets, None


if __name__ == "__main__":
    features = np.random.randn(500, 14)
    targets = np.random.randn(500)

    dataset = TimeSeriesDataset(features, targets, seq_len=60)

    print(f"Dataset length: {len(dataset)}")
    X, y = dataset[0]
    print(f"Sample X shape: {X.shape}")
    print(f"Sample y shape: {y.shape}")
