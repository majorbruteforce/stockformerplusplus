"""
Sliding window dataset for time series forecasting.
Creates sequences of fixed length for model input.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for sliding window time series data.

    Creates sequences of length seq_len from features,
    with target being the log return at the specified horizon.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        seq_len: int = 60,
        include_time: bool = False,
    ):
        """
        Args:
            features: Feature array of shape (n_samples, n_features)
            targets: Target array of shape (n_samples,)
            seq_len: Length of input sequence
            include_time: If True, include normalized time indices
        """
        self.features = torch.FloatTensor(features.astype(np.float32))
        self.targets = torch.FloatTensor(targets.astype(np.float32))
        self.seq_len = seq_len
        self.include_time = include_time

        self.n_samples = len(self.features) - seq_len
        self.n_features = self.features.shape[1]

        if self.n_samples <= 0:
            raise ValueError(f"Not enough samples: {len(self.features)} < seq_len + 1")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sequence and its target.

        Returns:
            X: Input sequence of shape (seq_len, n_features)
            y: Target value
        """
        X = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len]

        if self.include_time:
            time_indices = torch.arange(
                idx, idx + self.seq_len, dtype=torch.float32
            ).unsqueeze(1) / float(self.__len__())
            X = torch.cat([X, time_indices], dim=1)

        return X, y


def create_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    seq_len: int = 60,
    batch_size: int = 32,
    shuffle: bool = False,
    include_time: bool = False,
) -> Tuple[TimeSeriesDataset, torch.Tensor, torch.Tensor]:
    """
    Create sliding window dataset and extract all targets.

    Args:
        features: Feature array
        targets: Target array
        seq_len: Sequence length
        batch_size: Batch size (for tensor creation)
        shuffle: Whether to shuffle (for train set)
        include_time: Include time indices

    Returns:
        dataset: PyTorch Dataset
        all_targets: All targets as tensor (for evaluation)
        dates: Date indices for alignment
    """
    dataset = TimeSeriesDataset(features, targets, seq_len, include_time)

    all_targets = torch.stack([dataset[i][1] for i in range(len(dataset))])

    return dataset, all_targets, None


if __name__ == "__main__":
    features = np.random.randn(500, 14)
    targets = np.random.randn(500)

    dataset = TimeSeriesDataset(features, targets, seq_len=60)

    print(f"Dataset length: {len(dataset)}")
    X, y = dataset[0]
    print(f"Sample X shape: {X.shape}")
    print(f"Sample y shape: {y.shape}")
