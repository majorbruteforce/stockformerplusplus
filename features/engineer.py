"""
Feature engineering module for financial time series.
Creates technical indicators and features for model training.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from config import FEATURE_CONFIG


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns: log(price_t / price_{t-1})"""
    return np.log(prices / prices.shift(1))


def compute_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling mean with specified window."""
    return series.rolling(window=window, min_periods=window).mean()


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index.

    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    """Add one-hot encoded day-of-week features."""
    dow = df.index.dayofweek
    dow_dummies = pd.get_dummies(dow, prefix="dow")
    dow_dummies.index = df.index
    for i in range(7):
        col_name = f"dow_{i}"
        if col_name not in dow_dummies.columns:
            dow_dummies[col_name] = 0
    dow_dummies = dow_dummies[[f"dow_{i}" for i in range(7)]]
    return dow_dummies


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all features from raw OHLCV data.

    Features created:
    - open, high, low, close, volume (raw)
    - log_returns (target)
    - 5-day rolling mean
    - 20-day rolling mean
    - 14-day RSI
    - day-of-week (one-hot)
    """
    features = df.copy()

    features["log_return"] = compute_log_returns(features["adjusted_close"])

    features["rolling_mean_5"] = compute_rolling_mean(
        features["adjusted_close"], FEATURE_CONFIG["rolling_mean_short"]
    )

    features["rolling_mean_20"] = compute_rolling_mean(
        features["adjusted_close"], FEATURE_CONFIG["rolling_mean_long"]
    )

    features["rsi_14"] = compute_rsi(
        features["adjusted_close"], FEATURE_CONFIG["rsi_period"]
    )

    dow_features = compute_day_of_week(features)
    for col in dow_features.columns:
        features[col] = dow_features[col]

    feature_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "log_return",
        "rolling_mean_5",
        "rolling_mean_20",
        "rsi_14",
    ] + [f"dow_{i}" for i in range(7)]

    features = features[feature_cols]

    features = features.dropna()

    return features


def create_targets(features: pd.DataFrame, horizons: list = None) -> pd.DataFrame:
    """
    Create target variables for different prediction horizons.

    For horizon h, target is log_return shifted by -h (predicting future return)
    """
    if horizons is None:
        horizons = FEATURE_CONFIG["horizons"]

    targets = pd.DataFrame(index=features.index)

    for h in horizons:
        targets[f"target_h{h}"] = features["log_return"].shift(-h)

    targets = targets.dropna()

    return targets


def split_data(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    train_ratio: float = FEATURE_CONFIG["train_ratio"],
    val_ratio: float = FEATURE_CONFIG["val_ratio"],
    test_ratio: float = FEATURE_CONFIG["test_ratio"],
) -> Tuple[dict, dict]:
    """
    Chronologically split data into train, validation, and test sets.

    Returns:
        Dictionary with train/val/test splits for features and targets
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    n = len(features)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = features.index[:train_end]
    val_idx = features.index[train_end:val_end]
    test_idx = features.index[val_end:]

    aligned_targets = targets.loc[features.index]

    data_splits = {
        "train": {
            "features": features.loc[train_idx],
            "targets": aligned_targets.loc[train_idx],
        },
        "val": {
            "features": features.loc[val_idx],
            "targets": aligned_targets.loc[val_idx],
        },
        "test": {
            "features": features.loc[test_idx],
            "targets": aligned_targets.loc[test_idx],
        },
    }

    scalers = {}
    for split in ["train", "val", "test"]:
        scalers[split] = {}

    feature_cols = [c for c in features.columns if not c.startswith("dow_")]
    scalers["feature"] = StandardScaler()
    scalers["feature"].fit(data_splits["train"]["features"][feature_cols])

    for split in ["train", "val", "test"]:
        split_features = data_splits[split]["features"].copy()

        split_features[feature_cols] = scalers["feature"].transform(
            split_features[feature_cols]
        )

        data_splits[split]["features"] = split_features

    return data_splits, scalers


def prepare_data(df: pd.DataFrame, horizon: int = 1) -> Tuple[dict, dict]:
    """
    Full data preparation pipeline.

    Args:
        df: Raw OHLCV DataFrame
        horizon: Prediction horizon (1 or 5)

    Returns:
        data_splits: Dictionary with train/val/test splits
        scalers: Dictionary with fitted scalers
    """
    features = engineer_features(df)
    targets = create_targets(features, horizons=[horizon])

    common_idx = features.index.intersection(targets.index)
    features = features.loc[common_idx]
    targets = targets.loc[common_idx]

    data_splits, scalers = split_data(features, targets)

    print(f"\nData split for horizon={horizon}:")
    print(f"  Train: {len(data_splits['train']['features'])} samples")
    print(f"  Val:   {len(data_splits['val']['features'])} samples")
    print(f"  Test:  {len(data_splits['test']['features'])} samples")

    return data_splits, scalers


if __name__ == "__main__":
    from data.fetcher import load_or_fetch_data

    df = load_or_fetch_data()
    features = engineer_features(df)
    targets = create_targets(features)

    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"\nFeature columns: {features.columns.tolist()}")
