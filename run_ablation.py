#!/usr/bin/env python
"""
Ablation study runner and visualization.
"""

import config

config.set_seed(config.SEED)
config.TRAIN_CONFIG["epochs"] = 15  # Quick for testing

import numpy as np
import pandas as pd
import torch
import os
import json

from data.fetcher import load_or_fetch_data
from features.engineer import engineer_features, compute_market_status, create_targets
from utils.dataset import TimeSeriesDataset
from utils.training import train_model, get_predictions
from utils.metrics import compute_all_metrics
from main import create_model
from torch.utils.data import DataLoader
from config import FEATURE_CONFIG, TRAIN_CONFIG, DEVICE, RESULT_DIR


def quick_walk_forward_splits(
    features,
    targets,
    market_features,
    train_days=600,
    val_days=100,
    step_days=100,
    seq_len=60,
):
    """Generate fewer folds for quick testing."""
    n = len(features)
    total = train_days + val_days + step_days
    feat_idx = features.index
    tgt_idx = targets.index
    mkt_idx = market_features.index

    for start in range(0, n - total + 1, step_days):
        train_end = start + train_days
        val_end = train_end + val_days
        test_end = val_end + step_days

        train_f = feat_idx[start:train_end]
        val_f = feat_idx[train_end - seq_len : val_end]
        test_f = feat_idx[val_end - seq_len : test_end]

        aligned_tgt = targets.loc[feat_idx]
        aligned_mkt = market_features.loc[feat_idx]

        yield {
            "train": {
                "features": features.loc[train_f],
                "targets": aligned_tgt.loc[train_f],
            },
            "val": {"features": features.loc[val_f], "targets": aligned_tgt.loc[val_f]},
            "test": {
                "features": features.loc[test_f],
                "targets": aligned_tgt.loc[test_f],
            },
            "train_mkt": aligned_mkt.loc[train_f].values,
            "val_mkt": aligned_mkt.loc[val_f].values,
            "test_mkt": aligned_mkt.loc[test_f].values,
        }


def run_ablation():
    """Run ablation study."""
    print("Loading data...")
    df = load_or_fetch_data()

    features = engineer_features(df)
    market_features = compute_market_status(df)
    targets = create_targets(features, horizons=[1])

    common_idx = features.index.intersection(targets.index).intersection(
        market_features.index
    )
    features = features.loc[common_idx]
    market_features = market_features.loc[common_idx]
    targets = targets.loc[common_idx]

    input_dim = features.shape[1]
    market_dim = market_features.shape[1]
    print(f"Data: {len(features)} samples, in={input_dim}, mkt={market_dim}")

    # Ablation configurations: (name, model_name, include_time, market_dim)
    configs = [
        ("Transformer", "transformer", False, 0),
        ("+Time2Vec", "time2vec_transformer", True, 0),
        ("+MarketGating", "time2vec_transformer", True, market_dim),
    ]

    results = {}
    all_preds = {}
    all_targets = None

    for name, model_name, include_time, mkt_dim in configs:
        print(f"\n=== {name} ===")

        fold_preds = []
        fold_targets_list = []

        for fold, splits in enumerate(
            quick_walk_forward_splits(features, targets, market_features)
        ):
            # Create fresh model each fold
            model = create_model(model_name, input_dim, 1, include_time, mkt_dim)

            # Create datasets
            train_ds = TimeSeriesDataset(
                splits["train"]["features"].values,
                splits["train"]["targets"]["target_h1"].values,
                splits["train_mkt"] if mkt_dim > 0 else None,
                FEATURE_CONFIG["seq_len"],
                include_time,
            )
            val_ds = TimeSeriesDataset(
                splits["val"]["features"].values,
                splits["val"]["targets"]["target_h1"].values,
                splits["val_mkt"] if mkt_dim > 0 else None,
                FEATURE_CONFIG["seq_len"],
                include_time,
            )
            test_ds = TimeSeriesDataset(
                splits["test"]["features"].values,
                splits["test"]["targets"]["target_h1"].values,
                splits["test_mkt"] if mkt_dim > 0 else None,
                FEATURE_CONFIG["seq_len"],
                include_time,
            )

            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

            # Train
            train_model(
                model,
                train_loader,
                val_loader,
                epochs=TRAIN_CONFIG["epochs"],
                device=DEVICE,
                verbose=False,
            )

            # Predict
            preds, targs = get_predictions(model, test_loader, DEVICE)
            if preds.ndim > 1:
                preds = preds[:, 0]

            fold_preds.append(preds)
            fold_targets_list.append(targs)
            print(
                f"  Fold {fold + 1}/{sum(1 for _ in quick_walk_forward_splits(features, targets, market_features))} done",
                end="\r",
            )

        # Aggregate
        final_preds = np.concatenate(fold_preds)
        final_targs = np.concatenate(fold_targets_list)
        if all_targets is None:
            all_targets = final_targs

        metrics = compute_all_metrics(final_preds, final_targs)
        results[name] = metrics
        all_preds[name] = final_preds

        print(
            f"\n{name}: Acc={metrics['Dir_Acc']:.1f}%, Sharpe={metrics['Sharpe']:.3f}, Ret={metrics['Cum_Return']:.1f}%"
        )

    # Save results
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(os.path.join(RESULT_DIR, "ablation_results.json"), "w") as f:
        json.dump(
            {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
            f,
            indent=2,
        )

    np.save(os.path.join(RESULT_DIR, "ablation_preds.npy"), all_preds)
    np.save(os.path.join(RESULT_DIR, "ablation_targets.npy"), all_targets)

    print("\n=== ABLATION TABLE ===")
    print("| Model | MAE | RMSE | Dir Acc | Sharpe | Cum Ret |")
    print("|-------|-----|------|---------|--------|---------|")
    for name, m in results.items():
        print(
            f"| {name} | {m['MAE']:.4f} | {m['RMSE']:.4f} | {m['Dir_Acc']:.1f}% | {m['Sharpe']:.3f} | {m['Cum_Return']:.1f}% |"
        )

    return results, all_preds, all_targets


if __name__ == "__main__":
    run_ablation()
