#!/usr/bin/env python
"""Minimal valid ablation - fast execution with proper scientific checks."""

import config

config.set_seed(config.SEED)
config.TRAIN_CONFIG["epochs"] = 10

import numpy as np
import pandas as pd
import os
import json

from data.fetcher import load_or_fetch_data
from features.engineer import (
    engineer_features,
    compute_market_status,
    create_targets,
    generate_walk_forward_splits,
)
from utils.dataset import TimeSeriesDataset
from utils.training import train_model, get_predictions
from utils.metrics import compute_all_metrics
from main import create_model
from torch.utils.data import DataLoader
from config import FEATURE_CONFIG, TRAIN_CONFIG, DEVICE, RESULT_DIR


def run_quick_ablation():
    print("Loading data...")
    df = load_or_fetch_data()

    features = engineer_features(df)
    market_features = compute_market_status(df)
    targets = create_targets(features, horizons=[1])

    common = features.index.intersection(targets.index).intersection(
        market_features.index
    )
    features = features.loc[common].reset_index(drop=True)
    market_features = market_features.loc[common].reset_index(drop=True)
    targets = targets.loc[common].reset_index(drop=True)

    in_d, mkt_d = features.shape[1], market_features.shape[1]
    print(f"Features: {in_d}, Market: {mkt_d}")

    # 3 folds only
    configs = [
        ("Transformer", "transformer", False, 0),
        ("+Time2Vec", "time2vec_transformer", True, 0),
        ("+Gating", "time2vec_transformer", True, mkt_d),
    ]

    results = {}
    all_preds = {}
    all_targets_arr = None

    for model_name, create_name, include_time, mkt_dim in configs:
        print(f"\n{model_name}:")

        preds_list, targs_list = [], []

        # 3 quick folds
        splits_gen = generate_walk_forward_splits(
            features,
            targets,
            market_features,
            train_days=600,
            val_days=80,
            step_days=150,
            seq_len=60,
        )

        for fold, splits in enumerate(splits_gen):
            # Build datasets
            tr_f = splits["train"]["features"].values
            tr_t = splits["train"]["targets"]["target_h1"].values
            vl_f = splits["val"]["features"].values
            vl_t = splits["val"]["targets"]["target_h1"].values
            te_f = splits["test"]["features"].values
            te_t = splits["test"]["targets"]["target_h1"].values

            train_ds = TimeSeriesDataset(tr_f, tr_t, None, 60, include_time)
            val_ds = TimeSeriesDataset(vl_f, vl_t, None, 60, include_time)
            test_ds = TimeSeriesDataset(te_f, te_t, None, 60, include_time)

            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

            # Fresh model each fold
            model = create_model(create_name, in_d, 1, include_time, mkt_dim)

            # Train with MSE
            train_model(
                model,
                train_loader,
                val_loader,
                epochs=10,
                device=DEVICE,
                verbose=False,
                loss_type="mse",
            )

            # Predict
            preds, targs = get_predictions(model, test_loader, DEVICE)
            if preds.ndim > 1:
                preds = preds[:, 0]

            preds_list.append(preds)
            targs_list.append(targs)

            print(f"  f{fold + 1}", end="")

        # Aggregate
        full_p = np.concatenate(preds_list)
        full_t = np.concatenate(targs_list)

        if all_targets_arr is None:
            all_targets_arr = full_t

        all_preds[model_name] = full_p

        metrics = compute_all_metrics(full_p, full_t)
        results[model_name] = metrics

        print(
            f" | MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.2f}, Dir={metrics['Dir_Acc']:.0f}%, Sharpe={metrics['Sharpe']:.2f}"
        )

    # Save
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(os.path.join(RESULT_DIR, "ablation_v2_results.json"), "w") as f:
        json.dump(
            {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
            f,
            indent=2,
        )

    np.save(os.path.join(RESULT_DIR, "ablation_v2_preds.npy"), all_preds)
    np.save(os.path.join(RESULT_DIR, "ablation_v2_targets.npy"), all_targets_arr)

    print("\n=== ABLATION TABLE ===")
    print("| Model | MAE | RMSE | R² | Dir% | Sharpe | Ret% | MDD% |")
    print("|-------|-----|------|-----|------|--------|------|------|")
    for n, m in results.items():
        print(
            f"| {n} | {m['MAE']:.4f} | {m['RMSE']:.4f} | {m['R2']:.2f} | {m['Dir_Acc']:.0f}% | {m['Sharpe']:.2f} | {m['Cum_Return']:.1f}% | {m['MDD']:.1f}% |"
        )

    return results, all_preds, all_targets_arr


if __name__ == "__main__":
    run_quick_ablation()
