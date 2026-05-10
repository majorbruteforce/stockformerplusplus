#!/usr/bin/env python
"""Run Time2Vec only - full folds."""

import config

config.set_seed(42)
config.TRAIN_CONFIG["epochs"] = 5

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
from config import DEVICE, RESULT_DIR

print("=" * 60)
print("MODEL 2: +TIME2VEC")
print("=" * 60)

print("\n[1/4] Loading data...")
df = load_or_fetch_data()
features = engineer_features(df)
market_features = compute_market_status(df)
targets = create_targets(features, horizons=[1])

common = features.index.intersection(targets.index).intersection(market_features.index)
features = features.loc[common].reset_index(drop=True)
market_features = market_features.loc[common].reset_index(drop=True)
targets = targets.loc[common].reset_index(drop=True)

in_d = features.shape[1]
print(f"Input dim: {in_d}")

print("\n[2/4] Running walk-forward folds...")
model_name = "time2vec_transformer"
include_time = True
market_dim = 0

preds_list = []
targs_list = []
fold_results = []

splits_gen = generate_walk_forward_splits(
    features,
    targets,
    market_features,
    train_days=400,
    val_days=80,
    step_days=100,
    seq_len=60,
)

fold_num = 0
for fold_data in splits_gen:
    tr_f = fold_data["train"]["features"].values
    tr_t = fold_data["train"]["targets"]["target_h1"].values
    vl_f = fold_data["val"]["features"].values
    vl_t = fold_data["val"]["targets"]["target_h1"].values
    te_f = fold_data["test"]["features"].values
    te_t = fold_data["test"]["targets"]["target_h1"].values

    train_ds = TimeSeriesDataset(tr_f, tr_t, None, 60, include_time)
    val_ds = TimeSeriesDataset(vl_f, vl_t, None, 60, include_time)
    test_ds = TimeSeriesDataset(te_f, te_t, None, 60, include_time)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = create_model(model_name, in_d, 1, include_time, market_dim)

    train_model(
        model,
        train_loader,
        val_loader,
        epochs=5,
        device=DEVICE,
        verbose=False,
        loss_type="mse",
    )

    preds, targs = get_predictions(model, test_loader, DEVICE)
    if preds.ndim > 1:
        preds = preds[:, 0]

    preds_list.append(preds)
    targs_list.append(targs)

    fold_num += 1
    fold_m = compute_all_metrics(preds, targs)
    fold_results.append(fold_m)
    print(
        f"  Fold {fold_num}: Dir={fold_m['Dir_Acc']:.0f}%, Sharpe={fold_m['Sharpe']:.2f}, Ret={fold_m['Cum_Return']:.1f}%"
    )

print(f"\n[3/4] Aggregating {fold_num} folds...")

all_preds = np.concatenate(preds_list)
all_targs = np.concatenate(targs_list)

print(f"Prediction stats: mean={all_preds.mean():.4f}, std={all_preds.std():.4f}")
print(f"Target stats: mean={all_targs.mean():.4f}, std={all_targs.std():.4f}")

final_metrics = compute_all_metrics(all_preds, all_targs)

print(f"\n[4/4] FINAL RESULTS - TIME2VEC:")
print("-" * 40)
print(f"MAE:  {final_metrics['MAE']:.4f}")
print(f"RMSE: {final_metrics['RMSE']:.4f}")
print(f"R2:   {final_metrics['R2']:.2f}")
print(f"Dir:  {final_metrics['Dir_Acc']:.1f}%")
print(f"Sharpe: {final_metrics['Sharpe']:.3f}")
print(f"Return: {final_metrics['Cum_Return']:.1f}%")
print(f"MDD:  {final_metrics['MDD']:.1f}%")
print(f"PF:   {final_metrics['Profit_Factor']:.2f}")

# Save
model_results = {
    "model": "Time2Vec",
    "metrics": {k: float(v) for k, v in final_metrics.items()},
    "n_folds": fold_num,
    "fold_results": [{k: float(v) for k, v in f.items()} for f in fold_results],
}

with open(os.path.join(RESULT_DIR, "model2_time2vec.json"), "w") as f:
    json.dump(model_results, f, indent=2)

np.save(os.path.join(RESULT_DIR, "model2_preds.npy"), all_preds)
np.save(os.path.join(RESULT_DIR, "model2_targs.npy"), all_targs)

print(f"\n✓ Results saved to results/model2_time2vec.json")
