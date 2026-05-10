#!/usr/bin/env python
"""Full ablation with working walk-forward generator."""

import config

config.set_seed(42)
config.TRAIN_CONFIG["epochs"] = 15

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
from config import DEVICE, RESULT_DIR, TRAIN_CONFIG

print("Loading...")
df = load_or_fetch_data()
features = engineer_features(df)
market_features = compute_market_status(df)
targets = create_targets(features, horizons=[1])

common = features.index.intersection(targets.index).intersection(market_features.index)
features = features.loc[common].reset_index(drop=True)
market_features = market_features.loc[common].reset_index(drop=True)
targets = targets.loc[common].reset_index(drop=True)

in_d = features.shape[1]
mkt_d = market_features.shape[1]
print(f"Input: {in_d}, Market: {mkt_d}")

# Models
configs = [
    ("Transformer", "transformer", False, 0),
    ("+T2V", "time2vec_transformer", True, 0),
    ("+Gating", "time2vec_transformer", True, mkt_d),
]

results = {}
all_preds = {}
all_targs = None

for mname, mtype, inc_t, mdim in configs:
    print(f"\n=== {mname} ===")

    preds_list, targs_list = [], []

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

        train_ds = TimeSeriesDataset(tr_f, tr_t, None, 60, inc_t)
        val_ds = TimeSeriesDataset(vl_f, vl_t, None, 60, inc_t)
        test_ds = TimeSeriesDataset(te_f, te_t, None, 60, inc_t)

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        model = create_model(mtype, in_d, 1, inc_t, mdim)

        train_model(
            model,
            train_loader,
            val_loader,
            epochs=TRAIN_CONFIG["epochs"],
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
        print(f"  Fold {fold_num}", end="")

    fp = np.concatenate(preds_list)
    ft = np.concatenate(targs_list)
    if all_targs is None:
        all_targs = ft

    m = compute_all_metrics(fp, ft)
    results[mname] = m
    all_preds[mname] = fp

    print(
        f"\n  {mname}: MAE={m['MAE']:.4f}, R2={m['R2']:.2f}, Dir={m['Dir_Acc']:.0f}%, Sharpe={m['Sharpe']:.2f}"
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
np.save(os.path.join(RESULT_DIR, "ablation_v2_targets.npy"), all_targs)

print("\n=== ABLATION TABLE ===")
print("| Model | MAE | RMSE | R² | Dir% | Sharpe | Ret% | MDD% |")
print("|-------|-----|------|-----|------|--------|------|------|")
for n, m in results.items():
    print(
        f"| {n} | {m['MAE']:.4f} | {m['RMSE']:.4f} | {m['R2']:.2f} | {m['Dir_Acc']:.0f}% | {m['Sharpe']:.2f} | {m['Cum_Return']:.1f}% | {abs(m['MDD']):.1f}% |"
    )
