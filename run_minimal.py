#!/usr/bin/env python
"""Minimal working ablation."""

import config

config.set_seed(42)
config.TRAIN_CONFIG["epochs"] = 8

import numpy as np
import pandas as pd
import os
import json

from data.fetcher import load_or_fetch_data
from features.engineer import engineer_features, compute_market_status, create_targets
from utils.dataset import TimeSeriesDataset
from utils.training import train_model, get_predictions
from utils.metrics import compute_all_metrics
from main import create_model
from torch.utils.data import DataLoader
from config import DEVICE, RESULT_DIR

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
print(f"Dim: {in_d}, Market: {mkt_d}")

# Manual 3-fold walk-forward
folds = [
    (0, 400),  # train 0-400
    (300, 700),  # train 300-700
    (600, 1000),  # train 600-1000
]

models = [
    ("Transformer", "transformer", False, 0),
    ("+T2V", "time2vec_transformer", True, 0),
    ("+Gate", "time2vec_transformer", True, mkt_d),
]

results = {}
all_preds = {}
all_targs = None

for mname, mtype, inc_t, mdim in models:
    print(f"\n{mname}:")
    preds_l, targs_l = [], []

    for fold_idx, (tr_start, tr_end) in enumerate(folds):
        # Train: tr_start to tr_end
        # Val: tr_end to tr_end+50
        # Test: tr_end+50 to tr_end+100

        tr_f = features.iloc[tr_start:tr_end].values
        tr_t = targets.iloc[tr_start:tr_end]["target_h1"].values

        vl_start, vl_end = tr_end - 60, tr_end + 50
        vl_f = features.iloc[vl_start:vl_end].values
        vl_t = targets.iloc[vl_end - 60 : vl_end]["target_h1"].values

        te_start, te_end = vl_end - 60, vl_end + 50
        te_f = features.iloc[te_start:te_end].values
        te_t = targets.iloc[te_end - 60 : te_end]["target_h1"].values

        # Skip if not enough data
        if len(te_t) < 10:
            continue

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
            epochs=8,
            device=DEVICE,
            verbose=False,
            loss_type="mse",
        )

        preds, targs = get_predictions(model, test_loader, DEVICE)
        if preds.ndim > 1:
            preds = preds[:, 0]

        preds_l.append(preds)
        targs_l.append(targs)
        print(f" f{fold_idx + 1}", end="")

    fp = np.concatenate(preds_l)
    ft = np.concatenate(targs_l)
    if all_targs is None:
        all_targs = ft

    m = compute_all_metrics(fp, ft)
    results[mname] = m
    all_preds[mname] = fp
    print(f" | R2={m['R2']:.1f}, Dir={m['Dir_Acc']:.0f}%, Sharpe={m['Sharpe']:.1f}")

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

print("\n=== RESULTS ===")
for n, m in results.items():
    print(
        f"{n}: MAE={m['MAE']:.4f}, R2={m['R2']:.1f}, Dir={m['Dir_Acc']:.0f}%, Sharpe={m['Sharpe']:.1f}, Ret={m['Cum_Return']:.1f}%"
    )
