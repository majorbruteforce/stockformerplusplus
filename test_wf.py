#!/usr/bin/env python
"""Ablation using proven walk-forward generator."""

import config

config.set_seed(42)
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
from config import DEVICE, RESULT_DIR

print("Loading data...")
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

# Test walk-forward generator
print("Testing walk-forward...")
splits_gen = generate_walk_forward_splits(
    features,
    targets,
    market_features,
    train_days=400,
    val_days=80,
    step_days=150,
    seq_len=60,
)

# Get just first 2 splits
splits = [next(splits_gen), next(splits_gen)]
print(f"Got {len(splits)} splits")

# Quick test: train one model on first split
print("\nTesting single fold...")
spl = splits[0]
tr_f = spl["train"]["features"].values
tr_t = spl["train"]["targets"]["target_h1"].values
vl_f = spl["val"]["features"].values
vl_t = spl["val"]["targets"]["target_h1"].values
te_f = spl["test"]["features"].values
te_t = spl["test"]["targets"]["target_h1"].values

print(f"Train: {tr_f.shape}, Val: {vl_f.shape}, Test: {te_f.shape}")

train_ds = TimeSeriesDataset(tr_f, tr_t, None, 60, False)
val_ds = TimeSeriesDataset(vl_f, vl_t, None, 60, False)
test_ds = TimeSeriesDataset(te_f, te_t, None, 60, False)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

model = create_model("transformer", in_d, 1, False, 0)
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

print(f"Predictions: {preds.mean():.4f}, {preds.std():.4f}")
print(f"Targets: {targs.mean():.4f}, {targs.std():.4f}")

m = compute_all_metrics(preds, targs)
print(f"R2={m['R2']:.2f}, Dir={m['Dir_Acc']:.0f}%, Sharpe={m['Sharpe']:.2f}")

print("\nWalk-forward generator works correctly!")
