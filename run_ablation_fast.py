#!/usr/bin/env python
"""Fast ablation study - fixed indexing."""

import config

config.set_seed(config.SEED)
config.TRAIN_CONFIG["epochs"] = 5

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
from config import FEATURE_CONFIG, TRAIN_CONFIG, DEVICE, RESULT_DIR


def run_fast_ablation():
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
    print(f"Data: {len(features)} samples, in={in_d}, mkt={mkt_d}")

    SEQ_LEN = 60
    configs = [
        ("Transformer", "transformer", False, 0),
        ("+Time2Vec", "time2vec_transformer", True, 0),
        ("+Gating", "time2vec_transformer", True, mkt_d),
    ]

    results, all_preds, all_t = {}, {}, None

    # 3 folds with proper windows
    for name, model_n, inc_t, m_d in configs:
        print(f"\n{name}:")
        preds_l, targs_l = [], []

        # 3 windows: start, middle, end
        windows = [(500, 1000), (1000, 1500), (1500, 2000)]

        for fold, (train_start, train_end) in enumerate(windows):
            model = create_model(model_n, in_d, 1, inc_t, m_d)

            # Train: train_start to train_end
            # Val: train_end to train_end+50
            # Test: train_end+50 to train_end+100

            # Add seq_len lookback for features
            tr_f = features.iloc[train_start:train_end].values
            tr_t = targets.iloc[train_start:train_end]["target_h1"].values
            tr_m = (
                market_features.iloc[train_start:train_end].values if m_d > 0 else None
            )

            # Val needs seq_len lookback from train_end-60
            vl_start = max(0, train_end - SEQ_LEN)
            vl_f = features.iloc[vl_start : train_end + 50].values
            vl_t = targets.iloc[vl_start : train_end + 50]["target_h1"].values
            vl_m = (
                market_features.iloc[vl_start : train_end + 50].values
                if m_d > 0
                else None
            )

            # Test needs seq_len lookback from val_end-60 = train_end+50-60 = train_end-10
            te_start = max(0, train_end + 50 - SEQ_LEN)
            te_f = features.iloc[te_start : train_end + 100].values
            te_t = targets.iloc[te_start : train_end + 100]["target_h1"].values
            te_m = (
                market_features.iloc[te_start : train_end + 100].values
                if m_d > 0
                else None
            )

            train_ds = TimeSeriesDataset(tr_f, tr_t, tr_m, SEQ_LEN, inc_t)
            val_ds = TimeSeriesDataset(vl_f, vl_t, vl_m, SEQ_LEN, inc_t)
            test_ds = TimeSeriesDataset(te_f, te_t, te_m, SEQ_LEN, inc_t)

            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

            train_model(
                model, train_loader, val_loader, epochs=5, device=DEVICE, verbose=False
            )
            p, t = get_predictions(model, test_loader, DEVICE)
            if p.ndim > 1:
                p = p[:, 0]

            preds_l.append(p)
            targs_l.append(t)
            print(f"  fold {fold + 1}", end="", flush=True)

        fp = np.concatenate(preds_l)
        ft = np.concatenate(targs_l)
        if all_t is None:
            all_t = ft

        m = compute_all_metrics(fp, ft)
        results[name] = m
        all_preds[name] = fp
        print(
            f"\n  -> Acc={m['Dir_Acc']:.1f}%, Sharpe={m['Sharpe']:.3f}, Ret={m['Cum_Return']:.1f}%"
        )

    # Save
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(os.path.join(RESULT_DIR, "ablation_results.json"), "w") as f:
        json.dump(
            {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
            f,
            indent=2,
        )
    np.save(os.path.join(RESULT_DIR, "ablation_preds.npy"), all_preds)
    np.save(os.path.join(RESULT_DIR, "ablation_targets.npy"), all_t)

    print("\n=== ABLATION TABLE ===")
    for n, m in results.items():
        print(
            f"{n}: MAE={m['MAE']:.4f}, Acc={m['Dir_Acc']:.1f}%, Sharpe={m['Sharpe']:.3f}, Ret={m['Cum_Return']:.1f}%"
        )

    return results, all_preds, all_t


if __name__ == "__main__":
    run_fast_ablation()
