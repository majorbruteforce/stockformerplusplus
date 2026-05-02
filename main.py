"""
Main execution script for Stockformer+ benchmark.
Trains and evaluates all four models on financial time series data.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
from tqdm import tqdm

import config
from config import (
    FEATURE_CONFIG,
    MODEL_CONFIG,
    TRAIN_CONFIG,
    DEVICE,
    RESULT_DIR,
    DATA_DIR,
)
from data.fetcher import load_or_fetch_data
from features.engineer import prepare_data, engineer_features, create_targets
from utils.dataset import TimeSeriesDataset
from utils.training import train_model, get_predictions
from utils.metrics import compute_all_metrics, format_metrics_table, find_best_model
from utils.plotting import plot_all_results
from models.rnn_lstm import RNNModel, LSTMModel
from models.stockformer import Stockformer
from models.time2vec_transformer import Time2VecTransformer
from features.engineer import prepare_data, engineer_features, create_targets, compute_market_status, generate_walk_forward_splits

def prepare_dataloaders(
    data_splits: Dict,
    horizon: int,
    batch_size: int = TRAIN_CONFIG["batch_size"],
    seq_len: int = FEATURE_CONFIG["seq_len"],
    include_time: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test sets.
    """
    target_col = f"target_h{horizon}"

    # --- NEW: Safely extract market features if they exist ---
    train_market = data_splits["train"]["market_features"].values if "market_features" in data_splits["train"] else None
    val_market = data_splits["val"]["market_features"].values if "market_features" in data_splits["val"] else None
    test_market = data_splits["test"]["market_features"].values if "market_features" in data_splits["test"] else None
    # ---------------------------------------------------------

    train_dataset = TimeSeriesDataset(
        features=data_splits["train"]["features"].values,
        targets=data_splits["train"]["targets"][target_col].values,
        market_features=train_market, # <-- NEW
        seq_len=seq_len,
        include_time=include_time,
    )

    val_dataset = TimeSeriesDataset(
        features=data_splits["val"]["features"].values,
        targets=data_splits["val"]["targets"][target_col].values,
        market_features=val_market, # <-- NEW
        seq_len=seq_len,
        include_time=include_time,
    )

    test_dataset = TimeSeriesDataset(
        features=data_splits["test"]["features"].values,
        targets=data_splits["test"]["targets"][target_col].values,
        market_features=test_market, # <-- NEW
        seq_len=seq_len,
        include_time=include_time,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def create_model(
    model_name: str, input_dim: int, horizon: int, include_time: bool = False, market_dim: int = 0
) -> torch.nn.Module:
    """
    Factory function to create model by name.
    """
    cfg = MODEL_CONFIG[model_name]

    if model_name == "rnn":
        model = RNNModel(
            input_dim=input_dim,
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            horizon=horizon,
        )
    elif model_name == "lstm":
        model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            horizon=horizon,
        )
    elif model_name == "stockformer":
        model = Stockformer(
            input_dim=input_dim,
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            horizon=horizon,
        )
    elif model_name == "time2vec_transformer":
        model = Time2VecTransformer(
            input_dim=input_dim,
            market_dim=market_dim, # <-- NEW
            t2v_dim=cfg["t2v_dim"],
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            horizon=horizon,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def train_and_evaluate_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    horizon: int,
    input_dim: int,
    include_time: bool = False,
    market_dim: int = 0, # <-- NEW
    verbose: bool = True,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Training {model_name} (horizon={horizon})")
        print(f"{'=' * 60}")

    # --- CHANGED HERE ---
    model = create_model(model_name, input_dim, horizon, include_time, market_dim)
    # --------------------

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"Model parameters: {num_params:,}")

    # ... rest of the function remains exactly the same ...

    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=TRAIN_CONFIG["epochs"],
        lr=TRAIN_CONFIG["learning_rate"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        patience=TRAIN_CONFIG["patience"],
        min_delta=TRAIN_CONFIG["min_delta"],
        device=DEVICE,
        verbose=verbose,
    )

    predictions, targets = get_predictions(model, test_loader, DEVICE)

    if predictions.ndim > 1 and predictions.shape[1] > 1:
        predictions = predictions[:, 0]

    metrics = compute_all_metrics(predictions, targets)

    if verbose:
        print(f"\n{model_name} Results:")
        print(f"  MAE: {metrics['MAE']:.6f}")
        print(f"  RMSE: {metrics['RMSE']:.6f}")
        print(f"  R2: {metrics['R2']:.4f}")
        print(f"  Dir Acc: {metrics['Dir_Acc']:.2f}%")
        print(f"  Sharpe: {metrics['Sharpe']:.4f}")
        print(f"  MDD: {metrics['MDD']:.2f}%")
        print(f"  Cum Return: {metrics['Cum_Return']:.2f}%")

    return metrics, history, predictions


# def run_benchmark(horizon: int = 1):
#     """
#     Run full benchmark for all models.
#     """
#     print(f"\n{'#' * 70}")
#     print(f"# Stockformer+ Benchmark (Horizon = {horizon})")
#     print(f"{'#' * 70}")

#     print("\n[1/5] Loading data...")
#     df = load_or_fetch_data()
#     print(f"Loaded {len(df)} days of data")

#     print("\n[2/5] Preparing features and splits...")
#     data_splits, scalers = prepare_data(df, horizon=horizon)

#     input_dim = data_splits["train"]["features"].shape[1]
#     print(f"Input dimension: {input_dim}")

#     # --- Extract market dimension ---
#     market_dim = data_splits["train"]["market_features"].shape[1] if "market_features" in data_splits["train"] else 0
#     print(f"Market dimension: {market_dim}")
#     # --------------------------------

#     print("\n[3/5] Creating data loaders...")
#     train_loader, val_loader, test_loader = prepare_dataloaders(
#         data_splits,
#         horizon,
#         batch_size=TRAIN_CONFIG["batch_size"],
#         seq_len=FEATURE_CONFIG["seq_len"],
#         include_time=False,
#     )

#     train_loader_t2v, val_loader_t2v, test_loader_t2v = prepare_dataloaders(
#         data_splits,
#         horizon,
#         batch_size=TRAIN_CONFIG["batch_size"],
#         seq_len=FEATURE_CONFIG["seq_len"],
#         include_time=True,
#     )

#     test_target_col = f"target_h{horizon}"
#     test_targets_full = data_splits["test"]["targets"][test_target_col].values
#     seq_len = FEATURE_CONFIG["seq_len"]
#     test_targets = test_targets_full[seq_len:]

#     model_names = ["rnn", "lstm", "stockformer", "time2vec_transformer"]

#     results = {}
#     histories = {}
#     all_predictions = {}

#     print("\n[4/5] Training all models...")

#     for model_name in model_names:
#         if model_name == "time2vec_transformer":
#             t_loader, v_loader, te_loader = (
#                 train_loader_t2v,
#                 val_loader_t2v,
#                 test_loader_t2v,
#             )
#             include_time = True
#         else:
#             t_loader, v_loader, te_loader = train_loader, val_loader, test_loader
#             include_time = False

#         metrics, history, predictions = train_and_evaluate_model(
#             model_name,
#             t_loader,
#             v_loader,
#             te_loader,
#             horizon=horizon,
#             input_dim=input_dim,
#             include_time=include_time,
#             market_dim=market_dim, # <-- Passing market_dim here
#             verbose=True,
#         )

#         results[model_name] = metrics
#         histories[model_name] = history
#         all_predictions[model_name] = predictions

#     print("\n[5/5] Generating plots and saving results...")
#     os.makedirs(RESULT_DIR, exist_ok=True)

#     plot_all_results(
#         all_predictions,
#         {"h1": test_targets, "h5": test_targets},
#         histories,
#         results,
#         horizon,
#         RESULT_DIR,
#     )

#     print("\n" + "=" * 70)
#     print("RESULTS COMPARISON TABLE")
#     print("=" * 70)
#     print(format_metrics_table(results))

#     best_model = find_best_model(results, metric="Sharpe")
#     print(f"\nBest Model (by Sharpe Ratio): {best_model}")

#     results_path = os.path.join(RESULT_DIR, f"results_h{horizon}.json")
#     results_serializable = {
#         k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()
#     }
#     with open(results_path, "w") as f:
#         json.dump(results_serializable, f, indent=2)

#     print(f"\nResults saved to: {results_path}")

#     return results, histories, all_predictions

def run_benchmark(horizon: int = 1):
    print(f"\n{'#' * 70}")
    print(f"# Stockformer+ Walk-Forward Benchmark (Horizon = {horizon})")
    print(f"{'#' * 70}")

    print("\n[1/5] Loading and Aligning Global Data...")
    df = load_or_fetch_data()
    
    # Generate full unbroken timelines for features, market state, and targets
    features = engineer_features(df)
    market_features = compute_market_status(df)
    targets = create_targets(features, horizons=[horizon])
    
    # Intersect so we have perfectly aligned dates across all dataframes
    common_idx = features.index.intersection(targets.index).intersection(market_features.index)
    features = features.loc[common_idx]
    market_features = market_features.loc[common_idx]
    targets = targets.loc[common_idx]

    input_dim = features.shape[1]
    market_dim = market_features.shape[1]
    
    print(f"Input dimension: {input_dim} | Market dimension: {market_dim}")
    print(f"Total available trading days: {len(features)}")

    model_names = ["rnn", "lstm", "stockformer", "time2vec_transformer"]
    results = {}
    histories = {}
    all_predictions = {}
    test_targets_master = None

    print("\n[2/5] Initiating Walk-Forward Training loops...")

    for model_name in model_names:
        print(f"\n{'=' * 60}")
        print(f"Training {model_name.upper()}")
        print(f"{'=' * 60}")

        include_time = (model_name == "time2vec_transformer")
        
        # 1. Initialize the model ONCE.
        # It will continuously fine-tune its weights as it walks forward through time.
        model = create_model(model_name, input_dim, horizon, include_time, market_dim)
        
        fold_predictions = []
        fold_targets = []
        model_history = {"train_loss": [], "val_loss": [], "lr": []}
        
       # 2. Spin up the rolling window generator
        splits_generator = generate_walk_forward_splits(
            features, targets, market_features, 
            train_days=1000, val_days=150, step_days=60,
            seq_len=FEATURE_CONFIG["seq_len"] # <-- ADD THIS LINE
        )
        
        for fold, data_splits in enumerate(splits_generator):
            train_loader, val_loader, test_loader = prepare_dataloaders(
                data_splits, horizon, include_time=include_time
            )
            
            # Train/Fine-tune on this specific window (verbose=False to keep console clean)
            history = train_model(
                model, train_loader, val_loader, 
                epochs=TRAIN_CONFIG["epochs"], 
                device=DEVICE, verbose=False
            )
            
            # Predict the subsequent 60 unknown days
            preds, targs = get_predictions(model, test_loader, DEVICE)
            if preds.ndim > 1 and preds.shape[1] > 1:
                preds = preds[:, 0]
                
            fold_predictions.append(preds)
            fold_targets.append(targs)
            
            model_history["train_loss"].extend(history["train_loss"])
            model_history["val_loss"].extend(history["val_loss"])
            
            print(f"  Fold {fold + 1} (60-day step) | Best Val Loss: {history['best_val_loss']:.6f}")

        # 3. Stitch all the 60-day steps together to evaluate the total multi-year journey
        final_preds = np.concatenate(fold_predictions)
        final_targets = np.concatenate(fold_targets)
        
        # Save master targets for the plotter (only need to do this once)
        if test_targets_master is None:
            test_targets_master = final_targets
            
        metrics = compute_all_metrics(final_preds, final_targets)
        
        results[model_name] = metrics
        histories[model_name] = model_history
        all_predictions[model_name] = final_preds

        print(f"\n{model_name.upper()} Total Backtest Results:")
        print(f"  Dir Acc: {metrics['Dir_Acc']:.2f}% | Sharpe: {metrics['Sharpe']:.4f} | Cum Return: {metrics['Cum_Return']:.2f}%")

    print("\n[5/5] Generating plots and saving results...")
    os.makedirs(RESULT_DIR, exist_ok=True)

    plot_all_results(
        all_predictions,
        {f"h{horizon}": test_targets_master},
        histories,
        results,
        horizon,
        RESULT_DIR,
    )

    print("\n" + "=" * 70)
    print("RESULTS COMPARISON TABLE")
    print("=" * 70)
    print(format_metrics_table(results))

    best_model = find_best_model(results, metric="Sharpe")
    print(f"\nBest Model (by Sharpe Ratio): {best_model}")

    results_path = os.path.join(RESULT_DIR, f"results_h{horizon}.json")
    results_serializable = {
        k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()
    }
    with open(results_path, "w") as f:
        json.dump(results_serializable, f, indent=2)

    return results, histories, all_predictions

def main():
    """Main entry point."""
    config.set_seed(config.SEED)

    print(f"Device: {DEVICE}")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")

    horizons = FEATURE_CONFIG["horizons"]

    all_results = {}

    for h in horizons:
        results, histories, predictions = run_benchmark(horizon=h)
        all_results[f"h{h}"] = {
            "results": results,
            "histories": histories,
            "predictions": predictions,
        }

    print("\n" + "#" * 70)
    print("# BENCHMARK COMPLETE")
    print("#" * 70)

    return all_results


if __name__ == "__main__":
    all_results = main()
