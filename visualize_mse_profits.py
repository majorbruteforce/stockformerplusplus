#!/usr/bin/env python
"""
MSE vs Profits visualization - shows prediction quality vs trading performance.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

from config import RESULT_DIR, FEATURE_CONFIG


def compute_trading_metrics(preds, targets, transaction_cost=0.001):
    """Compute trading strategy metrics from predictions."""
    position = np.where(preds > 0, 1, -1)
    raw_returns = position * targets
    net_costs = np.abs(np.diff(np.concatenate([[0], position]))) * transaction_cost
    strategy_returns = raw_returns - net_costs

    return strategy_returns


def plot_mse_vs_profits(results, preds_dict, targets, save_path):
    """Create MSE vs Profit comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    models = list(results.keys())
    transaction_cost = FEATURE_CONFIG["transaction_cost"]

    # 1. MSE Comparison
    ax = axes[0, 0]
    mae_values = [results[m]["MAE"] for m in models]
    ax.bar(
        models, mae_values, color=["#2E86AB", "#E94F37", "#A23B72"], edgecolor="black"
    )
    ax.set_title(
        "Mean Absolute Error (lower = better predictions)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel("MAE")
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(mae_values):
        ax.text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # 2. Cumulative Returns Comparison
    ax = axes[0, 1]
    ret_values = [results[m]["Cum_Return"] for m in models]
    colors = ["green" if v > 0 else "red" for v in ret_values]
    ax.bar(models, ret_values, color=colors, edgecolor="black", alpha=0.7)
    ax.set_title("Cumulative Return (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Return (%)")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(ret_values):
        ax.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Sharpe Ratio Comparison
    ax = axes[1, 0]
    sharpe_values = [results[m]["Sharpe"] for m in models]
    colors = ["green" if v > 0 else "red" for v in sharpe_values]
    ax.bar(models, sharpe_values, color=colors, edgecolor="black", alpha=0.7)
    ax.set_title(
        "Sharpe Ratio (higher = better risk-adjusted)", fontsize=12, fontweight="bold"
    )
    ax.set_ylabel("Sharpe")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(sharpe_values):
        ax.text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # 4. Prediction Quality vs Trading Quality Scatter
    ax = axes[1, 1]
    for idx, m in enumerate(models):
        mae = results[m]["MAE"]
        sharpe = results[m]["Sharpe"]
        ax.scatter(mae, sharpe, s=200, label=m, edgecolor="black", linewidth=2)
        ax.annotate(
            m, (mae, sharpe), xytext=(5, 5), textcoords="offset points", fontsize=10
        )

    ax.set_xlabel("MAE (prediction error, lower=better)")
    ax.set_ylabel("Sharpe Ratio (trading performance, higher=better)")
    ax.set_title(
        "Prediction Quality vs Trading Quality", fontsize=12, fontweight="bold"
    )
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_returns_distribution(preds_dict, targets, save_path):
    """Plot distribution of actual returns vs strategy returns."""
    fig, axes = plt.subplots(len(preds_dict), 2, figsize=(14, 5 * len(preds_dict)))

    if len(preds_dict) == 1:
        axes = axes.reshape(1, -1)

    for idx, (name, preds) in enumerate(preds_dict.items()):
        position = np.where(preds > 0, 1, -1)
        strategy_returns = position * targets

        # Actual returns distribution
        ax1 = axes[idx, 0]
        ax1.hist(targets, bins=30, alpha=0.7, color="#2E86AB", edgecolor="black")
        ax1.axvline(x=0, color="red", linestyle="--", linewidth=2)
        ax1.set_title(
            f"{name}: Actual Returns Distribution", fontsize=12, fontweight="bold"
        )
        ax1.set_xlabel("Log Return")
        ax1.set_ylabel("Frequency")
        ax1.grid(True, alpha=0.3)

        # Strategy returns distribution
        ax2 = axes[idx, 1]
        ax2.hist(
            strategy_returns, bins=30, alpha=0.7, color="#E94F37", edgecolor="black"
        )
        ax2.axvline(x=0, color="green", linestyle="--", linewidth=2)
        ax2.set_title(
            f"{name}: Strategy Returns Distribution", fontsize=12, fontweight="bold"
        )
        ax2.set_xlabel("Log Return")
        ax2.set_ylabel("Frequency")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_rolling_metrics(preds_dict, targets, window=20, save_path=None):
    """Plot rolling Sharpe and returns."""
    fig, axes = plt.subplots(len(preds_dict), 2, figsize=(14, 5 * len(preds_dict)))

    if len(preds_dict) == 1:
        axes = axes.reshape(1, -1)

    transaction_cost = FEATURE_CONFIG["transaction_cost"]

    for idx, (name, preds) in enumerate(preds_dict.items()):
        position = np.where(preds > 0, 1, -1)
        raw_returns = position * targets
        net_costs = np.abs(np.diff(np.concatenate([[0], position]))) * transaction_cost
        strategy_returns = raw_returns - net_costs

        # Rolling returns
        rolling_ret = pd.Series(strategy_returns).rolling(window).sum()
        ax1 = axes[idx, 0]
        ax1.plot(rolling_ret, color="#2E86AB", linewidth=1.5)
        ax1.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax1.set_title(
            f"{name}: Rolling {window}-period Returns", fontsize=12, fontweight="bold"
        )
        ax1.set_ylabel("Cumulative Return")
        ax1.grid(True, alpha=0.3)

        # Rolling Sharpe
        rolling_sharpe = (
            pd.Series(strategy_returns).rolling(window).mean()
            / pd.Series(strategy_returns).rolling(window).std()
        )
        ax2 = axes[idx, 1]
        ax2.plot(rolling_sharpe, color="#E94F37", linewidth=1.5)
        ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax2.set_title(
            f"{name}: Rolling {window}-period Sharpe", fontsize=12, fontweight="bold"
        )
        ax2.set_ylabel("Sharpe Ratio")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("Loading results for MSE vs Profits visualization...")

    results_path = os.path.join(RESULT_DIR, "ablation_results.json")
    preds_path = os.path.join(RESULT_DIR, "ablation_preds.npy")
    targets_path = os.path.join(RESULT_DIR, "ablation_targets.npy")

    with open(results_path, "r") as f:
        results = json.load(f)

    all_preds = np.load(preds_path, allow_pickle=True).item()
    targets = np.load(targets_path, allow_pickle=True)

    print("Generating MSE vs Profits plots...")

    plot_mse_vs_profits(
        results, all_preds, targets, os.path.join(RESULT_DIR, "mse_vs_profits.png")
    )
    plot_returns_distribution(
        all_preds, targets, os.path.join(RESULT_DIR, "returns_distribution.png")
    )
    plot_rolling_metrics(
        all_preds,
        targets,
        window=20,
        save_path=os.path.join(RESULT_DIR, "rolling_metrics.png"),
    )

    print("\nDone! Additional plots saved to results/")


if __name__ == "__main__":
    import pandas as pd

    main()
