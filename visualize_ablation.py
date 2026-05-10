#!/usr/bin/env python
"""
Ablation visualization - comparison plots and prediction plots.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

from config import RESULT_DIR


def plot_ablation_comparison(results, save_path):
    """Generate ablation comparison bar charts."""
    models = list(results.keys())

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    metrics = ["MAE", "RMSE", "Dir_Acc", "Sharpe", "Cum_Return", "MDD"]
    titles = [
        "MAE (lower better)",
        "RMSE (lower better)",
        "Direction Accuracy (%)",
        "Sharpe Ratio (higher better)",
        "Cumulative Return (%)",
        "Max Drawdown (%)",
    ]

    for ax, metric, title in zip(axes.flat, metrics, titles):
        values = [results[m][metric] for m in models]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))

        bars = ax.bar(models, values, color=colors, edgecolor="black", linewidth=1.2)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f"{val:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_predictions_vs_actual(preds_dict, targets, save_path):
    """Plot predictions vs actual for each model."""
    n_models = len(preds_dict)
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 5 * n_models))

    if n_models == 1:
        axes = axes.reshape(1, -1)

    for idx, (name, preds) in enumerate(preds_dict.items()):
        # Time series plot
        ax1 = axes[idx, 0]
        ax1.plot(
            targets[:200], label="Actual", alpha=0.8, linewidth=1.5, color="#2E86AB"
        )
        ax1.plot(
            preds[:200], label="Predicted", alpha=0.8, linewidth=1.5, color="#E94F37"
        )
        ax1.set_title(f"{name}: Predictions vs Actual", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Log Return")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Scatter plot
        ax2 = axes[idx, 1]
        ax2.scatter(targets, preds, alpha=0.4, s=10, color="#2E86AB")

        # Perfect prediction line
        min_val = min(targets.min(), preds.min())
        max_val = max(targets.max(), preds.max())
        ax2.plot(
            [min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect"
        )

        ax2.set_title(f"{name}: Scatter Plot", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Actual")
        ax2.set_ylabel("Predicted")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_equity_curves(preds_dict, targets, save_path):
    """Plot equity curves for all models."""
    from config import FEATURE_CONFIG

    n_models = len(preds_dict)
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#2E86AB", "#E94F37", "#A23B72", "#F18F01"]

    for idx, (name, preds) in enumerate(preds_dict.items()):
        position = np.where(preds > 0, 1, -1)
        raw_returns = position * targets
        transaction_cost = FEATURE_CONFIG["transaction_cost"]
        net_costs = np.abs(np.diff(np.concatenate([[0], position]))) * transaction_cost
        strategy_returns = raw_returns - net_costs
        cumulative_wealth = np.cumprod(1 + strategy_returns)

        ax.plot(
            cumulative_wealth, label=name, linewidth=2, color=colors[idx % len(colors)]
        )

    # Buy and hold baseline
    buy_hold = np.cumprod(1 + targets)
    ax.plot(
        buy_hold,
        label="Buy & Hold",
        linewidth=2,
        color="gray",
        linestyle="--",
        alpha=0.7,
    )

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cumulative Wealth")
    ax.set_title("Ablation: Equity Curves Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_loss_comparison(histories, save_path):
    """Plot training loss curves for all models."""
    fig, axes = plt.subplots(1, len(histories), figsize=(5 * len(histories), 4))

    if len(histories) == 1:
        axes = [axes]

    for idx, (name, history) in enumerate(histories.items()):
        ax = axes[idx]
        ax.plot(history["train_loss"], label="Train", linewidth=2)
        ax.plot(history["val_loss"], label="Val", linewidth=2)
        ax.set_title(f"{name}: Loss Curves", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_ablation_summary(results, save_path):
    """Create a summary table as an image."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    # Create table data
    models = list(results.keys())
    cols = ["Model", "MAE", "RMSE", "Dir Acc (%)", "Sharpe", "Cum Ret (%)", "MDD (%)"]

    cell_text = []
    for m in models:
        r = results[m]
        cell_text.append(
            [
                m,
                f"{r['MAE']:.4f}",
                f"{r['RMSE']:.4f}",
                f"{r['Dir_Acc']:.1f}",
                f"{r['Sharpe']:.3f}",
                f"{r['Cum_Return']:.1f}",
                f"{r['MDD']:.2f}",
            ]
        )

    table = ax.table(cellText=cell_text, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(len(cols)):
        table[(0, j)].set_facecolor("#2E86AB")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Style rows with alternating colors
    for i in range(1, len(models) + 1):
        for j in range(len(cols)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#E8E8E8")

    ax.set_title("Ablation Study Results", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("Loading ablation results...")

    results_path = os.path.join(RESULT_DIR, "ablation_results.json")
    preds_path = os.path.join(RESULT_DIR, "ablation_preds.npy")
    targets_path = os.path.join(RESULT_DIR, "ablation_targets.npy")

    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found!")
        return

    with open(results_path, "r") as f:
        results = json.load(f)

    all_preds = np.load(preds_path, allow_pickle=True).item()
    targets = np.load(targets_path, allow_pickle=True)

    print(f"Loaded results for: {list(results.keys())}")
    print(f"Targets shape: {targets.shape}")

    # Generate all plots
    print("\nGenerating visualizations...")

    plot_ablation_comparison(
        results, os.path.join(RESULT_DIR, "ablation_comparison.png")
    )
    plot_predictions_vs_actual(
        all_preds, targets, os.path.join(RESULT_DIR, "ablation_predictions.png")
    )
    plot_equity_curves(
        all_preds, targets, os.path.join(RESULT_DIR, "ablation_equity.png")
    )
    plot_ablation_summary(results, os.path.join(RESULT_DIR, "ablation_table.png"))

    print("\nDone! All plots saved to results/")


if __name__ == "__main__":
    main()
