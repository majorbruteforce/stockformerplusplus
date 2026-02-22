"""
Plotting functions for visualization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from config import RESULT_DIR


def plot_prediction_vs_actual(
    predictions: np.ndarray,
    targets: np.ndarray,
    model_name: str,
    horizon: int,
    save_path: Optional[str] = None,
):
    """
    Plot predictions vs actual values.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(targets, label="Actual", alpha=0.7)
    axes[0].plot(predictions, label="Predicted", alpha=0.7)
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Log Return")
    axes[0].set_title(f"{model_name}: Predictions vs Actual (h={horizon})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(targets, predictions, alpha=0.5, s=10)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect")
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title(f"{model_name}: Scatter Plot (h={horizon})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_equity_curve(
    predictions: np.ndarray,
    targets: np.ndarray,
    model_name: str,
    horizon: int,
    transaction_cost: float = 0.001,
    save_path: Optional[str] = None,
):
    """
    Plot equity curve for trading strategy.
    """
    position = np.where(predictions > 0, 1, -1)

    raw_returns = position * targets

    net_costs = np.abs(np.diff(np.concatenate([[0], position]))) * transaction_cost
    strategy_returns = raw_returns - net_costs

    cumulative_wealth = np.cumprod(1 + strategy_returns)

    buy_hold = np.cumprod(1 + targets)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(cumulative_wealth, label=f"{model_name} Strategy", linewidth=2)
    ax.plot(buy_hold, label="Buy & Hold", alpha=0.7, linewidth=1.5)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cumulative Wealth")
    ax.set_title(f"{model_name}: Equity Curve (h={horizon})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    final_return = (cumulative_wealth[-1] - 1) * 100
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_drawdown(
    predictions: np.ndarray,
    targets: np.ndarray,
    model_name: str,
    horizon: int,
    transaction_cost: float = 0.001,
    save_path: Optional[str] = None,
):
    """
    Plot drawdown curve.
    """
    position = np.where(predictions > 0, 1, -1)

    raw_returns = position * targets
    net_costs = np.abs(np.diff(np.concatenate([[0], position]))) * transaction_cost
    strategy_returns = raw_returns - net_costs

    cumulative_wealth = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative_wealth)
    drawdowns = (cumulative_wealth - running_max) / running_max * 100

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(range(len(drawdowns)), drawdowns, 0, alpha=0.3, color="red")
    ax.plot(drawdowns, color="red", linewidth=1)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title(f"{model_name}: Drawdown (h={horizon})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_loss_curves(
    histories: Dict[str, Dict[str, List[float]]], save_path: Optional[str] = None
):
    """
    Plot training/validation loss curves for all models.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    model_names = list(histories.keys())

    for idx, model_name in enumerate(model_names):
        ax = axes[idx]

        history = histories[model_name]

        ax.plot(history["train_loss"], label="Train Loss", alpha=0.8)
        ax.plot(history["val_loss"], label="Val Loss", alpha=0.8)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (Huber)")
        ax.set_title(f"{model_name}: Training History")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax.set_yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_comparison_bar(
    results: Dict[str, Dict[str, float]],
    metric: str,
    title: str,
    save_path: Optional[str] = None,
):
    """
    Plot comparison bar chart for a specific metric.
    """
    model_names = list(results.keys())
    values = [results[m][metric] for m in model_names]

    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(model_names, values, color=colors)

    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{val:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_all_results(
    all_predictions: Dict[str, np.ndarray],
    all_targets: Dict[str, np.ndarray],
    histories: Dict[str, Dict[str, List[float]]],
    all_metrics: Dict[str, Dict[str, float]],
    horizon: int,
    result_dir: str = RESULT_DIR,
):
    """
    Generate all plots for all models.
    """
    os.makedirs(result_dir, exist_ok=True)

    for model_name, predictions in all_predictions.items():
        targets_key = f"h{horizon}"

        plot_prediction_vs_actual(
            predictions,
            all_targets[targets_key],
            model_name,
            horizon,
            save_path=os.path.join(
                result_dir, f"{model_name}_predictions_h{horizon}.png"
            ),
        )

        plot_equity_curve(
            predictions,
            all_targets[targets_key],
            model_name,
            horizon,
            save_path=os.path.join(result_dir, f"{model_name}_equity_h{horizon}.png"),
        )

        plot_drawdown(
            predictions,
            all_targets[targets_key],
            model_name,
            horizon,
            save_path=os.path.join(result_dir, f"{model_name}_drawdown_h{horizon}.png"),
        )

    plot_loss_curves(histories, save_path=os.path.join(result_dir, "loss_curves.png"))

    metrics_list = ["MAE", "RMSE", "R2", "Sharpe", "MDD", "Cum_Return"]

    for metric in metrics_list:
        title = f"Model Comparison: {metric}"
        plot_comparison_bar(
            all_metrics,
            metric,
            title,
            save_path=os.path.join(result_dir, f"comparison_{metric.lower()}.png"),
        )


if __name__ == "__main__":
    print("Plotting utilities loaded successfully")
