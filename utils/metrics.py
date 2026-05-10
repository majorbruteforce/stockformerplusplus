"""
Evaluation metrics for ML and financial performance.
"""

import numpy as np
from typing import Dict, Any
from config import FEATURE_CONFIG


def compute_ml_metrics(
    predictions: np.ndarray, targets: np.ndarray
) -> Dict[str, float]:
    """
    Compute machine learning metrics.

    Metrics:
        MAE: Mean Absolute Error
        RMSE: Root Mean Squared Error
        R2: R-squared (coefficient of determination)
        Directional Accuracy: % correct sign prediction
    """
    mae = np.mean(np.abs(predictions - targets))

    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    directional_accuracy = np.mean(np.sign(predictions) == np.sign(targets)) * 100

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Dir_Acc": directional_accuracy}


def compute_financial_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    transaction_cost: float = FEATURE_CONFIG["transaction_cost"],
) -> Dict[str, float]:
    """
    Compute financial backtesting metrics.

    Trading strategy:
        - If predicted return > 0: Long position
        - Else: Short position
        - Hold for 1 period

    Metrics:
        Cumulative Return: Total return of strategy
        Sharpe Ratio: Risk-adjusted return (annualized)
        Maximum Drawdown: Largest peak-to-trough decline
        Profit Factor: Gross profits / gross losses
    """
    n = len(predictions)

    position = np.where(predictions > 0, 1, -1)

    raw_returns = position * targets

    net_returns = np.abs(np.diff(np.concatenate([[0], position]))) * transaction_cost
    strategy_returns = raw_returns - net_returns

    cumulative_return = np.prod(1 + strategy_returns) - 1

    mean_return = np.mean(strategy_returns)
    std_return = np.std(strategy_returns)

    if std_return > 0:
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    cumulative_wealth = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative_wealth)
    drawdowns = (cumulative_wealth - running_max) / running_max
    max_drawdown = np.min(drawdowns) * 100

    gross_profits = np.sum(strategy_returns[strategy_returns > 0])
    gross_losses = np.abs(np.sum(strategy_returns[strategy_returns < 0]))

    profit_factor = gross_profits / gross_losses if gross_losses > 0 else 0.0

    return {
        "Cum_Return": cumulative_return * 100,
        "Sharpe": sharpe_ratio,
        "MDD": max_drawdown,
        "Profit_Factor": profit_factor,
    }


def compute_all_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    transaction_cost: float = FEATURE_CONFIG["transaction_cost"],
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Returns:
        Dictionary with all ML and financial metrics
    """
    ml_metrics = compute_ml_metrics(predictions, targets)
    fin_metrics = compute_financial_metrics(predictions, targets, transaction_cost)

    return {**ml_metrics, **fin_metrics}


def format_metrics_table(results: Dict[str, Dict[str, float]]) -> str:
    """
    Format metrics as a comparison table.

    Args:
        results: Dictionary mapping model names to their metrics

    Returns:
        Formatted table string
    """
    header = "| Model | MAE | RMSE | R² | Dir Acc | Sharpe | MDD | Cum Return |"
    separator = "|------:|-----:|-----:|-----:|--------:|-------:|----:|----------:|"

    rows = [header, separator]

    for model_name, metrics in results.items():
        row = f"| {model_name} | {metrics['MAE']:.6f} | {metrics['RMSE']:.6f} | "
        row += f"{metrics['R2']:.4f} | {metrics['Dir_Acc']:.2f}% | "
        row += f"{metrics['Sharpe']:.4f} | {metrics['MDD']:.2f}% | "
        row += f"{metrics['Cum_Return']:.2f}% |"
        rows.append(row)

    return "\n".join(rows)


def find_best_model(
    results: Dict[str, Dict[str, float]], metric: str = "Sharpe"
) -> str:
    """
    Find best performing model based on specified metric.

    Args:
        results: Dictionary mapping model names to metrics
        metric: Metric to use for ranking

    Returns:
        Name of best model
    """
    if metric == "MAE" or metric == "RMSE" or metric == "MDD":
        return min(results.keys(), key=lambda x: results[x][metric])
    else:
        return max(results.keys(), key=lambda x: results[x][metric])


if __name__ == "__main__":
    np.random.seed(42)
    predictions = np.random.randn(100) * 0.01
    targets = np.random.randn(100) * 0.01

    metrics = compute_all_metrics(predictions, targets)
    print("Sample metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
