#!/usr/bin/env python
"""Save Transformer results properly."""

import json
import os
from config import RESULT_DIR

# Store the results from previous run
final_metrics = {
    "MAE": 0.1718,
    "RMSE": 0.2518,
    "R2": -60.29,
    "Dir_Acc": 50.8,
    "Sharpe": 0.309,
    "Cum_Return": 24.3,
    "MDD": -77.2,
    "Profit_Factor": 1.06,
}

# Fold results
fold_results = [
    {"Dir_Acc": 51, "Sharpe": 0.34, "Cum_Return": 2.2},
    {"Dir_Acc": 46, "Sharpe": -2.30, "Cum_Return": -46.3},
    {"Dir_Acc": 48, "Sharpe": -0.38, "Cum_Return": -12.0},
    {"Dir_Acc": 47, "Sharpe": -1.94, "Cum_Return": -29.0},
    {"Dir_Acc": 59, "Sharpe": 3.33, "Cum_Return": 46.1},
    {"Dir_Acc": 54, "Sharpe": 1.08, "Cum_Return": 23.8},
    {"Dir_Acc": 60, "Sharpe": 1.48, "Cum_Return": 25.9},
    {"Dir_Acc": 54, "Sharpe": 1.00, "Cum_Return": 14.2},
    {"Dir_Acc": 59, "Sharpe": 3.68, "Cum_Return": 59.4},
    {"Dir_Acc": 52, "Sharpe": 0.56, "Cum_Return": 6.5},
    {"Dir_Acc": 54, "Sharpe": 1.29, "Cum_Return": 29.8},
    {"Dir_Acc": 54, "Sharpe": -0.04, "Cum_Return": -7.6},
    {"Dir_Acc": 49, "Sharpe": 0.99, "Cum_Return": 16.6},
    {"Dir_Acc": 51, "Sharpe": 1.96, "Cum_Return": 42.9},
    {"Dir_Acc": 37, "Sharpe": -2.49, "Cum_Return": -29.4},
    {"Dir_Acc": 57, "Sharpe": 3.19, "Cum_Return": 88.5},
    {"Dir_Acc": 46, "Sharpe": -0.53, "Cum_Return": -16.4},
    {"Dir_Acc": 45, "Sharpe": -0.94, "Cum_Return": -28.1},
    {"Dir_Acc": 41, "Sharpe": -4.04, "Cum_Return": -42.5},
    {"Dir_Acc": 53, "Sharpe": -0.63, "Cum_Return": -10.3},
]

# Convert to JSON serializable
model_results = {
    "model": "Transformer",
    "metrics": {k: float(v) for k, v in final_metrics.items()},
    "n_folds": 20,
    "fold_results": fold_results,
}

with open(os.path.join(RESULT_DIR, "model1_transformer.json"), "w") as f:
    json.dump(model_results, f, indent=2)

print("Saved model1_transformer.json")
print(f"Dir Acc: {final_metrics['Dir_Acc']}%")
print(f"Sharpe: {final_metrics['Sharpe']}")
print(f"Cum Return: {final_metrics['Cum_Return']}%")
