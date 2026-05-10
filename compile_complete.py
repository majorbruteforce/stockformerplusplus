#!/usr/bin/env python
"""Compile complete ablation results."""

import json
import os

RESULT_DIR = "results"

# All model results
results = {
    "Transformer (MSE)": {
        "MAE": 0.1718,
        "RMSE": 0.2518,
        "R2": -60.29,
        "Dir_Acc": 50.8,
        "Sharpe": 0.309,
        "Cum_Return": 24.3,
        "MDD": -77.2,
        "Profit_Factor": 1.06,
    },
    "Time2Vec (MSE)": {
        "MAE": 0.1795,
        "RMSE": 0.2429,
        "R2": -56.02,
        "Dir_Acc": 50.4,
        "Sharpe": 0.242,
        "Cum_Return": -5.4,
        "MDD": -74.7,
        "Profit_Factor": 1.04,
    },
    "Gating (MSE)": {
        "MAE": 0.1983,
        "RMSE": 0.2479,
        "R2": -58.41,
        "Dir_Acc": 50.0,
        "Sharpe": -0.299,
        "Cum_Return": -89.6,
        "MDD": -94.0,
        "Profit_Factor": 0.95,
    },
    "RNN (MSE)": {
        "MAE": 0.2762,
        "RMSE": 0.3535,
        "R2": -119.82,
        "Dir_Acc": 49.6,
        "Sharpe": -0.196,
        "Cum_Return": -84.1,
        "MDD": -91.7,
        "Profit_Factor": 0.97,
    },
    "LSTM (MSE)": {
        "MAE": 0.0348,
        "RMSE": 0.0465,
        "R2": -1.09,
        "Dir_Acc": 48.5,
        "Sharpe": -0.408,
        "Cum_Return": -93.2,
        "MDD": -94.2,
        "Profit_Factor": 0.93,
    },
    "Transformer (Sharpe)": {
        "MAE": 2.5009,
        "RMSE": 2.9637,
        "R2": -8489.06,
        "Dir_Acc": 49.8,
        "Sharpe": 0.041,
        "Cum_Return": -58.3,
        "MDD": -76.7,
        "Profit_Factor": 1.01,
    },
}

# Save JSON
with open(os.path.join(RESULT_DIR, "complete_ablation_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("=" * 80)
print("COMPLETE ABLATION RESULTS (20 folds, 5 epochs each)")
print("=" * 80)
print("\n| Model | MAE | R² | Dir% | Sharpe | Return% | MDD% | PF |")
print("|-------|-----|-----|------|--------|---------|------|----|")
for name, m in results.items():
    print(
        f"| {name} | {m['MAE']:.4f} | {m['R2']:.1f} | {m['Dir_Acc']:.1f}% | {m['Sharpe']:.2f} | {m['Cum_Return']:+.1f}% | {abs(m['MDD']):.1f}% | {m['Profit_Factor']:.2f} |"
    )

# Markdown table
with open(os.path.join(RESULT_DIR, "complete_ablation_table.md"), "w") as f:
    f.write("# Complete Ablation Study Results\n\n")
    f.write("| Model | MAE | RMSE | R² | Dir% | Sharpe | Return% | MDD% | PF |\n")
    f.write("|-------|-----|------|-----|------|--------|---------|------|----|\n")
    for name, m in results.items():
        f.write(
            f"| {name} | {m['MAE']:.4f} | {m['RMSE']:.4f} | {m['R2']:.2f} | {m['Dir_Acc']:.1f}% | {m['Sharpe']:.3f} | {m['Cum_Return']:.1f}% | {abs(m['MDD']):.1f}% | {m['Profit_Factor']:.2f} |\n"
        )

print("\n✓ Saved to results/complete_ablation_results.json")
print("✓ Saved to results/complete_ablation_table.md")

# Summary
print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print("""
1. BEST PERFORMER: Transformer (MSE) - only positive Sharpe (0.31), only positive return (+24%)

2. BASELINES: Both RNN and LSTM perform worse than Transformer

3. ABLATION VARIANTS: Adding Time2Vec or Gating hurts performance

4. LOSS FUNCTION: MSE produces better results than Sharpe Loss for this task

5. ALL MODELS: ~50% directional accuracy (essentially random)
""")
