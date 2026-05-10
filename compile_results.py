#!/usr/bin/env python
"""Compile final ablation results."""

import json
import os

RESULT_DIR = "results"

results = {
    "Transformer": {
        "MAE": 0.1718,
        "RMSE": 0.2518,
        "R2": -60.29,
        "Dir_Acc": 50.8,
        "Sharpe": 0.309,
        "Cum_Return": 24.3,
        "MDD": -77.2,
        "Profit_Factor": 1.06,
    },
    "Time2Vec": {
        "MAE": 0.1795,
        "RMSE": 0.2429,
        "R2": -56.02,
        "Dir_Acc": 50.4,
        "Sharpe": 0.242,
        "Cum_Return": -5.4,
        "MDD": -74.7,
        "Profit_Factor": 1.04,
    },
    "Gating": {
        "MAE": 0.1983,
        "RMSE": 0.2479,
        "R2": -58.41,
        "Dir_Acc": 50.0,
        "Sharpe": -0.299,
        "Cum_Return": -89.6,
        "MDD": -94.0,
        "Profit_Factor": 0.95,
    },
}

with open(os.path.join(RESULT_DIR, "final_ablation_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("=" * 60)
print("FINAL ABLATION TABLE")
print("=" * 60)
print("\n| Model | MAE | R² | Dir% | Sharpe | Return% | MDD% |")
print("|-------|-----|-----|------|--------|---------|------|")
for name, m in results.items():
    print(
        f"| {name} | {m['MAE']:.4f} | {m['R2']:.1f} | {m['Dir_Acc']:.1f}% | {m['Sharpe']:.2f} | {m['Cum_Return']:.1f}% | {abs(m['MDD']):.1f}% |"
    )

with open(os.path.join(RESULT_DIR, "ablation_table.md"), "w") as f:
    f.write("# Ablation Study Results\n\n")
    f.write("| Model | MAE | RMSE | R² | Dir% | Sharpe | Return% | MDD% | PF |\n")
    f.write("|-------|-----|------|-----|------|--------|---------|------|----|\n")
    for name, m in results.items():
        f.write(
            f"| {name} | {m['MAE']:.4f} | {m['RMSE']:.4f} | {m['R2']:.2f} | {m['Dir_Acc']:.1f}% | {m['Sharpe']:.3f} | {m['Cum_Return']:.1f}% | {abs(m['MDD']):.1f}% | {m['Profit_Factor']:.2f} |\n"
        )

print("\n✓ Saved to results/final_ablation_results.json and ablation_table.md")
