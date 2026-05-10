# Stockformer++ Research Paper Readiness Report

## Executive Summary

This document summarizes the current state of the Stockformer++ repository for academic paper preparation.

## Scientific Validity Audit

### Issues Fixed ✅

1. **Model State Leakage Across Folds**
   - Location: `main.py:349-374`
   - Problem: Model weights persisted across walk-forward folds
   - Fix: Models now reinitialized fresh for each fold

2. **Position Calculation Bug**
   - Location: `utils/metrics.py:56-58`
   - Problem: Dead code - `tanh(pred)` computed but immediately overwritten
   - Fix: Removed dead code, now uses consistent binary positions

3. **Prediction Scaling Issue**
   - Problem: Predictions around -5.2 while targets around 0
   - Root Cause: SharpeLoss encouraged extreme predictions
   - Fix: Added MSE loss option in `utils/training.py`

4. **Train/Evaluation Mismatch**
   - Problem: Training used `tanh(pred)`, evaluation used `sign(pred)`
   - Fix: Consistent MSE-based training and evaluation

### Verified Working

Single fold test with MSE loss:
- Predictions: mean=0.0836, std=0.0335 (correct scale)
- Targets: mean=0.0008, std=0.0205 (correct scale)
- R² = -18.84 (realistic - model not learning well but scales are correct)

### Remaining Issues

1. **Full Ablation Not Complete**
   - Need to run ablation with MSE loss (walk-forward generator works but full run times out)
   - Need to test multiple seeds for statistical validity

2. **RC-T2V Not Implemented**
   - Paper claims "Regime-Conditioned Time2Vec"
   - Only basic Time2Vec exists - no HMM or regime detection

3. **Sharpe-Loss Ablation**
   - Need to compare MSE vs Sharpe-loss training effects

4. **Missing Components**
   - Attention visualization
   - Fold-wise statistical analysis
   - Bootstrap confidence intervals

## Bug Fixes Applied

| File | Change |
|------|--------|
| `main.py` | Model reinitialized each fold |
| `utils/metrics.py` | Removed dead position code |
| `utils/plotting.py` | Fixed position calculation |
| `utils/training.py` | Added MSE loss option |
| `utils/mse_loss.py` | New file for MSE loss |

## Current Ablation Results (Old/Fixed)

| Model | MAE | R² | Dir Acc | Sharpe | Cum Ret |
|-------|-----|-----|---------|--------|---------|
| Transformer | 5.39 | -27697 | 55.3% | 1.57 | 49.7% |
| +Time2Vec | 4.88 | -22891 | 55.3% | 1.57 | 49.7% |
| +Gating | 4.70 | -20819 | 54.0% | 1.29 | 37.3% |

*Note: These results are from SharpeLoss which produced invalid predictions. Results need to be regenerated with MSE loss.*

## Files Created

```
paper_assets/
├── figures/
├── tables/
├── latex/
├── results/
├── configs/
└── logs/
```

## Next Steps

1. Run full ablation with MSE loss
2. Implement RC-T2V if claimed in paper
3. Add statistical significance testing
4. Generate publication-quality figures
5. Create LaTeX tables

## Reproducibility

To reproduce experiments with fixes:
```bash
# Normal benchmark
python main.py

# With MSE loss (fixed scaling)
# Edit utils/training.py: loss_type='mse'
```

## Configuration

- Seed: 42 (fixed in `config.py`)
- Device: CUDA if available
- Epochs: 50 (default), 15 (quick)
- Batch size: 32

## Known Weaknesses

1. Single stock (NVDA) - need multiple stocks for generalization claims
2. Limited folds for statistical significance
3. No hyperparameter sensitivity analysis
4. No comparison with external baselines (TFT, Informer, etc.)

---

Generated: May 2026