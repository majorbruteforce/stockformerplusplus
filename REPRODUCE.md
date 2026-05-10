# Stockformer++ Reproducibility Guide

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Benchmark
```bash
python main.py
```

### Run Ablation Study
```bash
python run_full_ablation.py
```

## Configuration

All configuration in `config.py`:
- `SEED = 42` - Fixed random seed
- `SYMBOL = "NVDA"` - Stock to use
- `TRAIN_CONFIG["epochs"]` - Number of training epochs
- `FEATURE_CONFIG["seq_len"]` - Input sequence length

## Bug Fixes Applied

### 1. Model State Reset (FIXED)
The benchmark now reinitializes models fresh for each walk-forward fold to prevent temporal leakage.

### 2. Prediction Scaling (FIXED)
Added MSE loss option. To use:
```python
# In training.py, line ~229
train_model(..., loss_type='mse')  # Use MSE instead of SharpeLoss
```

### 3. Position Calculation (FIXED)
Removed dead code in `utils/metrics.py` and `utils/plotting.py`.

## Expected Results (After Fixes)

With MSE loss, expect:
- Predictions in similar scale as targets (both ~0)
- R² in realistic range (negative but not -20000)
- Directional accuracy around 50-55%

## Files

- `config.py` - Hyperparameters
- `main.py` - Main benchmark
- `run_full_ablation.py` - Ablation study
- `run_minimal.py` - Quick test
- `test_wf.py` - Walk-forward test
- `visualize_ablation.py` - Generate plots

## Results Location

Results saved to `results/`:
- `ablation_results.json` - Ablation metrics
- `*.png` - Visualizations

## Known Issues

1. Single stock only (NVDA)
2. Limited statistical testing
3. RC-T2V not fully implemented
4. Attention visualization not complete

## Contact

For issues, check:
1. `paper_assets/VALIDITY_AUDIT.md` - Audit report
2. `results/` - Past experiment logs