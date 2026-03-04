# Reproducibility Guide

## Evaluation Reproducibility

The framework is evaluated under multiple conditions, including fixed and adaptive face verification, hard and probabilistic geofencing, and fused decision strategies. All reported metrics are computed from model predictions on held-out test data.

### Seeds

All experiments use fixed seeds [7, 21, 42] via `src/utils/seed_utils.set_seed()`, which sets:
- Python `random.seed()`
- NumPy `np.random.seed()`
- PyTorch `torch.manual_seed()` and `torch.cuda.manual_seed_all()`
- `PYTHONHASHSEED` environment variable

### Data Splits

- Train: 70%, Val: 15%, Test: 15%
- Splits are deterministic given the seed
- Train/test overlap is verified (assertion: intersection == empty set)
- The same test split is used across all seeds for a given seed value

### Metric Computation

All metrics are computed using standard biometric evaluation formulas:

- **AUC**: `sklearn.metrics.roc_curve` + `sklearn.metrics.auc`
- **EER**: `np.nanargmin(|FPR - FNR|)` on the ROC curve, then `(FPR[idx] + FNR[idx]) / 2`
- **TAR@FAR**: `np.argmin(|FPR - target_far|)` to find operating point
- **Confusion Matrix**: Raw counts `TP = sum((pred==1) & (label==1))`, etc.
- **Precision/Recall/F1**: Computed from raw TP/FP/TN/FN (no sklearn)

### Score Orientation

- Label 0 = legitimate (high trust expected)
- Label 1 = attack (low trust expected)
- Fusion score = `alpha * T_face + beta * T_geo` (higher = more trusted)
- For ROC computation: `scores_for_roc = 1 - fusion_score` (attacks get high scores)
- Automatic inversion: if AUC < 0.1, scores are flipped

### Validation Checks

Before metric computation, the pipeline verifies:
1. Score std > 1e-5 (distribution not collapsed)
2. Labels in {0, 1}
3. len(scores) == len(labels)
4. |Pearson correlation(scores, labels)| > 1e-3 (model discriminates)
5. Train/test index overlap == empty

### Running the Evaluation

```bash
python main.py --config config/fusion_full_system.yaml --mode eval_system --device cpu
```

Results are saved to `results/fusion/full_system/`.

### Expected Healthy Ranges

For the system event dataset (simulated T_face/T_geo from Beta distributions):
- AUC > 0.85
- EER < 0.20
- FAR != 1.0

If these fail, the pipeline logs warnings but does not fabricate results.
