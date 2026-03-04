# Risk-Aware Vehicle Authorization Framework

A multi-modal biometric fusion framework combining adaptive-margin face verification, probabilistic geofencing, and score-level fusion for vehicle authorization.


Download the LFW dataset from:
https://www.kaggle.com/datasets/jessicali9530/lfw-dataset

Extract into:
data/face/lfw-deepfunneled/



## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate system events data (if not present)
python main.py --config config/fusion_full_system.yaml --mode gen_system_events

# Run full system evaluation
python main.py --config config/fusion_full_system.yaml --mode eval_system --device cpu
```

## Architecture

1. **Face Verification**: ResNet-50 backbone + ArcFace head with quality-adaptive margins. Outputs T_face in [0, 1].
2. **Geofence Model**: Probabilistic MLP outputting T_geo in [0, 1] based on GPS features.
3. **Score-Level Fusion**: `final_score = alpha * T_face + beta * T_geo` (never binary decision fusion).
4. **Multi-Seed Evaluation**: Runs across seeds [7, 21, 42] with mean +/- std aggregation.

## Evaluation Pipeline

The evaluation pipeline (`python main.py --mode eval_system`) performs:

- Score-level fusion of T_face and T_geo from held-out test data
- Debug logging with collapse detection (std < 1e-5 raises error)
- Label alignment verification with correlation check
- ROC/AUC computation via `sklearn.metrics.roc_curve`
- EER from FPR-FNR intersection on the ROC curve
- TAR@FAR at 0.01 and 0.001
- Confusion matrix at EER threshold from raw TP/FP/TN/FN counts
- Accuracy, Precision, Recall, F1 from raw counts (no sklearn black box)
- Per-sample latency measurement via `time.perf_counter()`
- Score distribution histograms per class
- Multi-seed aggregation (mean +/- std)
- Automatic score inversion detection (AUC < 0.1 triggers flip)

## Output Files

After running `eval_system`, results are saved to `results/fusion/full_system/`:

```
results/fusion/full_system/
├── final_metrics.csv              # Aggregated metrics with mean +/- std
├── system_comparison.csv          # Same data, comparison format
├── aggregated_metrics.json        # JSON format aggregated results
├── evaluation_report.txt          # Human-readable report
├── latency_report.json            # Per-sample latency statistics
├── figures/
│   ├── score_distribution.png     # Score histogram by class
│   ├── roc_multi_seed.png         # Multi-seed ROC overlay
│   └── score_distribution_seed*.png
├── seed_7/
│   ├── metrics.json               # Per-seed metrics
│   ├── roc_curve.png
│   └── det_curve.png
├── seed_21/
│   └── ...
└── seed_42/
    └── ...
```

## Metric Guarantees

- All metrics derived from real model inference on held-out test data
- No synthetic metrics, no random numbers, no fabricated scores
- EER computed from ROC intersection (not manual FAR+FRR/2)
- TAR@FAR computed by finding closest FPR operating point
- Confusion matrix from raw TP/FP/TN/FN counts
- Score distribution collapse raises RuntimeError
- Uncorrelated scores raise RuntimeError

## Running Individual Components

### Face Verification Only

Two configurations are available: **baseline** (fixed-margin ArcFace) and **adaptive** (quality-aware margins).

```bash
# --- Face Baseline (fixed margin) ---
# Train
python main.py --config config/face_baseline.yaml --mode train_face --device cpu

# Evaluate (requires trained checkpoint in results/face/baseline/best_model.pt)
python main.py --config config/face_baseline.yaml --mode eval_face --device cpu

# --- Face Adaptive (quality-aware margin) ---
# Train
python main.py --config config/face_adaptive.yaml --mode train_face --device cpu

# Evaluate (requires trained checkpoint in results/face/adaptive/best_model.pt)
python main.py --config config/face_adaptive.yaml --mode eval_face --device cpu
```

**Note:** You must run `train_face` before `eval_face` — evaluation loads the saved `best_model.pt` checkpoint. Use `--device cuda` if a GPU is available for faster training.

### Geofence Model Only

Two configurations are available: **baseline** (hard boundary, rule-based) and **probabilistic** (learned MLP).

```bash
# --- Geo Baseline (hard boundary) ---
# No training needed — rule-based model
# Generate location data first (if not present)
python main.py --config config/geo_baseline.yaml --mode gen_geo_data --device cpu

# Evaluate
python main.py --config config/geo_baseline.yaml --mode eval_geo --device cpu

# --- Geo Probabilistic (learned MLP) ---
# Train
python main.py --config config/geo_prob.yaml --mode train_geo --device cpu

# Evaluate (requires trained checkpoint in results/geo/probabilistic/geo_model.pt)
python main.py --config config/geo_prob.yaml --mode eval_geo --device cpu
```

**Note:** The geo baseline is rule-based and requires no training. The probabilistic model must be trained before evaluation.

### Fusion (Full System)

```bash
# Generate system events (if not present)
python main.py --config config/fusion_full_system.yaml --mode gen_system_events --device cpu

# Train fusion model
python main.py --config config/fusion_full_system.yaml --mode train_fusion --device cpu

# Run multi-seed system evaluation
python main.py --config config/fusion_full_system.yaml --mode eval_system --device cpu

```

### All Available Modes

| Mode | Description | Requires Training First |
|------|-------------|------------------------|
| `train_face` | Train face recognition model (ArcFace) | No |
| `eval_face` | Evaluate trained face model on test set | Yes (`best_model.pt`) |
| `train_geo` | Train probabilistic geofence model | No |
| `eval_geo` | Evaluate geofence model on test set | Probabilistic: Yes; Baseline: No |
| `train_fusion` | Train fusion model on system events | No |
| `eval_system` | Full multi-seed system evaluation | No (uses pre-computed T_face/T_geo) |
| `gen_geo_data` | Generate synthetic location data | No |
| `gen_system_events` | Generate synthetic system events | No |

## Configuration

All experiments are YAML-driven. See `config/fusion_full_system.yaml` for the full system configuration.
