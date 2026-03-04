"""
Evaluation Engine — Publication-Grade Biometric Metrics.

LABEL CONVENTION (enforced throughout):
    1 = genuine (legitimate user)
    0 = impostor (attack)

SCORE CONVENTION:
    Higher score = more likely genuine.
    pos_label=1 in sklearn roc_curve means genuine is the positive class.

METRIC DEFINITIONS:
    FAR = FP / (FP + TN) = impostors accepted / total impostors
    FRR = FN / (FN + TP) = genuine rejected / total genuine

This module provides strict, auditable metric computation with:
- Debug logging for T_face, T_geo, fusion, and decision scores
- Score distribution collapse detection
- Label alignment verification
- sklearn-based ROC/AUC/EER computation
- TAR@FAR at standard operating points
- Confusion matrix from raw counts (no sklearn black box)
- Automatic score inversion detection

All metrics come from real model inference. No synthetic data.
No random numbers. No fabricated scores.
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import roc_curve, auc

logger = logging.getLogger(__name__)


def debug_log_scores(
    T_face: np.ndarray,
    T_geo: np.ndarray,
    fusion_score: np.ndarray,
    decision_score: np.ndarray,
):
    """
    STEP 1: Print strict debug logging before metric computation.

    Logs min, max, mean, std for each score array.
    Raises RuntimeError if any distribution has collapsed (std < 1e-5).
    """
    logger.info("=" * 70)
    logger.info("EVALUATION ENGINE — SCORE DISTRIBUTION DEBUG")
    logger.info("=" * 70)

    for name, arr in [
        ("T_face", T_face),
        ("T_geo", T_geo),
        ("Fusion score", fusion_score),
        ("Final decision score", decision_score),
    ]:
        _min = float(arr.min())
        _max = float(arr.max())
        _mean = float(arr.mean())
        _std = float(arr.std())

        logger.info(
            f"  {name:25s}: min={_min:.6f}  max={_max:.6f}  "
            f"mean={_mean:.6f}  std={_std:.6f}"
        )

        if _std < 1e-5:
            raise RuntimeError(
                f"Score distribution collapsed — likely bug in inference. "
                f"{name}: std={_std:.8f}"
            )

    logger.info("=" * 70)


def verify_label_alignment(
    scores: np.ndarray,
    labels: np.ndarray,
):
    """
    STEP 2: Verify label alignment before computing metrics.

    Prints counts and checks correlation.
    Raises on mismatch or zero correlation.
    """
    logger.info("--- Label Alignment Check ---")
    logger.info(f"  len(scores) = {len(scores)}")
    logger.info(f"  len(labels) = {len(labels)}")
    logger.info(f"  num positives (label=1) = {int(np.sum(labels == 1))}")
    logger.info(f"  num negatives (label=0) = {int(np.sum(labels == 0))}")
    logger.info(f"  unique labels = {np.unique(labels).tolist()}")

    assert set(np.unique(labels).tolist()).issubset({0, 1}), \
        f"Labels must be in {{0, 1}}, got {np.unique(labels).tolist()}"
    assert len(scores) == len(labels), \
        f"len(scores)={len(scores)} != len(labels)={len(labels)}"

    corr = np.corrcoef(scores, labels)[0, 1]
    logger.info(f"  Pearson correlation(scores, labels) = {corr:.6f}")

    if abs(corr) < 1e-3:
        raise RuntimeError(
            f"Scores not correlated with labels — model not discriminating. "
            f"corr={corr:.6f}"
        )


def compute_biometric_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, Any]:
    """
    STEP 3-5: Compute all biometric evaluation metrics.

    Uses sklearn for ROC/AUC/EER. Confusion matrix from raw counts.

    Args:
        labels: Ground truth, {0, 1}. 1 = genuine, 0 = impostor.
        scores: Continuous scores. Higher = more likely genuine.

    Returns:
        Dict with all metrics.
    """
    # STEP 3: sklearn-based ROC and AUC
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # STEP 11: Automatic score inversion detection
    if roc_auc < 0.1:
        logger.warning(
            f"AUC={roc_auc:.4f} < 0.1 — score inverted. Flipping scores."
        )
        scores = 1.0 - scores
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        roc_auc = auc(fpr, tpr)

    if np.all(fpr >= 0.99):
        logger.warning("FAR=1.0 everywhere — threshold direction inverted.")

    # EER: intersection of FPR and FNR
    fnr = 1.0 - tpr
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_index] + fnr[eer_index]) / 2.0)
    eer_threshold = float(thresholds[eer_index])

    # STEP 4: TAR@FAR
    def _tar_at_far(target_far):
        idx = np.argmin(np.abs(fpr - target_far))
        return float(tpr[idx])

    tar_at_far_001 = _tar_at_far(0.01)
    tar_at_far_0001 = _tar_at_far(0.001)

    # STEP 5: Confusion matrix at EER threshold — from raw counts, NO sklearn
    preds = (scores >= eer_threshold).astype(int)

    TP = int(np.sum((preds == 1) & (labels == 1)))
    FP = int(np.sum((preds == 1) & (labels == 0)))
    TN = int(np.sum((preds == 0) & (labels == 0)))
    FN = int(np.sum((preds == 0) & (labels == 1)))
    N = TP + FP + TN + FN

    accuracy = (TP + TN) / N if N > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    far = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    frr = FN / (FN + TP) if (FN + TP) > 0 else 0.0

    metrics = {
        'roc_auc': float(roc_auc),
        'eer': eer,
        'eer_threshold': eer_threshold,
        'tar_at_far_001': tar_at_far_001,
        'tar_at_far_0001': tar_at_far_0001,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'far': far,
        'frr': frr,
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'n_samples': N,
    }

    # Log summary
    logger.info("--- Biometric Metrics ---")
    for k in ['roc_auc', 'eer', 'eer_threshold', 'tar_at_far_001',
              'accuracy', 'precision', 'recall', 'f1', 'far', 'frr']:
        logger.info(f"  {k:25s}: {metrics[k]:.6f}")
    logger.info(f"  {'TP':25s}: {TP}")
    logger.info(f"  {'FP':25s}: {FP}")
    logger.info(f"  {'TN':25s}: {TN}")
    logger.info(f"  {'FN':25s}: {FN}")

    return metrics
