"""
Score Calibration and Threshold Utilities.

Provides threshold calibration, score normalization, and trust score validation.
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class ThresholdCalibrator:
    """Calibrate decision thresholds on validation data."""

    def __init__(self):
        self.threshold = 0.5
        self.calibrated = False

    def calibrate(self, scores: np.ndarray, labels: np.ndarray,
                  criterion: str = 'eer') -> float:
        """Find optimal threshold based on criterion."""
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        fnr = 1.0 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        self.threshold = float(thresholds[eer_idx])
        self.calibrated = True
        logger.info(f"Calibrated threshold={self.threshold:.4f} using {criterion}")
        return self.threshold

    def apply(self, scores: np.ndarray) -> np.ndarray:
        """Apply threshold to produce binary predictions."""
        return (scores >= self.threshold).astype(int)


class ScoreNormalizer:
    """Normalize scores to [0, 1] range."""

    def __init__(self):
        self.min_val = 0.0
        self.max_val = 1.0
        self.fitted = False

    def fit(self, scores: np.ndarray):
        """Fit min-max normalization parameters."""
        self.min_val = float(scores.min())
        self.max_val = float(scores.max())
        self.fitted = True

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores."""
        if not self.fitted:
            self.fit(scores)
        denom = self.max_val - self.min_val
        if denom < 1e-10:
            return np.full_like(scores, 0.5)
        return (scores - self.min_val) / denom


class TrustScoreValidator:
    """Validate trust scores meet expected properties."""

    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        self.min_val = min_val
        self.max_val = max_val

    def validate(self, scores: np.ndarray, name: str = "trust") -> bool:
        """Validate trust scores are in expected range and not collapsed."""
        if scores.min() < self.min_val or scores.max() > self.max_val:
            logger.warning(f"{name} out of [{self.min_val}, {self.max_val}]")
            return False
        if scores.std() < 1e-5:
            logger.error(f"{name} collapsed: std={scores.std():.8f}")
            return False
        if np.isnan(scores).any():
            logger.error(f"{name} contains NaN")
            return False
        return True
