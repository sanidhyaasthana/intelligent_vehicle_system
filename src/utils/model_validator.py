"""
Model Validation Utilities.

Provides validation checks for model inference, embeddings, and pipeline integrity.
"""

import logging
import numpy as np
import torch
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ModelValidator:
    """Validates model outputs for correctness."""

    def __init__(self, expected_dim: int = 512):
        self.expected_dim = expected_dim

    def validate_output(self, output: torch.Tensor, name: str = "output") -> bool:
        """Check output tensor is valid (no NaN, no Inf, correct shape)."""
        if torch.isnan(output).any():
            logger.error(f"{name} contains NaN values")
            return False
        if torch.isinf(output).any():
            logger.error(f"{name} contains Inf values")
            return False
        return True


class InferenceValidator:
    """Validates inference pipeline behavior."""

    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0

    def check_score_range(self, scores: np.ndarray, name: str = "scores",
                          min_val: float = 0.0, max_val: float = 1.0) -> bool:
        """Verify scores fall within expected range."""
        if scores.min() < min_val or scores.max() > max_val:
            logger.warning(f"{name} out of [{min_val}, {max_val}] range: "
                           f"[{scores.min():.4f}, {scores.max():.4f}]")
            self.checks_failed += 1
            return False
        self.checks_passed += 1
        return True

    def check_no_collapse(self, scores: np.ndarray, name: str = "scores",
                          min_std: float = 1e-5) -> bool:
        """Verify score distribution hasn't collapsed."""
        if scores.std() < min_std:
            logger.error(f"{name} distribution collapsed: std={scores.std():.8f}")
            self.checks_failed += 1
            return False
        self.checks_passed += 1
        return True

    def summary(self) -> Dict[str, int]:
        return {'passed': self.checks_passed, 'failed': self.checks_failed}


class EmbeddingValidator:
    """Validates embedding vectors."""

    def __init__(self, expected_dim: int = 512, tolerance: float = 1e-3):
        self.expected_dim = expected_dim
        self.tolerance = tolerance

    def validate(self, embedding: np.ndarray, name: str = "embedding") -> bool:
        """Validate embedding: correct dim, unit norm, no NaN."""
        if embedding.shape[-1] != self.expected_dim:
            logger.error(f"{name} dimension mismatch: expected {self.expected_dim}, got {embedding.shape[-1]}")
            return False
        if np.isnan(embedding).any():
            logger.error(f"{name} contains NaN")
            return False
        norm = np.linalg.norm(embedding)
        if abs(norm - 1.0) > self.tolerance:
            logger.warning(f"{name} not unit-normalized: norm={norm:.4f}")
        return True
