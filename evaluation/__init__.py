"""Evaluation package for research-grade experiments."""

from .evaluation_engine import (
    debug_log_scores,
    verify_label_alignment,
    compute_biometric_metrics,
)
from .multi_seed_evaluator import aggregate_metrics, run_multi_seed_evaluation
from .figure_generation import (
    plot_roc,
    plot_roc_multi_seed,
    plot_det,
    plot_score_distribution,
    plot_confusion_matrix,
)

__all__ = [
    'debug_log_scores',
    'verify_label_alignment',
    'compute_biometric_metrics',
    'aggregate_metrics',
    'run_multi_seed_evaluation',
    'plot_roc',
    'plot_roc_multi_seed',
    'plot_det',
    'plot_score_distribution',
    'plot_confusion_matrix',
]
