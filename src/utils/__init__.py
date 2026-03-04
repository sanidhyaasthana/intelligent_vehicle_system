"""Utility modules for the vehicle authorization framework."""

from .config_utils import load_config, save_config, merge_configs
from .seed_utils import set_seed, get_seed
from .logger import Logger, get_logger
from .metrics import (
    compute_accuracy,
    compute_far,
    compute_frr,
    compute_tar,
    compute_eer,
    compute_roc,
    compute_auc,
    compute_confusion_matrix
)
from .quality_metrics import compute_quality_score, QualityEstimator
from .geo_utils import (
    haversine_distance,
    sample_gps_noise,
    point_in_polygon,
    point_in_circle
)
from .model_validator import ModelValidator, InferenceValidator, EmbeddingValidator
from .score_calibration import ThresholdCalibrator, ScoreNormalizer, TrustScoreValidator
from .scientific_logging import (
    PhaseLogger, FailureAssertion, ReproducibilityLogger,
    DiagnosticReport, ExpectedMetricRanges
)

__all__ = [
    'load_config', 'save_config', 'merge_configs',
    'set_seed', 'get_seed',
    'Logger', 'get_logger',
    'compute_accuracy', 'compute_far', 'compute_frr', 'compute_tar',
    'compute_eer', 'compute_roc', 'compute_auc', 'compute_confusion_matrix',
    'compute_quality_score', 'QualityEstimator',
    'haversine_distance', 'sample_gps_noise', 'point_in_polygon', 'point_in_circle',
    'ModelValidator', 'InferenceValidator', 'EmbeddingValidator',
    'ThresholdCalibrator', 'ScoreNormalizer', 'TrustScoreValidator',
    'PhaseLogger', 'FailureAssertion', 'ReproducibilityLogger',
    'DiagnosticReport', 'ExpectedMetricRanges',
]
