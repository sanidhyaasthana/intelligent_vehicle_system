"""
Models package initialization.
"""

from .backbones import ResNet50Backbone, MobileFaceNet, create_backbone
from .arcface_head import ArcFaceHead
from .adaptive_margin import compute_adaptive_margin, AdaptiveMarginScheduler
from .geo_model import BaselineGeoModel, ProbabilisticGeoModel, GeoModelWrapper
from .fusion_model import RiskFusionModel

__all__ = [
    'ResNet50Backbone',
    'MobileFaceNet',
    'create_backbone',
    'ArcFaceHead',
    'compute_adaptive_margin',
    'AdaptiveMarginScheduler',
    'BaselineGeoModel',
    'ProbabilisticGeoModel',
    'GeoModelWrapper',
    'RiskFusionModel',
]
