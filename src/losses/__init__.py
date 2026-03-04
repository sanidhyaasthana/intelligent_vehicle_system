"""
Losses package initialization.
"""

from .arcface_loss import ArcFaceLoss, AdaptiveArcFaceLoss

__all__ = [
    'ArcFaceLoss',
    'AdaptiveArcFaceLoss',
]
