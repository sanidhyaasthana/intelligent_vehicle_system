"""
Face Trust computation.

Computes T_face from cosine similarity, quality-adaptive margin,
and temperature scaling.
"""

import torch
import numpy as np


def compute_face_trust(
    cosine_sim: float,
    quality: float,
    m_base: float = 0.3,
    alpha: float = 0.3,
    tau: float = 0.1,
) -> float:
    """
    Compute face trust score.

    Formula:
        m_eff = m_base * (1 + alpha * (1 - Q))
        T_face = sigmoid((cosine_sim - m_eff) / tau)

    Args:
        cosine_sim: Cosine similarity between query and enrolled embedding.
        quality: Image quality in [0, 1].
        m_base: Base angular margin.
        alpha: Quality adaptation strength.
        tau: Temperature (sharpness of sigmoid).

    Returns:
        T_face in [0, 1].
    """
    Q_clamped = float(np.clip(quality, 0.0, 1.0))
    m_eff = m_base * (1.0 + alpha * (1.0 - Q_clamped))
    logit = (cosine_sim - m_eff) / tau
    T_face = 1.0 / (1.0 + np.exp(-logit))
    return float(T_face)
