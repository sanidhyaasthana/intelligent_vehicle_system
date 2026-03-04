"""
Geofence Trust computation.

Computes T_geo from distance using Gaussian CDF (error function).
"""

import numpy as np
import math


def geo_trust(
    distance: float,
    radius: float,
    sigma: float,
) -> float:
    """
    Compute geofence trust score using Gaussian CDF.

    Formula:
        z = (radius - distance) / sigma
        T_geo = 0.5 * (1 + erf(z / sqrt(2)))

    Args:
        distance: Distance from user to geofence center in meters.
        radius: Geofence radius in meters.
        sigma: Standard deviation for soft boundary.

    Returns:
        T_geo in [0, 1]. Higher = more inside geofence.
    """
    z = (radius - distance) / sigma
    T_geo = 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
    return float(T_geo)
