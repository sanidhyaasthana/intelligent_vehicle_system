"""
Quality Estimator module.

Provides a function to estimate face image quality from raw pixel data.
Used in the full evaluation pipeline for adaptive margin computation.
"""

import numpy as np
import cv2


def estimate_quality_from_image(image: np.ndarray) -> float:
    """
    Estimate image quality from a face image.

    Combines blur (Laplacian variance), brightness, and contrast
    into a single quality score in [0, 1].

    Args:
        image: RGB image as numpy array (H, W, 3), dtype uint8 or float32.

    Returns:
        Quality score in [0, 1], where 1 = high quality.
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.astype(np.uint8)
    else:
        img_uint8 = image.astype(np.uint8)

    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    # Blur metric: Laplacian variance
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = laplacian_var / (laplacian_var + 100.0)

    # Brightness metric
    brightness = float(np.mean(gray)) / 255.0
    if 0.2 <= brightness <= 0.8:
        brightness_score = 1.0
    elif brightness < 0.2:
        brightness_score = max(0.0, brightness / 0.2)
    else:
        brightness_score = max(0.0, (1.0 - brightness) / 0.2)

    # Contrast metric
    contrast = float(np.std(gray)) / 127.0
    contrast_score = min(contrast / 0.2, 1.0) if contrast < 0.2 else 1.0

    # Weighted combination
    quality = 0.5 * blur_score + 0.25 * brightness_score + 0.25 * contrast_score
    return float(np.clip(quality, 0.0, 1.0))
