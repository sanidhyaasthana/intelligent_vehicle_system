"""
Image Quality Metrics
=====================

Implements simple, interpretable quality metrics for face images.
These metrics are used to compute adaptive margins in the face verification system.

Quality Factors:
- Blur: Measured via variance of Laplacian (higher = sharper)
- Brightness: Mean pixel intensity (optimal around 127)
- Contrast: Standard deviation of pixel intensities (higher = more contrast)

Design Choices:
- All metrics normalized to [0, 1] for consistency
- Combined score uses weighted geometric mean for robustness
- Supports both numpy arrays and PyTorch tensors
"""

import numpy as np
import torch
from typing import Union, Dict, Optional, Tuple
import cv2


def compute_laplacian_variance(image: np.ndarray) -> float:
    """
    Compute blur metric using variance of Laplacian.

    Higher values indicate sharper images.
    This is a well-established no-reference blur metric.

    Args:
        image: Grayscale image as numpy array (H, W) or (H, W, 1).

    Returns:
        Variance of Laplacian response.

    Reference:
        Pech-Pacheco et al., "Diatom autofocusing in brightfield microscopy:
        a comparative study", ICPR 2000.
    """
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image = image[:, :, 0]

    # Ensure uint8 for cv2
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)

    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return float(np.var(laplacian))


def compute_brightness(image: np.ndarray) -> float:
    """
    Compute brightness as mean pixel intensity.

    Args:
        image: Image as numpy array, values in [0, 255] or [0, 1].

    Returns:
        Mean intensity normalized to [0, 1].
    """
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            # Convert to grayscale using luminance weights
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            gray = image[:, :, 0]
    else:
        gray = image

    mean_val = np.mean(gray)

    # Normalize to [0, 1]
    if mean_val > 1:
        mean_val = mean_val / 255.0

    return float(mean_val)


def compute_contrast(image: np.ndarray) -> float:
    """
    Compute contrast as standard deviation of pixel intensities.

    Higher values indicate more contrast.

    Args:
        image: Image as numpy array.

    Returns:
        Standard deviation normalized to [0, 1].
    """
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            gray = image[:, :, 0]
    else:
        gray = image

    std_val = np.std(gray)

    # Normalize assuming max std is about 127 for uint8 images
    if gray.max() > 1:
        std_val = std_val / 127.0
    else:
        std_val = std_val / 0.5

    return float(min(std_val, 1.0))


def normalize_blur_score(laplacian_var: float,
                         threshold: float = 100.0) -> float:
    """
    Normalize Laplacian variance to a quality score in [0, 1].

    Uses a sigmoid-like transformation for smooth normalization.

    Args:
        laplacian_var: Variance of Laplacian.
        threshold: Value at which score is 0.5.

    Returns:
        Normalized blur quality score (higher = sharper).
    """
    # Sigmoid normalization: score = var / (var + threshold)
    score = laplacian_var / (laplacian_var + threshold)
    return float(score)


def normalize_brightness_score(brightness: float,
                               optimal_range: Tuple[float, float] = (0.2, 0.8)) -> float:
    """
    Normalize brightness to a quality score in [0, 1].

    Optimal brightness is in the middle range; too dark or too bright is penalized.

    Args:
        brightness: Brightness value in [0, 1].
        optimal_range: Range of optimal brightness values.

    Returns:
        Normalized brightness quality score.
    """
    low, high = optimal_range
    mid = (low + high) / 2

    if low <= brightness <= high:
        # Within optimal range: high score
        return 1.0
    elif brightness < low:
        # Too dark: linear decrease
        return max(0.0, brightness / low)
    else:
        # Too bright: linear decrease
        return max(0.0, (1.0 - brightness) / (1.0 - high))


def normalize_contrast_score(contrast: float,
                             threshold: float = 0.2) -> float:
    """
    Normalize contrast to a quality score in [0, 1].

    Low contrast images get low scores.

    Args:
        contrast: Contrast value in [0, 1].
        threshold: Minimum acceptable contrast.

    Returns:
        Normalized contrast quality score.
    """
    if contrast >= threshold:
        return 1.0
    else:
        return contrast / threshold


def compute_quality_score(image: Union[np.ndarray, torch.Tensor],
                          method: str = 'combined',
                          weights: Optional[Dict[str, float]] = None,
                          laplacian_threshold: float = 100.0,
                          brightness_range: Tuple[float, float] = (0.2, 0.8),
                          contrast_threshold: float = 0.2) -> float:
    """
    Compute overall quality score for a face image.

    Combines blur, brightness, and contrast metrics into a single score.

    Args:
        image: Input image as numpy array or PyTorch tensor.
               Shape: (H, W), (H, W, C), or (C, H, W) for tensors.
        method: Scoring method ('combined', 'blur', 'brightness', 'contrast').
        weights: Optional dictionary of weights for each metric.
        laplacian_threshold: Threshold for blur normalization.
        brightness_range: Optimal brightness range.
        contrast_threshold: Minimum contrast threshold.

    Returns:
        Quality score in [0, 1].

    Example:
        >>> import numpy as np
        >>> img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        >>> score = compute_quality_score(img)
        >>> print(f"Quality: {score:.3f}")
    """
    # Convert tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        # Handle CHW format
        if len(image.shape) == 3 and image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))

    # Ensure proper range
    if image.max() <= 1.0:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.astype(np.uint8)

    # Default weights
    if weights is None:
        weights = {'blur': 0.4, 'brightness': 0.3, 'contrast': 0.3}

    if method == 'blur':
        laplacian_var = compute_laplacian_variance(image_uint8)
        return normalize_blur_score(laplacian_var, laplacian_threshold)

    elif method == 'brightness':
        brightness = compute_brightness(image_uint8)
        return normalize_brightness_score(brightness, brightness_range)

    elif method == 'contrast':
        contrast = compute_contrast(image_uint8)
        return normalize_contrast_score(contrast, contrast_threshold)

    elif method == 'combined':
        # Compute individual scores
        laplacian_var = compute_laplacian_variance(image_uint8)
        blur_score = normalize_blur_score(laplacian_var, laplacian_threshold)

        brightness = compute_brightness(image_uint8)
        brightness_score = normalize_brightness_score(brightness, brightness_range)

        contrast = compute_contrast(image_uint8)
        contrast_score = normalize_contrast_score(contrast, contrast_threshold)

        # Weighted combination (geometric mean for robustness)
        # Using weighted geometric mean: prod(score_i ^ w_i)
        scores = {
            'blur': blur_score,
            'brightness': brightness_score,
            'contrast': contrast_score
        }

        # Add small epsilon to avoid log(0)
        eps = 1e-8
        log_sum = sum(weights[k] * np.log(scores[k] + eps) for k in weights)
        combined = np.exp(log_sum)

        return float(np.clip(combined, 0.0, 1.0))

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_quality_scores_batch(images: Union[np.ndarray, torch.Tensor],
                                 method: str = 'combined',
                                 **kwargs) -> np.ndarray:
    """
    Compute quality scores for a batch of images.

    Args:
        images: Batch of images, shape (B, C, H, W) or (B, H, W, C).
        method: Scoring method.
        **kwargs: Additional arguments for compute_quality_score.

    Returns:
        Array of quality scores, shape (B,).
    """
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()

    # Determine format
    if len(images.shape) == 4:
        batch_size = images.shape[0]
        # Check if CHW or HWC
        if images.shape[1] in [1, 3]:
            # BCHW -> BHWC
            images = np.transpose(images, (0, 2, 3, 1))
    else:
        raise ValueError(f"Expected 4D batch, got shape {images.shape}")

    scores = np.array([
        compute_quality_score(images[i], method=method, **kwargs)
        for i in range(batch_size)
    ])

    return scores


class QualityEstimator:
    """
    Quality estimator with configurable parameters and caching.

    Designed for use during training where the same configuration
    is applied to many images.

    Attributes:
        method: Quality estimation method.
        weights: Metric weights for combined method.
        config: Full configuration dictionary.

    Example:
        >>> estimator = QualityEstimator(method='combined')
        >>> scores = estimator.estimate_batch(image_batch)
        >>> stats = estimator.get_statistics(scores)
    """

    def __init__(self,
                 method: str = 'combined',
                 weights: Optional[Dict[str, float]] = None,
                 laplacian_threshold: float = 100.0,
                 brightness_range: Tuple[float, float] = (0.2, 0.8),
                 contrast_threshold: float = 0.2):
        """
        Initialize quality estimator.

        Args:
            method: Estimation method.
            weights: Metric weights.
            laplacian_threshold: Blur normalization threshold.
            brightness_range: Optimal brightness range.
            contrast_threshold: Minimum contrast threshold.
        """
        self.method = method
        self.weights = weights or {'blur': 0.4, 'brightness': 0.3, 'contrast': 0.3}
        self.laplacian_threshold = laplacian_threshold
        self.brightness_range = brightness_range
        self.contrast_threshold = contrast_threshold

        # Statistics tracking
        self._scores_history = []

    def estimate(self, image: Union[np.ndarray, torch.Tensor]) -> float:
        """Estimate quality for a single image."""
        score = compute_quality_score(
            image,
            method=self.method,
            weights=self.weights,
            laplacian_threshold=self.laplacian_threshold,
            brightness_range=self.brightness_range,
            contrast_threshold=self.contrast_threshold
        )
        self._scores_history.append(score)
        return score

    def estimate_batch(self, images: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Estimate quality for a batch of images."""
        scores = compute_quality_scores_batch(
            images,
            method=self.method,
            weights=self.weights,
            laplacian_threshold=self.laplacian_threshold,
            brightness_range=self.brightness_range,
            contrast_threshold=self.contrast_threshold
        )
        self._scores_history.extend(scores.tolist())
        return scores

    def get_statistics(self,
                       scores: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute statistics for quality scores.

        Args:
            scores: Optional scores array. Uses history if None.

        Returns:
            Dictionary with mean, std, min, max, median.
        """
        if scores is None:
            scores = np.array(self._scores_history)

        if len(scores) == 0:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}

        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores))
        }

    def reset_history(self) -> None:
        """Clear score history."""
        self._scores_history = []

    def get_history(self) -> np.ndarray:
        """Get score history as numpy array."""
        return np.array(self._scores_history)


def compute_detailed_quality(image: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
    """
    Compute detailed quality breakdown for analysis.

    Args:
        image: Input image.

    Returns:
        Dictionary with individual and combined scores.
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        if len(image.shape) == 3 and image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))

    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # Raw metrics
    laplacian_var = compute_laplacian_variance(image)
    brightness = compute_brightness(image)
    contrast = compute_contrast(image)

    # Normalized scores
    blur_score = normalize_blur_score(laplacian_var)
    brightness_score = normalize_brightness_score(brightness)
    contrast_score = normalize_contrast_score(contrast)

    # Combined score
    combined = compute_quality_score(image, method='combined')

    return {
        'laplacian_variance': laplacian_var,
        'brightness_raw': brightness,
        'contrast_raw': contrast,
        'blur_score': blur_score,
        'brightness_score': brightness_score,
        'contrast_score': contrast_score,
        'combined_score': combined
    }
