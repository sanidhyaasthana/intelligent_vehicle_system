"""
Adaptive margin computation for quality-aware face verification.

This module computes per-sample angular margins based on image quality scores.
The intuition: 
- High-quality images (sharp, bright, unoccluded) should have tighter
  verification margins (harder to verify, higher m)
- Low-quality images (blurry, dark, occluded) should have looser margins
  (easier to verify, lower m)

This allows the system to be more tolerant to degraded images while
remaining strict for high-quality captures.

Design Choices:
- Linear interpolation between m_min and m_max
- Quality score q in [0, 1] where 1 = perfect
- Margin formula: m_i = m_min + (1 - q_i) * (m_max - m_min)
- Fully vectorized for efficiency
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union


def compute_adaptive_margin(
    quality_scores: Union[torch.Tensor, np.ndarray],
    m_min: float = 0.2,
    m_max: float = 0.5,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Compute adaptive angular margins based on quality scores.
    
    Args:
        quality_scores: Image quality scores in [0, 1]
                       (batch_size,) or scalar
        m_min: Minimum margin for highest quality images (default: 0.2 rad)
        m_max: Maximum margin for lowest quality images (default: 0.5 rad)
        
    Returns:
        Adaptive margins with same shape as input
        
    Formula:
        m_i = m_min + (1 - q_i) * (m_max - m_min)
    
    Interpretation:
        - Perfect quality (q=1): m = m_min (strict)
        - Poor quality (q=0): m = m_max (lenient)
    
    Example:
        >>> q = torch.tensor([1.0, 0.5, 0.0])
        >>> m = compute_adaptive_margin(q, m_min=0.2, m_max=0.5)
        >>> print(m)  # tensor([0.2, 0.35, 0.5])
    """
    if isinstance(quality_scores, np.ndarray):
        return m_min + (1.0 - quality_scores) * (m_max - m_min)
    else:
        return m_min + (1.0 - quality_scores) * (m_max - m_min)


def get_adaptive_margin_stats(
    margins: Union[torch.Tensor, np.ndarray],
) -> dict:
    """
    Compute statistics of adaptive margins for logging/monitoring.
    
    Args:
        margins: Adaptive margin values
        
    Returns:
        Dictionary with keys: mean, std, min, max
    """
    if isinstance(margins, torch.Tensor):
        margins = margins.detach().cpu().numpy()
    
    return {
        'mean': float(np.mean(margins)),
        'std': float(np.std(margins)),
        'min': float(np.min(margins)),
        'max': float(np.max(margins)),
    }


class AdaptiveMarginScheduler:
    """
    Scheduler for gradually changing m_min and m_max during training.
    
    Strategy: Start with looser margins (m_max > 0.5) and gradually
    tighten over epochs for curriculum learning.
    
    Args:
        initial_m_min: Initial m_min (default: 0.1)
        initial_m_max: Initial m_max (default: 0.7)
        final_m_min: Final m_min (default: 0.2)
        final_m_max: Final m_max (default: 0.5)
        total_epochs: Total number of training epochs
    """
    
    def __init__(
        self,
        initial_m_min: float = 0.1,
        initial_m_max: float = 0.7,
        final_m_min: float = 0.2,
        final_m_max: float = 0.5,
        total_epochs: int = 100,
    ):
        self.initial_m_min = initial_m_min
        self.initial_m_max = initial_m_max
        self.final_m_min = final_m_min
        self.final_m_max = final_m_max
        self.total_epochs = total_epochs
    
    def get_margins(self, epoch: int) -> tuple:
        """
        Get m_min and m_max for a given epoch.
        
        Args:
            epoch: Current epoch (0-indexed)
            
        Returns:
            Tuple of (m_min, m_max) for this epoch
        """
        progress = min(epoch / self.total_epochs, 1.0)
        
        m_min = self.initial_m_min + (self.final_m_min - self.initial_m_min) * progress
        m_max = self.initial_m_max + (self.final_m_max - self.initial_m_max) * progress
        
        return m_min, m_max


def visualize_adaptive_margin(quality_scores: np.ndarray, m_min: float, m_max: float):
    """
    Create a visualization of how margins change with quality.
    
    Args:
        quality_scores: Array of quality scores
        m_min: Minimum margin
        m_max: Maximum margin
    """
    import matplotlib.pyplot as plt
    
    q_range = np.linspace(0, 1, 100)
    m_range = compute_adaptive_margin(q_range, m_min, m_max)
    
    plt.figure(figsize=(10, 6))
    plt.plot(q_range, m_range, 'b-', linewidth=2, label='Adaptive Margin')
    plt.axhline(y=m_min, color='g', linestyle='--', label=f'm_min = {m_min:.2f}')
    plt.axhline(y=m_max, color='r', linestyle='--', label=f'm_max = {m_max:.2f}')
    
    # Overlay histogram of actual quality scores
    plt.hist(quality_scores, bins=30, alpha=0.3, range=(0, 1), label='Quality Distribution')
    
    plt.xlabel('Image Quality Score q', fontsize=12)
    plt.ylabel('Angular Margin m (radians)', fontsize=12)
    plt.title('Adaptive Margin Function', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()
