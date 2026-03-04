"""
Loss functions for face recognition: ArcFace and Adaptive ArcFace.

Standard ArcFace:
    L = - log( e^(s*cos(θ_yi + m)) / (e^(s*cos(θ_yi + m)) + Σ_j≠yi e^(s*cos(θ_j))) )

Adaptive ArcFace:
    Same formula but m_i varies per sample based on quality q_i:
    m_i = m_min + (1 - q_i) * (m_max - m_min)

Design Choices:
- Softmax cross-entropy for numerical stability
- Detach margins from computation graph (not learnable)
- Log summation trick for numerical stability
- Support for weighted loss (for handling class imbalance)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ArcFaceLoss(nn.Module):
    """
    Standard ArcFace loss with fixed angular margin.
    
    Args:
        num_classes: Number of identity classes
        embedding_dim: Dimension of embeddings
        scale: Temperature scaling factor s (default: 64)
        margin: Angular margin m in radians (default: 0.5)
        label_smoothing: Label smoothing factor (default: 0.0)
    
    Input:
        logits: Pre-computed ArcFace logits from ArcFaceHead (batch, num_classes)
        labels: Class labels (batch,)
        
    Output:
        Scalar loss value
    
    Reference:
        Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face
        Recognition", CVPR 2019
    """
    
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        scale: float = 64.0,
        margin: float = 0.5,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.margin = margin
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute ArcFace loss.
        
        Args:
            logits: Pre-computed ArcFace logits (batch, num_classes)
            labels: Target class labels (batch,)
            weights: Optional per-sample weights (batch,)
            
        Returns:
            Scalar loss
        """
        # Cross-entropy loss with label smoothing
        loss = F.cross_entropy(
            logits, labels,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        
        # Apply sample weights if provided
        if weights is not None:
            loss = loss * weights
            loss = loss.mean()
        else:
            loss = loss.mean()
        
        return loss


class AdaptiveArcFaceLoss(nn.Module):
    """
    Adaptive ArcFace loss with quality-dependent angular margins.
    
    Per-sample margins m_i are computed from image quality scores:
    m_i = m_min + (1 - q_i) * (m_max - m_min)
    
    Then standard ArcFace formula is applied with margin m_i for sample i.
    
    Args:
        num_classes: Number of identity classes
        embedding_dim: Dimension of embeddings
        scale: Temperature scaling factor s (default: 64)
        m_min: Minimum margin for high-quality images (default: 0.2)
        m_max: Maximum margin for low-quality images (default: 0.5)
        label_smoothing: Label smoothing factor (default: 0.0)
        log_margin_stats: Whether to log margin statistics (default: True)
    
    Input:
        embeddings: Face embeddings (batch, embedding_dim), L2-normalized
        labels: Class labels (batch,)
        quality_scores: Image quality scores in [0, 1] (batch,)
        weights: Optional per-sample weights (batch,)
        
    Output:
        Scalar loss
    """
    
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        scale: float = 64.0,
        m_min: float = 0.2,
        m_max: float = 0.5,
        label_smoothing: float = 0.0,
        log_margin_stats: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.m_min = m_min
        self.m_max = m_max
        self.label_smoothing = label_smoothing
        self.log_margin_stats = log_margin_stats
        
        # Weight matrix (will be normalized in forward)
        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # Stats tracking
        self.register_buffer('num_updates', torch.tensor(0))
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        quality_scores: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Adaptive ArcFace loss.
        
        Args:
            embeddings: Face embeddings (batch, embedding_dim), L2-normalized
            labels: Target class labels (batch,)
            quality_scores: Image quality in [0, 1] (batch,)
            weights: Optional per-sample weights (batch,)
            
        Returns:
            Scalar loss
        
        Implementation Details:
        1. Normalize weight matrix to unit norm
        2. Compute cosine similarities: cos(θ_j) = W_j^T * x
        3. Compute adaptive margins: m_i = m_min + (1 - q_i) * (m_max - m_min)
        4. Apply angular margin: θ_i → θ_i + m_i for true class
        5. Compute logits: s * cos(θ)
        6. Apply cross-entropy loss
        """
        # Normalize weights
        w = F.normalize(self.weight, p=2, dim=1)  # (num_classes, embedding_dim)
        
        # Compute cosine similarities
        cos_sim = torch.matmul(embeddings, w.t())  # (batch, num_classes)
        
        # Clamp to avoid numerical issues with acos (requires [-1, 1])
        # Use more conservative bounds to avoid NaN at edge cases
        cos_sim = torch.clamp(cos_sim, -0.99999, 0.99999)
        
        # Compute angles (safe with more conservative clamping)
        theta = torch.acos(cos_sim)  # (batch, num_classes)
        
        # Compute adaptive margins
        # m_i = m_min + (1 - q_i) * (m_max - m_min)
        m_adaptive = self.m_min + (1.0 - quality_scores.unsqueeze(1)) * (self.m_max - self.m_min)
        
        # Create one-hot label encoding
        one_hot = torch.zeros_like(cos_sim)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        
        # Apply margin only to correct class
        m_applied = one_hot * m_adaptive
        theta = theta + m_applied
        
        # Compute logits with scale
        logits = self.scale * torch.cos(theta)
        
        # Cross-entropy loss
        loss = F.cross_entropy(
            logits, labels,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        
        # Apply sample weights if provided
        if weights is not None:
            loss = loss * weights
            loss = loss.mean()
        else:
            loss = loss.mean()
        
        # Log margin statistics
        if self.log_margin_stats and self.num_updates % 100 == 0:
            m_stats = self._compute_margin_stats(m_adaptive)
            q_stats = self._compute_quality_stats(quality_scores)
            logger.info(
                f"Adaptive Margin Stats (update {self.num_updates.item()}): "
                f"m_mean={m_stats['mean']:.4f}, m_std={m_stats['std']:.4f}, "
                f"m_range=[{m_stats['min']:.4f}, {m_stats['max']:.4f}], "
                f"q_mean={q_stats['mean']:.4f}, q_std={q_stats['std']:.4f}"
            )
        
        self.num_updates += 1
        
        return loss
    
    def _compute_margin_stats(self, margins: torch.Tensor) -> dict:
        """Compute statistics of margins."""
        margins = margins.detach().cpu()
        return {
            'mean': float(margins.mean()),
            'std': float(margins.std()),
            'min': float(margins.min()),
            'max': float(margins.max()),
        }
    
    def _compute_quality_stats(self, quality_scores: torch.Tensor) -> dict:
        """Compute statistics of quality scores."""
        q = quality_scores.detach().cpu()
        return {
            'mean': float(q.mean()),
            'std': float(q.std()),
            'min': float(q.min()),
            'max': float(q.max()),
        }
    
    def forward_test(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity scores without margin (for testing).
        
        Args:
            embeddings: Face embeddings (batch, embedding_dim)
            
        Returns:
            Cosine similarities (batch, num_classes)
        """
        w = F.normalize(self.weight, p=2, dim=1)
        cos_sim = torch.matmul(embeddings, w.t())
        return cos_sim
