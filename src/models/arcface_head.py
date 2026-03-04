"""
ArcFace classification head with weight normalization.

The ArcFace head applies weight normalization and bias-free classification
for improved numerical stability and interpretability.

Design Choices:
- Weight normalization: All weights normalized to unit norm
- No bias term: Simplifies interpretation as distance-based decision
- Large margin (0.5 rad ≈ 30°) for intra-class cohesion
- Scale parameter s ≈ 64 for numerical stability
"""

import torch
import torch.nn as nn
import math


class ArcFaceHead(nn.Module):
    """
    ArcFace classification head with weight normalization.
    
    Args:
        embedding_dim: Dimension of input embeddings (e.g., 512)
        num_classes: Number of identity classes
        scale: Temperature parameter s for scaling logits (default: 64)
        margin: Angular margin m in radians (default: 0.5)
    
    Shape:
        - Input: (batch_size, embedding_dim)
        - Output: (batch_size, num_classes)
    
    Reference:
        Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face
        Recognition", CVPR 2019
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.5,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        
        # Weight matrix: (num_classes, embedding_dim)
        # Will be normalized to unit norm in forward pass
        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_dim))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute ArcFace logits with angular margin.
        
        Args:
            embeddings: Face embeddings (batch_size, embedding_dim)
                       Should be L2-normalized
            labels: Class labels (batch_size,)
            
        Returns:
            Logits with margin applied (batch_size, num_classes)
        
        Formula:
            1. cos(θ_j) = W_j^T * x / (||W_j|| * ||x||)
            2. For true class: θ_i → θ_i + m
            3. logit_j = s * cos(θ_j)
        """
        # Normalize weight matrix
        w = torch.nn.functional.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarities
        cos_sim = torch.matmul(embeddings, w.t())  # (batch_size, num_classes)
        
        # Clamp to avoid numerical issues with arccos
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        
        # Compute angles
        theta = torch.acos(cos_sim)  # (batch_size, num_classes)
        
        # Create one-hot label encoding
        one_hot = torch.zeros_like(cos_sim)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        
        # Apply angular margin to correct class
        theta = theta + one_hot * self.margin
        
        # Compute final logits
        logits = self.scale * torch.cos(theta)
        
        return logits
    
    def forward_test(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity scores without margin (for testing/verification).
        
        Args:
            embeddings: Face embeddings (batch_size, embedding_dim)
            
        Returns:
            Cosine similarities (batch_size, num_classes)
        """
        w = torch.nn.functional.normalize(self.weight, p=2, dim=1)
        cos_sim = torch.matmul(embeddings, w.t())
        return cos_sim
