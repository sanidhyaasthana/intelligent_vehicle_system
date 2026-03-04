"""
Neural network backbones for face embedding extraction.

Implements:
- ResNet-50: Standard backbone for face recognition
- MobileFaceNet: Lightweight alternative for edge deployment

Both produce 512-D embeddings suitable for ArcFace loss.

Design Choices:
- Use torchvision.models for ResNet-50 for standardization
- Implement MobileFaceNet from scratch for research transparency
- L2 normalization applied post-embedding for stable distance metrics
- No batch normalization on final embedding layer
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple, Optional


class ResNet50Backbone(nn.Module):
    """
    ResNet-50 backbone for face embedding extraction.
    
    Args:
        embedding_dim: Dimension of output embedding (default: 512)
        pretrained: Whether to load ImageNet pretrained weights
    
    Returns:
        Embeddings of shape (batch_size, embedding_dim)
    """
    
    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Load ResNet-50 from torchvision
        resnet50 = models.resnet50(pretrained=pretrained)
        
        # Remove classification head
        self.backbone = nn.Sequential(*list(resnet50.children())[:-1])
        
        # Add embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Linear(resnet50.fc.in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim, affine=False),  # No learnable params
        )
        
        logger_info = f"ResNet50 backbone initialized with embedding_dim={embedding_dim}"
        if pretrained:
            logger_info += " (ImageNet pretrained)"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract face embedding.
        
        Args:
            x: Input images (batch_size, 3, H, W)
            
        Returns:
            Normalized embeddings (batch_size, embedding_dim)
        """
        # Extract features
        feat = self.backbone(x)  # (batch_size, 2048, 1, 1)
        feat = torch.flatten(feat, 1)  # (batch_size, 2048)
        
        # Project to embedding dimension
        embedding = self.embedding_layer(feat)  # (batch_size, embedding_dim)
        
        # L2 normalization
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding


class MobileFaceNet(nn.Module):
    """
    MobileFaceNet: Lightweight face embedding network.
    
    Efficient architecture for edge deployment with ~6M parameters
    vs 23M for ResNet-50.
    
    Args:
        embedding_dim: Dimension of output embedding (default: 512)
    
    Returns:
        Embeddings of shape (batch_size, embedding_dim)
    
    Reference:
        Chen et al., "MobileFaceNets: Efficient CNNs for Face Recognition
        on Mobile Devices", CCBR 2018
    """
    
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Depthwise separable blocks
        self.dw_sep_1 = self._build_depthwise_sep_block(64, 64, 1)
        self.dw_sep_2 = self._build_depthwise_sep_block(64, 128, 2)
        self.dw_sep_3 = self._build_depthwise_sep_block(128, 128, 1)
        self.dw_sep_4 = self._build_depthwise_sep_block(128, 128, 2)
        self.dw_sep_5 = self._build_depthwise_sep_block(128, 128, 1)
        self.dw_sep_6 = self._build_depthwise_sep_block(128, 256, 2)
        self.dw_sep_7 = self._build_depthwise_sep_block(256, 256, 1)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim, affine=False),
        )
    
    def _build_depthwise_sep_block(
        self, in_channels: int, out_channels: int, stride: int
    ) -> nn.Module:
        """Build depthwise separable block."""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=3, stride=stride, padding=1,
                groups=in_channels, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract face embedding.
        
        Args:
            x: Input images (batch_size, 3, H, W)
            
        Returns:
            Normalized embeddings (batch_size, embedding_dim)
        """
        x = self.conv1(x)
        x = self.dw_sep_1(x)
        x = self.dw_sep_2(x)
        x = self.dw_sep_3(x)
        x = self.dw_sep_4(x)
        x = self.dw_sep_5(x)
        x = self.dw_sep_6(x)
        x = self.dw_sep_7(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        # Embedding projection
        embedding = self.embedding_layer(x)
        
        # L2 normalization
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding


def create_backbone(backbone_type: str = "resnet50", **kwargs) -> nn.Module:
    """
    Factory function to create a backbone.
    
    Args:
        backbone_type: 'resnet50' or 'mobilefacenet'
        **kwargs: Additional arguments passed to backbone constructor
        
    Returns:
        Initialized backbone module
    """
    if backbone_type.lower() == "resnet50":
        model = ResNet50Backbone(**kwargs)
    elif backbone_type.lower() == "mobilefacenet":
        model = MobileFaceNet(**kwargs)
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")

    # ENFORCE: freeze backbone globally (face backbone is never trainable)
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    return model
