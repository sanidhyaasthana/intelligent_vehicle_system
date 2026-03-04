"""
EmbeddingExtractor: wraps backbone for face embedding extraction.
Provides .extract() for batch inference.
"""
import torch
from .backbones import create_backbone

class EmbeddingExtractor:
    def __init__(self, backbone_name="resnet50", embedding_dim=512, device="cpu"):
        self.device = torch.device(device)
        self.backbone = create_backbone(backbone_name, embedding_dim=embedding_dim)
        self.backbone.to(self.device)

    def extract(self, batch_tensor):
        """
        Args:
            batch_tensor: torch.Tensor of shape (batch, 3, H, W), float32, range [0,1]
        Returns:
            torch.Tensor of shape (batch, embedding_dim), L2-normalized
        """
        self.backbone.eval()
        with torch.no_grad():
            batch_tensor = batch_tensor.to(self.device)
            embeddings = self.backbone(batch_tensor)
        return embeddings