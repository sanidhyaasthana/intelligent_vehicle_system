"""
Training script for face recognition model.
"""

import logging
from pathlib import Path
from typing import Dict, Any
import torch

from ..models import create_backbone, ArcFaceHead
from ..datasets import build_face_dataloader
from ..losses import ArcFaceLoss
from ..utils.seed_utils import set_seed

logger = logging.getLogger(__name__)


def train_face(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train face recognition model with ArcFace loss.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with training results
    """
    set_seed(config.get('seed', 42))
    device = torch.device(config.get('device', 'cpu'))

    model_config = config.get('model', {})
    train_config = config.get('training', {})

    experiment_name = (
        config.get('experiment_name')
        or config.get('experiment', {}).get('name')
        or 'face_train'
    )
    results_dir = Path(
        config.get('results', {}).get('dir')
        or config.get('training', {}).get('results_dir', 'results/face')
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    backbone_type = model_config.get('backbone', 'resnet50')
    embedding_dim = model_config.get('embedding_dim', 512)
    num_classes = model_config.get('num_classes', 1000)

    logger.info(f"Training face model: {backbone_type}, dim={embedding_dim}")

    backbone = create_backbone(backbone_type, embedding_dim=embedding_dim)
    head = ArcFaceHead(embedding_dim, num_classes)

    backbone.to(device)
    head.to(device)

    # Build dataloaders
    train_loader = build_face_dataloader(config, split='train')
    val_loader = build_face_dataloader(config, split='val')

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")

    optimizer = torch.optim.SGD(
        list(backbone.parameters()) + list(head.parameters()),
        lr=train_config.get('learning_rate', 0.1),
        momentum=0.9,
        weight_decay=train_config.get('weight_decay', 5e-4),
    )
    criterion = torch.nn.CrossEntropyLoss()

    epochs = train_config.get('epochs', 50)
    best_val_acc = 0.0

    for epoch in range(epochs):
        backbone.train()
        head.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['identity'].to(device)

            embeddings = backbone(images)
            logits = head(embeddings, labels)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    # Save checkpoint
    checkpoint = {
        'backbone_state': backbone.state_dict(),
        'head_state': head.state_dict(),
        'epoch': epochs,
    }
    torch.save(checkpoint, results_dir / 'best_model.pt')
    logger.info(f"Saved model to {results_dir / 'best_model.pt'}")

    return {
        'experiment': experiment_name,
        'epochs_trained': epochs,
        'final_loss': avg_loss if n_batches > 0 else 0.0,
    }
