"""
Evaluation script for face recognition model.
"""

import torch
import logging
from pathlib import Path
from typing import Dict, Any

from ..models import create_backbone, ArcFaceHead
from ..datasets import build_face_dataloader
from ..utils.metrics import compute_face_metrics, save_roc_curve
from ..utils.seed_utils import set_seed

logger = logging.getLogger(__name__)


def eval_face(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate trained face recognition model on test set.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with evaluation metrics
    """
    set_seed(config.get('seed', 42))
    device = torch.device(config.get('device', 'cuda'))

    experiment_name = (
        config.get('experiment_name')
        or config.get('experiment', {}).get('name')
        or 'face_eval'
    )
    results_dir = Path(
        config.get('results', {}).get('dir')
        or config.get('training', {}).get('results_dir', 'results/face')
    )

    logger.info(f"Evaluating face model: {experiment_name}")
    
    # Load model
    model_config = config.get('model', {})
    backbone = create_backbone(
        model_config.get('backbone', 'resnet50'),
        embedding_dim=model_config.get('embedding_dim', 512),
    )
    head = ArcFaceHead(
        model_config.get('embedding_dim', 512),
        model_config.get('num_classes', 1000),
    )
    
    checkpoint_path = results_dir / 'best_model.pt'
    if not checkpoint_path.exists():
        logger.error(f"Model not found: {checkpoint_path}")
        return {'error': 'Model checkpoint not found'}
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone.load_state_dict(checkpoint['backbone_state'])
    head.load_state_dict(checkpoint['head_state'])
    
    backbone.to(device)
    head.to(device)
    backbone.eval()
    head.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    
    # Load test data
    test_loader = build_face_dataloader(config, split='test')
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Evaluate
    embeddings_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['identity']
            
            embeddings = backbone(images)
            
            embeddings_list.append(embeddings.cpu())
            labels_list.append(labels)
    
    embeddings = torch.cat(embeddings_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    
    # Compute metrics
    metrics = compute_face_metrics(embeddings, labels)
    
    logger.info(f"Test Metrics:")
    for key, val in metrics.items():
        if isinstance(val, float):
            logger.info(f"  {key}: {val:.4f}")
    
    # Save ROC curve
    roc_path = results_dir / 'roc_curve.png'
    save_roc_curve(embeddings, labels, roc_path)
    logger.info(f"Saved ROC curve to {roc_path}")
    
    return {
        'experiment': experiment_name,
        'metrics': metrics,
        'test_samples': len(labels),
    }
