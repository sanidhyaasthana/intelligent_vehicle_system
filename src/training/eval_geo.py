"""
Evaluation script for geofence model.
"""

import logging
from pathlib import Path
from typing import Dict, Any
import torch

from ..models import GeoModelWrapper
from ..datasets import build_geo_dataloader
from ..utils.metrics import compute_geo_metrics
from ..utils.seed_utils import set_seed

logger = logging.getLogger(__name__)


def eval_geo(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate trained geofence model on test set.
    
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
        or 'geo_eval'
    )
    results_dir = Path(
        config.get('results', {}).get('dir')
        or config.get('training', {}).get('results_dir', 'results/geo')
    )

    logger.info(f"Evaluating geofence model: {experiment_name}")
    
    # Build model
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'baseline')
    
    wrapper = GeoModelWrapper(model_type, device=device)
    
    # Load trained weights if probabilistic
    if model_type == 'probabilistic':
        model_path = results_dir / 'geo_model.pt'
        if model_path.exists():
            wrapper.load(str(model_path))
            logger.info(f"Loaded model from {model_path}")
    
    # Load test data
    train_loader = build_geo_dataloader(config, split='train')
    norm_params = train_loader.dataset.get_normalization_params()
    test_loader = build_geo_dataloader(config, split='test', normalization_params=norm_params)
    
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Evaluate
    predictions_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            labels = batch['label']
            
            preds = wrapper.predict(features)
            
            predictions_list.append(preds.cpu())
            labels_list.append(labels)
    
    predictions = torch.cat(predictions_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    
    # Compute metrics
    metrics = compute_geo_metrics(predictions, labels)
    
    logger.info(f"Test Metrics:")
    for key, val in metrics.items():
        if isinstance(val, float):
            logger.info(f"  {key}: {val:.4f}")
    
    return {
        'experiment': experiment_name,
        'metrics': metrics,
        'test_samples': len(labels),
    }
