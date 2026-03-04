"""
Training script for geofence model.
"""

import logging
from pathlib import Path
from typing import Dict, Any
import torch

from ..models import GeoModelWrapper
from ..datasets import build_geo_dataloader
from ..utils.seed_utils import set_seed

logger = logging.getLogger(__name__)


def train_geo(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train geofence model.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with training results
    """
    set_seed(config.get('seed', 42))
    device = config.get('device', 'cpu')

    model_config = config.get('model', {})
    train_config = config.get('training', {})
    model_type = model_config.get('type', 'probabilistic')

    experiment_name = (
        config.get('experiment_name')
        or config.get('experiment', {}).get('name')
        or 'geo_train'
    )
    results_dir = Path(
        config.get('results', {}).get('dir')
        or config.get('training', {}).get('results_dir', 'results/geo')
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training geofence model: {model_type}")

    wrapper = GeoModelWrapper(model_type, device=device)

    # Build dataloaders
    train_loader = build_geo_dataloader(config, split='train')
    val_loader = build_geo_dataloader(
        config, split='val',
        normalization_params=train_loader.dataset.get_normalization_params()
    )

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")

    # Train
    history = wrapper.fit(
        train_loader,
        val_loader,
        num_epochs=train_config.get('epochs', 50),
        learning_rate=train_config.get('learning_rate', 0.01),
        weight_decay=train_config.get('weight_decay', 1e-4),
    )

    # Save model
    model_path = results_dir / 'geo_model.pt'
    wrapper.save(str(model_path))

    return {
        'experiment': experiment_name,
        'model_type': model_type,
        'model_path': str(model_path),
        'history': history,
    }
