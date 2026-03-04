"""
Fusion model training script.

Trains the RiskFusionModel on system event data (T_face, T_geo + context -> label).
Supports both rule-based (threshold optimization) and learned (MLP) fusion.
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any

from ..models.fusion_model import RiskFusionModel
from ..datasets.system_event_dataset import SystemEventDataset, build_system_event_dataloader
from ..utils.seed_utils import set_seed

logger = logging.getLogger(__name__)


def train_fusion(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train fusion model on system events.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with training results
    """
    seed = config.get('seed', config.get('experiment', {}).get('seed', 42))
    set_seed(seed)
    device = config.get('device', 'cpu')

    fusion_config = config.get('fusion', {})
    training_config = config.get('training', {})
    mode = fusion_config.get('mode', 'rule_based')

    logger.info(f"Training fusion model in '{mode}' mode")

    # Build dataloaders
    train_loader = build_system_event_dataloader(config, split='train')
    val_loader = build_system_event_dataloader(config, split='val')

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")

    if mode == 'rule_based':
        return _train_rule_based(config, train_loader, val_loader)
    else:
        return _train_learned(config, train_loader, val_loader, device)


def _train_rule_based(config, train_loader, val_loader):
    """Optimize thresholds for rule-based fusion."""
    fusion_config = config.get('fusion', {}).get('rule_based', {})
    alpha = fusion_config.get('alpha', 0.6)
    beta = fusion_config.get('beta', 0.4)

    model = RiskFusionModel(mode='rule_based', alpha=alpha, beta=beta)

    # Collect validation data
    T_face_list, T_geo_list, labels_list = [], [], []
    for batch in val_loader:
        T_face_list.append(batch['T_face'].numpy())
        T_geo_list.append(batch['T_geo'].numpy())
        labels_list.append(np.array([batch['label']]).flatten() if isinstance(batch['label'], int) else batch['label'].numpy())

    T_face_val = np.concatenate(T_face_list)
    T_geo_val = np.concatenate(T_geo_list)
    labels_val = np.concatenate(labels_list)

    # Optimize thresholds
    model.optimize_thresholds(T_face_val, T_geo_val, labels_val, metric='eer')

    # Save
    results_dir = Path(config.get('results', {}).get('dir', 'results/fusion/full_system'))
    results_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(results_dir / 'fusion_model.pt'))

    return {
        'mode': 'rule_based',
        'tau1': model.tau1,
        'tau2': model.tau2,
        'alpha': model.alpha,
        'beta': model.beta,
    }


def _train_learned(config, train_loader, val_loader, device):
    """Train learned MLP fusion model."""
    fusion_config = config.get('fusion', {}).get('learned', {})
    training_config = config.get('training', {})

    hidden_dims = fusion_config.get('hidden_dims', [32, 16])
    dropout = fusion_config.get('dropout', 0.1)

    # Input dim = number of features in SystemEventDataset.__getitem__
    input_dim = 6  # T_face, T_geo, quality, is_inside, time_sin, time_cos

    model = RiskFusionModel(mode='learned', hidden_dim=hidden_dims[0])
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.get('learning_rate', 0.001),
        weight_decay=training_config.get('weight_decay', 0.0001),
    )
    criterion = nn.BCELoss()

    epochs = training_config.get('epochs', 50)
    best_val_loss = float('inf')
    patience = training_config.get('early_stopping', {}).get('patience', 15)
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            T_face = batch['T_face'].float().to(device)
            T_geo = batch['T_geo'].float().to(device)

            result = model(T_face, T_geo)
            # Use T_fused as prediction score
            T_fused = result.get('T_fused', 1.0 - result['risk'])
            labels = batch['label'].float().to(device)

            loss = criterion(T_fused, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                T_face = batch['T_face'].float().to(device)
                T_geo = batch['T_geo'].float().to(device)

                result = model(T_face, T_geo)
                T_fused = result.get('T_fused', 1.0 - result['risk'])
                labels = batch['label'].float().to(device)

                loss = criterion(T_fused, labels)
                val_loss += loss.item()
                n_val += 1

        val_loss /= max(n_val, 1)

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            results_dir = Path(config.get('results', {}).get('dir', 'results/fusion/full_system'))
            results_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(results_dir / 'fusion_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    return {
        'mode': 'learned',
        'epochs_trained': epoch + 1,
        'best_val_loss': best_val_loss,
    }
