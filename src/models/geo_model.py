"""
Geofence Trust Models: Baseline and Probabilistic

Baseline Model:
    Simple hard threshold: Inside geofence → T_geo = 1, else T_geo = 0

Probabilistic Model:
    Learns to output continuous trust score T_geo ∈ [0, 1] using:
    - Logistic regression (fast, interpretable)
    - Small MLP (more expressive)

Features used:
    - lat, lon: Raw GPS coordinates
    - distance_to_home: Haversine distance to home
    - distance_to_geofence: Distance to nearest safe zone
    - time_sin, time_cos: Cyclical encoding of hour (0-23)
    - gps_accuracy: Estimated GPS error
    - speed: Current movement speed
    - is_boundary: Binary flag for boundary region proximity

Design Choices:
    - Probabilistic model uses logistic regression by default (interpretable)
    - Output is sigmoid-activated confidence score
    - Training uses binary cross-entropy loss
    - Evaluation uses standard classification metrics (FAR, FRR, EER)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaselineGeoModel(nn.Module):
    """
    Baseline geofence model: Hard threshold decision.
    
    Decision rule:
        T_geo = 1.0 if distance_to_geofence < threshold else 0.0
    
    Args:
        threshold: Distance threshold in meters (default: 50)
    """
    
    def __init__(self, threshold: float = 50.0):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute hard geofence decision.
        
        Args:
            features: (batch, 9) feature vector
                     Index 3 is distance_to_geofence
                     
        Returns:
            Trust scores (batch, 1) with values 0 or 1
        """
        distance_to_geofence = features[:, 3]
        trust = (distance_to_geofence < self.threshold).float()
        return trust.unsqueeze(1)
    
    def get_prediction_score(self, features: torch.Tensor) -> torch.Tensor:
        """Alias for forward for consistency with ProbabilisticGeoModel."""
        return self.forward(features)


class ProbabilisticGeoModel(nn.Module):
    """
    Probabilistic geofence model: Outputs continuous trust scores.
    
    Architecture options:
        - 'logistic': Simple logistic regression (1 hidden layer, fast)
        - 'mlp': Multi-layer perceptron (3 layers, more expressive)
    
    Args:
        input_dim: Feature dimension (default: 9)
        model_type: 'logistic' or 'mlp'
        hidden_dim: Hidden layer dimension for MLP (default: 64)
    """
    
    def __init__(
        self,
        input_dim: int = 9,
        model_type: str = 'logistic',
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.model_type = model_type
        
        if model_type == 'logistic':
            # Simple logistic regression
            self.fc = nn.Linear(input_dim, 1)
        elif model_type == 'mlp':
            # 3-layer MLP
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc3 = nn.Linear(hidden_dim // 2, 1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute probabilistic trust score.
        
        Args:
            features: (batch, input_dim) normalized feature vector
            
        Returns:
            Trust scores (batch, 1) in [0, 1] (sigmoid activated)
        """
        if self.model_type == 'logistic':
            logits = self.fc(features)
        else:  # mlp
            x = self.relu(self.fc1(features))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            logits = self.fc3(x)
        
        # Sigmoid activation for probability
        trust = torch.sigmoid(logits)
        return trust
    
    def get_prediction_score(self, features: torch.Tensor) -> torch.Tensor:
        """Alias for forward for consistency."""
        return self.forward(features)


class GeoModelWrapper:
    """
    Wrapper for training and evaluation of geofence models.
    
    Handles:
    - Model creation (baseline or probabilistic)
    - Training with Adam optimizer
    - Evaluation on test set
    - Threshold optimization for FAR/FRR
    """
    
    def __init__(
        self,
        model_type: Optional[str] = None,
        type: Optional[str] = None,
        device: str = 'cuda',
        **model_kwargs
    ):
        """
        Args:
            model_type / type: Accepts aliases for model selection.
                - Baseline aliases: 'baseline', 'hard', 'hard_threshold', 'rule', 'hard-threshold'
                - Probabilistic aliases: 'probabilistic', 'prob', 'logistic', 'mlp'
            device: 'cuda' or 'cpu'
            **model_kwargs: Arguments passed to model constructor
        """
        self.device = device

        # Backwards-compatible selection: prefer explicit model_type, fallback to type
        chosen = (model_type or type or 'baseline')
        chosen_key = str(chosen).lower()

        # Normalize aliases
        baseline_aliases = {'baseline', 'hard', 'hard_threshold', 'hard-threshold', 'rule', 'rule-based'}
        probabilistic_aliases = {'probabilistic', 'prob', 'logistic', 'mlp'}

        if chosen_key in baseline_aliases:
            self.model_type = 'baseline'
            self.model = BaselineGeoModel(**model_kwargs)
        elif chosen_key in probabilistic_aliases:
            self.model_type = 'probabilistic'
            # If user specified a specific probabilistic architecture (logistic/mlp), pass it through
            if chosen_key in {'logistic', 'mlp'} and 'model_type' not in model_kwargs:
                model_kwargs['model_type'] = chosen_key
            self.model = ProbabilisticGeoModel(**model_kwargs)
        else:
            raise ValueError(f"Unknown geo model type: {chosen}")

        self.model.to(device)
        logger.info(f"Initialized {self.model_type} geo model (alias: {chosen})")
    
    def fit(
        self,
        train_dataloader,
        val_dataloader,
        num_epochs: int = 50,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
    ) -> Dict[str, Any]:
        """
        Train the probabilistic model (no-op for baseline).
        
        Args:
            train_dataloader: Training DataLoader
            val_dataloader: Validation DataLoader
            num_epochs: Number of training epochs
            learning_rate: Adam learning rate
            weight_decay: L2 regularization
            
        Returns:
            Dictionary with training metrics
        """
        if self.model_type == 'baseline':
            logger.info("Baseline model is non-trainable, skipping fit")
            return {'epochs': 0}
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        criterion = nn.BCELoss()
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            num_batches = 0
            
            for batch in train_dataloader:
                features = batch['features'].to(self.device)
                labels = batch['label'].float().unsqueeze(1).to(self.device)
                
                optimizer.zero_grad()
                
                predictions = self.model(features)
                loss = criterion(predictions, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            train_loss = train_loss / num_batches
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_accuracy = 0.0
            num_batches = 0
            
            with torch.no_grad():
                for batch in val_dataloader:
                    features = batch['features'].to(self.device)
                    labels = batch['label'].float().unsqueeze(1).to(self.device)
                    
                    predictions = self.model(features)
                    loss = criterion(predictions, labels)
                    
                    val_loss += loss.item()
                    
                    # Accuracy with threshold 0.5
                    preds_binary = (predictions >= 0.5).float()
                    accuracy = (preds_binary == labels).float().mean()
                    val_accuracy += accuracy.item()
                    
                    num_batches += 1
            
            val_loss = val_loss / num_batches
            val_accuracy = val_accuracy / num_batches
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs}: "
                    f"train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, "
                    f"val_acc={val_accuracy:.4f}"
                )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        return history
    
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict trust scores.
        
        Args:
            features: (batch, input_dim)
            
        Returns:
            Trust scores (batch, 1) in [0, 1]
        """
        self.model.eval()
        with torch.no_grad():
            features = features.to(self.device)
            predictions = self.model.get_prediction_score(features)
        return predictions.cpu()
    
    def save(self, save_path: str):
        """Save model weights."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Saved geo model to {save_path}")
    
    def load(self, load_path: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        logger.info(f"Loaded geo model from {load_path}")
