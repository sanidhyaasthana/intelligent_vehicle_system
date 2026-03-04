"""
Fusion Model and Risk Engine

Combines face trust T_face and geofence trust T_geo into a final authorization
decision using two approaches:

1. Rule-Based Fusion:
   Risk R = alpha * (1 - T_face) + beta * (1 - T_geo)
   Decision based on thresholds tau1, tau2:
   - ALLOW if R < tau1 (low risk)
   - WARN if tau1 <= R < tau2 (medium risk)
   - BLOCK if R >= tau2 (high risk)

2. Learned Fusion:
   Train a small MLP or logistic regression on [T_face, T_geo, context]
   to predict P(authorized), then apply decision thresholds

Design Choices:
    - Default: Rule-based (interpretable, no training required)
    - Optional learned fusion for maximum flexibility
    - Extensive logging of T_face, T_geo, R distributions
    - Threshold optimization via grid search on validation set
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RiskFusionModel(nn.Module):
    """
    Risk fusion engine combining face and geofence trust.
    
    Args:
        mode: 'rule_based' or 'learned'
        alpha: Weight for face risk (1 - T_face) in rule-based mode
        beta: Weight for geo risk (1 - T_geo) in rule-based mode
        tau1: Threshold between ALLOW and WARN
        tau2: Threshold between WARN and BLOCK
        hidden_dim: Hidden dimension for learned fusion MLP
    """
    
    def __init__(
        self,
        mode: str = 'rule_based',
        alpha: float = 0.6,
        beta: float = 0.4,
        tau1: float = 0.3,
        tau2: float = 0.6,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.tau1 = tau1
        self.tau2 = tau2
        
        # Learned fusion components (only used if mode == 'learned')
        if mode == 'learned':
            # Input: [T_face, T_geo, ...optional context...]
            input_dim = 2  # T_face, T_geo
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc3 = nn.Linear(hidden_dim // 2, 1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
        
        # Statistics tracking
        self.register_buffer('num_calls', torch.tensor(0))
    
    def forward(
        self,
        T_face: torch.Tensor,
        T_geo: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute authorization decision.
        
        Args:
            T_face: Face trust scores (batch, 1) in [0, 1]
            T_geo: Geofence trust scores (batch, 1) in [0, 1]
            context: Optional additional context features (batch, dim)
            return_components: If True, return T_face, T_geo, R, T_fused
            
        Returns:
            Dictionary with keys:
            - 'decision': Final decision (ALLOW=0, WARN=1, BLOCK=2)
            - 'risk': Risk score R
            - 'decision_scores': Confidence scores for each decision
            - 'T_face': Face trust (if return_components)
            - 'T_geo': Geo trust (if return_components)
            - 'T_fused': Fused trust (if return_components)
        """
        # Ensure shapes
        if T_face.dim() == 1:
            T_face = T_face.unsqueeze(1)
        if T_geo.dim() == 1:
            T_geo = T_geo.unsqueeze(1)
        
        batch_size = T_face.shape[0]
        
        if self.mode == 'rule_based':
            # Compute risk: R = alpha * (1 - T_face) + beta * (1 - T_geo)
            risk = self.alpha * (1.0 - T_face) + self.beta * (1.0 - T_geo)
            T_fused = risk.squeeze(1)  # For consistency
        else:  # learned
            # Concatenate features
            features = torch.cat([T_face, T_geo], dim=1)
            if context is not None:
                features = torch.cat([features, context], dim=1)
            
            # Forward through MLP
            x = self.relu(self.fc1(features))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            T_fused = torch.sigmoid(self.fc3(x)).squeeze(1)
            
            # Convert to risk (inverse of trust)
            risk = (1.0 - T_fused).unsqueeze(1)
        
        # Make decision based on risk
        risk = risk.squeeze(1)
        decision = torch.zeros(batch_size, dtype=torch.long, device=T_face.device)
        
        # ALLOW if risk < tau1
        decision[(risk < self.tau1)] = 0
        # WARN if tau1 <= risk < tau2
        decision[(risk >= self.tau1) & (risk < self.tau2)] = 1
        # BLOCK if risk >= tau2
        decision[(risk >= self.tau2)] = 2
        
        # Create decision scores (soft scores for each class)
        decision_scores = self._compute_decision_scores(risk)
        
        result = {
            'decision': decision,
            'risk': risk,
            'decision_scores': decision_scores,
        }
        
        if return_components:
            result['T_face'] = T_face.squeeze(1)
            result['T_geo'] = T_geo.squeeze(1)
            result['T_fused'] = T_fused
        
        self.num_calls += 1
        
        return result
    
    def _compute_decision_scores(self, risk: torch.Tensor) -> torch.Tensor:
        """
        Compute soft scores for each decision class.
        
        Uses Gaussian-like functions centered at each threshold.
        """
        batch_size = risk.shape[0]
        num_classes = 3  # ALLOW, WARN, BLOCK
        
        scores = torch.zeros(batch_size, num_classes, device=risk.device)
        
        # Soft scoring function (higher score = more confidence in that decision)
        sigma = (self.tau2 - self.tau1) / 3.0  # Adjust spread
        
        # ALLOW score: high when risk is low
        allow_score = torch.exp(-((risk - self.tau1) ** 2) / (2 * sigma ** 2))
        
        # BLOCK score: high when risk is high
        block_score = torch.exp(-((risk - self.tau2) ** 2) / (2 * sigma ** 2))
        
        # WARN score: complement
        warn_score = 1.0 - allow_score - block_score
        warn_score = torch.clamp(warn_score, min=0.0)
        
        # Normalize
        total = allow_score + warn_score + block_score
        scores[:, 0] = allow_score / (total + 1e-8)
        scores[:, 1] = warn_score / (total + 1e-8)
        scores[:, 2] = block_score / (total + 1e-8)
        
        return scores
    
    def get_risk_stats(self) -> Dict[str, Any]:
        """Get statistics about fusion decisions."""
        return {
            'num_calls': self.num_calls.item(),
            'mode': self.mode,
            'params': {
                'alpha': self.alpha,
                'beta': self.beta,
                'tau1': self.tau1,
                'tau2': self.tau2,
            }
        }
    
    def optimize_thresholds(
        self,
        T_face_val: np.ndarray,
        T_geo_val: np.ndarray,
        labels_val: np.ndarray,
        metric: str = 'eer',
    ) -> Tuple[float, float]:
        """
        Optimize tau1 and tau2 on validation set via grid search.

        Args:
            T_face_val: Face trust scores (N,)
            T_geo_val: Geo trust scores (N,)
            labels_val: Ground truth labels: 1=genuine, 0=impostor
            metric: Optimization metric: 'eer', 'balanced_acc', or 'far_at_frr'

        Returns:
            Optimized (tau1, tau2)

        Label convention (enforced):
            1 = genuine (legitimate user)
            0 = impostor (attack)

        Metric definitions:
            FAR = impostors accepted / total impostors
            FRR = genuine rejected / total genuine
        """
        best_tau1, best_tau2 = self.tau1, self.tau2
        best_score = -1.0

        # Compute risk for validation set
        T_face_val_t = torch.from_numpy(T_face_val).float()
        T_geo_val_t = torch.from_numpy(T_geo_val).float()

        risk_val = self.alpha * (1.0 - T_face_val_t) + self.beta * (1.0 - T_geo_val_t)
        risk_val = risk_val.numpy()

        # Grid search over threshold combinations
        tau1_range = np.linspace(0.1, 0.5, 10)
        tau2_range = np.linspace(0.4, 0.8, 10)

        for tau1 in tau1_range:
            for tau2 in tau2_range:
                if tau1 >= tau2:
                    continue

                # Compute decisions: risk < tau1 -> ALLOW, risk >= tau2 -> BLOCK
                decisions = np.zeros_like(risk_val)
                decisions[risk_val < tau1] = 0  # ALLOW
                decisions[(risk_val >= tau1) & (risk_val < tau2)] = 1  # WARN
                decisions[risk_val >= tau2] = 2  # BLOCK

                # Binary: accepted (ALLOW=0) vs rejected (BLOCK=2)
                # accepted=1 means the system accepts the user
                accepted = (decisions < tau2).astype(int)  # ALLOW or WARN -> accepted

                # Label convention: 1=genuine, 0=impostor
                # TP = genuine correctly accepted
                # FP = impostor incorrectly accepted (FAR numerator)
                # FN = genuine incorrectly rejected (FRR numerator)
                # TN = impostor correctly rejected
                if metric == 'eer':
                    tp = np.sum((accepted == 1) & (labels_val == 1))
                    fp = np.sum((accepted == 1) & (labels_val == 0))
                    fn = np.sum((accepted == 0) & (labels_val == 1))
                    tn = np.sum((accepted == 0) & (labels_val == 0))

                    far = fp / (fp + tn + 1e-8)   # impostors accepted / total impostors
                    frr = fn / (fn + tp + 1e-8)    # genuine rejected / total genuine
                    eer = (far + frr) / 2.0
                    score = -eer  # Negative because we're maximizing
                elif metric == 'balanced_acc':
                    tp = np.sum((accepted == 1) & (labels_val == 1))
                    tn = np.sum((accepted == 0) & (labels_val == 0))
                    fp = np.sum((accepted == 1) & (labels_val == 0))
                    fn = np.sum((accepted == 0) & (labels_val == 1))

                    tpr = tp / (tp + fn + 1e-8)
                    tnr = tn / (tn + fp + 1e-8)
                    score = (tpr + tnr) / 2.0
                else:
                    score = 0.0

                if score > best_score:
                    best_score = score
                    best_tau1 = tau1
                    best_tau2 = tau2
        
        self.tau1 = best_tau1
        self.tau2 = best_tau2
        
        logger.info(
            f"Optimized thresholds on validation set: "
            f"tau1={best_tau1:.4f}, tau2={best_tau2:.4f}, score={best_score:.4f}"
        )
        
        return best_tau1, best_tau2
    
    def save(self, save_path: str):
        """Save model (for learned mode)."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'mode': self.mode,
            'alpha': self.alpha,
            'beta': self.beta,
            'tau1': self.tau1,
            'tau2': self.tau2,
        }
        
        if self.mode == 'learned':
            state['model_state'] = self.state_dict()
        
        torch.save(state, save_path)
        logger.info(f"Saved fusion model to {save_path}")
    
    def load(self, load_path: str, device: str = 'cuda'):
        """Load model."""
        state = torch.load(load_path, map_location=device)
        
        self.mode = state['mode']
        self.alpha = state['alpha']
        self.beta = state['beta']
        self.tau1 = state['tau1']
        self.tau2 = state['tau2']
        
        if self.mode == 'learned' and 'model_state' in state:
            self.load_state_dict(state['model_state'])
        
        logger.info(f"Loaded fusion model from {load_path}")
