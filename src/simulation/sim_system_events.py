"""
System-level event simulation for fusion model training.

Creates realistic system-level authentication events combining:
- Face verification results (T_face, quality scores, embeddings)
- Geofence verification results (T_geo, location features)
- Attack scenarios:
  * Correct face + wrong location
  * Wrong face + correct location
  * Both wrong
  * Edge cases (low quality face, boundary location)

Each event includes all necessary information for fusion model training
and evaluation.

LABEL CONVENTION (consistent across the entire pipeline):
    1 = genuine (legitimate user)
    0 = impostor (attack)

SCORE CONVENTION:
    T_face, T_geo are trust scores: higher = more genuine.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SystemEventGenerator:
    """
    Generate system-level authentication events.
    
    Combines face verification and geofence verification into unified events
    with ground truth labels.
    
    Args:
        num_legitimate: Number of legitimate events
        num_attack: Number of attack events
        seed: Random seed
    """
    
    def __init__(
        self,
        num_legitimate: int = 1000,
        num_attack: int = 500,
        seed: int = 42,
    ):
        self.num_legitimate = num_legitimate
        self.num_attack = num_attack
        self.seed = seed
        
        np.random.seed(seed)
    
    def _generate_legitimate_event(self) -> Dict:
        """
        Generate a legitimate authentication event.
        Both face and location match the authorized user.
        """
        # High face match (T_face close to 1)
        T_face = np.random.beta(8, 2)  # Mostly high scores
        
        # High geofence match (T_geo close to 1)
        T_geo = np.random.beta(8, 2)
        
        # Good image quality
        quality_score = np.random.beta(6, 2)
        
        # Random identity (for feature purposes)
        identity = np.random.randint(0, 100)
        
        # GPS inside geofence
        is_inside_geofence = 1.0
        
        return {
            'T_face': float(T_face),
            'T_geo': float(T_geo),
            'quality_score': float(quality_score),
            'identity': int(identity),
            'is_inside_geofence': float(is_inside_geofence),
            'time_of_day': float(np.random.randint(6, 23)),
            'label': 1,  # Genuine (legitimate user)
        }
    
    def _generate_attack_event(self) -> Dict:
        """
        Generate an attack event.
        One or both of face/geo matching fails.
        """
        attack_type = np.random.choice([
            'wrong_face',
            'wrong_location',
            'both_wrong',
            'low_quality',
            'boundary',
        ])
        
        if attack_type == 'wrong_face':
            # Attacker spoofed face but location is correct
            T_face = np.random.beta(2, 5)  # Low match score
            T_geo = np.random.beta(7, 3)  # Good location match
            quality_score = np.random.beta(5, 3)
            is_inside_geofence = 1.0
        
        elif attack_type == 'wrong_location':
            # Face is correct but location is spoofed
            T_face = np.random.beta(7, 2)  # Good match
            T_geo = np.random.beta(2, 6)  # Poor location match
            quality_score = np.random.beta(6, 2)
            is_inside_geofence = 0.0
        
        elif attack_type == 'both_wrong':
            # Both face and location are spoofed
            T_face = np.random.beta(2, 6)
            T_geo = np.random.beta(2, 6)
            quality_score = np.random.beta(3, 4)
            is_inside_geofence = 0.0
        
        elif attack_type == 'low_quality':
            # Correct user but very poor image quality
            T_face = np.random.beta(3, 3)  # Uncertain due to quality
            T_geo = np.random.beta(7, 2)  # Location is fine
            quality_score = np.random.beta(2, 6)  # Very low quality
            is_inside_geofence = 1.0
        
        else:  # boundary
            # Near geofence boundary (ambiguous)
            T_face = np.random.beta(7, 2)
            T_geo = np.random.beta(3, 4)  # Boundary uncertainty
            quality_score = np.random.beta(4, 3)
            is_inside_geofence = 0.3  # Partially inside
        
        identity = np.random.randint(0, 100)
        time_of_day = np.random.choice([np.random.randint(1, 6), np.random.randint(6, 23)])
        
        return {
            'T_face': float(T_face),
            'T_geo': float(T_geo),
            'quality_score': float(quality_score),
            'identity': int(identity),
            'is_inside_geofence': float(is_inside_geofence),
            'time_of_day': float(time_of_day),
            'label': 0,  # Impostor (attack)
        }
    
    def generate(self) -> pd.DataFrame:
        """Generate complete system event dataset."""
        events = []
        
        # Legitimate events
        for _ in range(self.num_legitimate):
            events.append(self._generate_legitimate_event())
        
        # Attack events
        for _ in range(self.num_attack):
            events.append(self._generate_attack_event())
        
        df = pd.DataFrame(events)
        
        logger.info(
            f"Generated {len(df)} system events: "
            f"{self.num_legitimate} legitimate, {self.num_attack} attacks"
        )
        
        return df
    
    def save(self, output_path: str) -> None:
        """Generate and save to CSV."""
        df = self.generate()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved system events to {output_path}")


class SystemEventDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for system events.
    
    Args:
        csv_path: Path to CSV with system events
        split: 'train', 'val', or 'test'
        split_ratio: (train, val, test) ratios
    """
    
    def __init__(
        self,
        csv_path: str,
        split: str = 'train',
        split_ratio: Tuple[float, float, float] = (0.6, 0.2, 0.2),
        seed: int = 42,
    ):
        self.data = pd.read_csv(csv_path)
        
        np.random.seed(seed)
        n = len(self.data)
        indices = np.random.permutation(n)
        
        train_size = int(n * split_ratio[0])
        val_size = int(n * split_ratio[1])
        
        if split == 'train':
            self.indices = indices[:train_size]
        elif split == 'val':
            self.indices = indices[train_size:train_size + val_size]
        else:  # test
            self.indices = indices[train_size + val_size:]
        
        logger.info(f"SystemEventDataset split '{split}': {len(self.indices)} samples")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict:
        sample_idx = self.indices[idx]
        row = self.data.iloc[sample_idx]
        
        features = torch.tensor([
            float(row['T_face']),
            float(row['T_geo']),
            float(row['quality_score']),
            float(row['is_inside_geofence']),
            float(row['time_of_day']) / 24.0,  # Normalize
        ], dtype=torch.float32)
        
        label = int(row['label'])
        
        return {
            'features': features,
            'label': label,
        }


def generate_system_events(
    output_path: str,
    num_legitimate: int = 1000,
    num_attack: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Convenience function to generate system events.
    
    Args:
        output_path: Where to save
        num_legitimate: Number of legitimate events
        num_attack: Number of attack events
        seed: Random seed
        
    Returns:
        Generated DataFrame
    """
    generator = SystemEventGenerator(
        num_legitimate=num_legitimate,
        num_attack=num_attack,
        seed=seed,
    )
    generator.save(output_path)
    return generator.generate()
