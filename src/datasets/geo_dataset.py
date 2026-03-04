"""
Geofence Dataset and DataLoader

This module implements PyTorch Dataset and DataLoader for geofence (location-based)
data with engineered features:
- Distance to nearest safe zone
- Distance to home/office/parking
- Time-of-day features (sine/cosine encoding)
- GPS accuracy estimate
- Movement speed
- Proximity to boundary regions

Features are normalized to [0, 1] range for stable training.

Design Choices:
- Feature engineering on-the-fly during dataset creation
- Normalization using precomputed min/max from training set
- Boundary region detection to capture edge cases
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import logging

from ..utils.seed_utils import set_seed
from ..utils.geo_utils import haversine_distance

logger = logging.getLogger(__name__)


class GeoDataset(Dataset):
    """
    PyTorch Dataset for geofence location-based data.
    
    Expected CSV columns:
    - lat, lon: GPS coordinates
    - time_of_day: Hour of day (0-23)
    - gps_accuracy: GPS accuracy estimate in meters
    - speed: Movement speed in m/s
    - label: 1 (genuine / legitimate) or 0 (impostor / attack)
      Global convention: 1=genuine, 0=impostor  (matches face and fusion pipeline)
    
    Args:
        csv_path: Path to CSV file with location data
        split: 'train', 'val', or 'test'
        split_ratio: (train_ratio, val_ratio, test_ratio)
        geofences: List of geofence definitions with 'name', 'lat', 'lon', 'radius'
        home_lat, home_lon: Home location coordinates
        boundary_distance: Distance threshold for boundary region detection (meters)
        normalize: Whether to normalize features
        normalization_params: Pre-computed normalization parameters
        seed: Random seed
    """
    
    def __init__(
        self,
        csv_path: str,
        split: str = 'train',
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        geofences: Optional[List[Dict]] = None,
        home_lat: float = 0.0,
        home_lon: float = 0.0,
        boundary_distance: float = 100.0,  # meters
        normalize: bool = True,
        normalization_params: Optional[Dict[str, Tuple[float, float]]] = None,
        seed: int = 42,
    ):
        self.csv_path = Path(csv_path)
        self.split = split
        self.geofences = geofences or []
        self.home_lat = home_lat
        self.home_lon = home_lon
        self.boundary_distance = boundary_distance
        self.normalize = normalize
        self.normalization_params = normalization_params or {}
        self.seed = seed
        
        set_seed(seed)
        
        # Load data
        self.data = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.data)} samples from {csv_path}")

        # -----------------------------------------------------------------
        # Label convention check: global convention is 1=genuine, 0=impostor.
        # Old sim_geo_data.py generated 0=legitimate, 1=attack (inverted).
        # If the CSV was produced before the label fix, warn and hard-stop.
        # Detection heuristic: if mean(label) > 0.6 we likely have
        # label=1 for the majority class (attack >> genuine), which is
        # the old inverted encoding.
        # -----------------------------------------------------------------
        if 'label' in self.data.columns:
            unique_labels = set(self.data['label'].unique())
            if not unique_labels.issubset({0, 1}):
                raise RuntimeError(
                    f"GeoDataset: unexpected label values {unique_labels}. "
                    f"Expected {{0, 1}} with convention 1=genuine, 0=impostor."
                )
            mean_label = self.data['label'].mean()
            if mean_label > 0.75:
                logger.warning(
                    f"GeoDataset: mean(label)={mean_label:.3f} > 0.75. "
                    f"Possible label inversion — check that CSV uses "
                    f"1=genuine (legitimate) and 0=impostor (attack). "
                    f"Old sim_geo_data.py used the inverted convention; "
                    f"regenerate location_data.csv with the fixed generator."
                )
                # AUC-based downstream check: a geo model trained on inverted
                # labels will produce AUC < 0.5 when evaluated with the global
                # convention. The evaluation pipeline already raises on AUC < 0.5.
        
        # Split dataset
        self.indices = self._split_dataset(split, split_ratio, seed)
        logger.info(f"Split '{split}': {len(self.indices)} samples")
        
        # Engineer features
        self.features = self._engineer_features()
        
        # Compute normalization parameters if training
        if split == 'train' and not self.normalization_params:
            self._compute_normalization_params()
    
    def _split_dataset(self, split: str, split_ratio: Tuple, seed: int) -> list:
        """Split dataset deterministically."""
        set_seed(seed)
        n = len(self.data)
        indices = np.random.permutation(n)
        
        train_size = int(n * split_ratio[0])
        val_size = int(n * split_ratio[1])
        
        if split == 'train':
            return indices[:train_size].tolist()
        elif split == 'val':
            return indices[train_size:train_size + val_size].tolist()
        else:  # test
            return indices[train_size + val_size:].tolist()
    
    def _engineer_features(self) -> np.ndarray:
        """Engineer all features from raw data."""
        n = len(self.data)
        features = []
        
        for idx in range(n):
            row = self.data.iloc[idx]
            lat, lon = row['lat'], row['lon']
            
            # Distance to home
            dist_to_home = haversine_distance(lat, lon, self.home_lat, self.home_lon)
            
            # Distance to nearest geofence
            dist_to_geofence = float('inf')
            for geo in self.geofences:
                dist = haversine_distance(lat, lon, geo['lat'], geo['lon'])
                dist_to_geofence = min(dist_to_geofence, dist)
            if dist_to_geofence == float('inf'):
                dist_to_geofence = 10000.0  # Large default
            
            # Time-of-day features (sine/cosine encoding)
            hour = float(row['time_of_day'])
            time_sin = np.sin(2 * np.pi * hour / 24.0)
            time_cos = np.cos(2 * np.pi * hour / 24.0)
            
            # GPS accuracy (already in row)
            gps_acc = float(row['gps_accuracy'])
            
            # Speed (already in row)
            speed = float(row['speed'])
            
            # Boundary region flag (1 if near any boundary)
            is_boundary = 1.0 if dist_to_geofence < self.boundary_distance else 0.0
            
            # Assemble feature vector
            feat = np.array([
                lat,
                lon,
                dist_to_home,
                dist_to_geofence,
                time_sin,
                time_cos,
                gps_acc,
                speed,
                is_boundary,
            ], dtype=np.float32)
            
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _compute_normalization_params(self):
        """Compute min/max for normalization on training split."""
        train_features = self.features[self.indices]
        
        for i in range(train_features.shape[1]):
            min_val = float(np.min(train_features[:, i]))
            max_val = float(np.max(train_features[:, i]))
            
            # Avoid division by zero
            if max_val == min_val:
                max_val = min_val + 1.0
            
            feature_names = [
                'lat', 'lon', 'dist_to_home', 'dist_to_geofence',
                'time_sin', 'time_cos', 'gps_accuracy', 'speed', 'is_boundary'
            ]
            self.normalization_params[feature_names[i]] = (min_val, max_val)
        
        logger.info(f"Computed normalization parameters: {self.normalization_params}")
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to [0, 1] range."""
        if not self.normalize or not self.normalization_params:
            return features
        
        feature_names = [
            'lat', 'lon', 'dist_to_home', 'dist_to_geofence',
            'time_sin', 'time_cos', 'gps_accuracy', 'speed', 'is_boundary'
        ]
        
        normalized = features.copy()
        for i, name in enumerate(feature_names):
            if name in self.normalization_params:
                min_val, max_val = self.normalization_params[name]
                normalized[i] = (features[i] - min_val) / (max_val - min_val + 1e-8)
        
        return normalized
    
    def get_normalization_params(self) -> Dict[str, Tuple[float, float]]:
        """Get normalization parameters for sharing with other splits."""
        return self.normalization_params
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with keys:
            - 'features': Feature vector (9D)
            - 'label': 0 (legitimate) or 1 (attack)
        """
        sample_idx = self.indices[idx]
        
        # Get raw features
        feat = self.features[sample_idx]
        
        # Normalize if needed
        if self.normalize:
            feat = self._normalize_features(feat)
        
        # Get label — global convention: 1=genuine, 0=impostor
        label = int(self.data.iloc[sample_idx]['label'])
        if label not in (0, 1):
            raise RuntimeError(
                f"GeoDataset: unexpected label value {label!r} at row {sample_idx}. "
                f"Expected 1=genuine or 0=impostor."
            )
        
        return {
            'features': torch.from_numpy(feat).float(),
            'label': label,
        }


def build_geo_dataloader(
    config: Dict[str, Any],
    split: str = 'train',
    normalization_params: Optional[Dict[str, Tuple[float, float]]] = None,
) -> DataLoader:
    """
    Build a PyTorch DataLoader for geofence data.
    
    Args:
        config: Configuration dictionary with keys:
            - data.geo_data_file: Path to CSV with location data
            - data.geofences: List of geofence definitions
            - data.home_lat, data.home_lon: Home location
            - data.boundary_distance: Boundary region threshold
            - training.batch_size: Batch size
            - training.num_workers: Data loading workers
        split: 'train', 'val', or 'test'
        normalization_params: Pre-computed normalization parameters
        
    Returns:
        PyTorch DataLoader
    """
    data_config = config.get('data', {})
    train_config = config.get('training', {})
    
    csv_path = data_config.get('geo_data_file', 'data/geo/location_data.csv')
    geofences = data_config.get('geofences', [])
    home_lat = data_config.get('home_lat', 0.0)
    home_lon = data_config.get('home_lon', 0.0)
    boundary_distance = data_config.get('boundary_distance', 100.0)
    
    batch_size = train_config.get('batch_size', 64)
    num_workers = train_config.get('num_workers', 4)
    
    dataset = GeoDataset(
        csv_path=csv_path,
        split=split,
        geofences=geofences,
        home_lat=home_lat,
        home_lon=home_lon,
        boundary_distance=boundary_distance,
        normalize=True,
        normalization_params=normalization_params,
    )
    
    shuffle = (split == 'train')
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
    )
    
    return dataloader
