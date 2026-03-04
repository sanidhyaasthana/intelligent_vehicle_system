"""
Face Dataset and DataLoader

This module implements PyTorch Dataset and DataLoader for face images with:
- Identity label loading from CSV/text files
- Standard preprocessing (resize to 112x112, normalization)
- Optional synthetic degradations (blur, brightness, occlusion)
- Per-sample quality score computation

Quality scores measure image degradation:
- Blur: Laplacian variance (sharp=high, blurry=low)
- Brightness: Mean intensity
- Contrast: Standard deviation of intensity
- Normalized to [0, 1] where 1=perfect quality

Design Choices:
- Store quality scores in memory for fast access during training
- Apply degradations stochastically during training, deterministically during eval
- Normalize embeddings post-extraction for stability
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import random
import logging

from ..utils.quality_metrics import compute_quality_score
from ..utils.seed_utils import set_seed


logger = logging.getLogger(__name__)


class FaceDataset(Dataset):
    """
    PyTorch Dataset for face images with identity labels and quality scores.
    
    Args:
        image_dir: Directory containing face images
        label_file: CSV/text file with columns [image_path, identity_id]
        image_size: Target image size (will resize to image_size x image_size)
        split: 'train', 'val', or 'test'
        split_ratio: (train_ratio, val_ratio, test_ratio)
        add_degradation: Whether to apply synthetic degradations during training
        degradation_types: List of ['blur', 'brightness', 'occlusion'] to apply
        normalize: Whether to normalize images to [-1, 1]
        seed: Random seed for reproducibility
        precompute_quality: If True, compute and cache quality scores
    """
    
    def __init__(
        self,
        image_dir: str,
        label_file: str,
        image_size: int = 112,
        split: str = 'train',
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        add_degradation: bool = False,
        degradation_types: list = None,
        normalize: bool = True,
        seed: int = 42,
        precompute_quality: bool = True,
    ):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.split = split
        self.add_degradation = add_degradation and (split == 'train')
        self.degradation_types = degradation_types or ['blur', 'brightness']
        self.normalize = normalize
        self.seed = seed
        
        set_seed(seed)
        
        # Load labels
        self.samples = self._load_labels(label_file)
        logger.info(f"Loaded {len(self.samples)} total samples from {label_file}")
        
        # Split dataset
        self.indices = self._split_dataset(split, split_ratio, seed)
        logger.info(f"Split '{split}': {len(self.indices)} samples")
        
        # Precompute quality scores
        self.quality_scores = {}
        if precompute_quality:
            self._precompute_quality_scores()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        if normalize:
            self.transform = transforms.Compose([
                self.transform,
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                   std=[0.5, 0.5, 0.5])
            ])
    
    def _load_labels(self, label_file: str) -> list:
        """Load image paths and identity labels."""
        samples = []
        label_file = Path(label_file)
        
        if label_file.suffix == '.csv':
            # CSV format
            df = pd.read_csv(label_file)
            for _, row in df.iterrows():
                image_path = self.image_dir / row['image_path']
                if image_path.exists():
                    samples.append({
                        'path': str(image_path),
                        'identity': int(row['identity_id']),
                    })
        else:
            # Text file - may be CSV or space-separated
            # Try to auto-detect format
            try:
                # Try CSV format first (even if suffix is .txt)
                df = pd.read_csv(label_file)
                for _, row in df.iterrows():
                    image_path = self.image_dir / row['image_path']
                    if image_path.exists():
                        samples.append({
                            'path': str(image_path),
                            'identity': int(row['identity_id']),
                        })
            except:
                # Fall back to space-separated format
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            image_path = self.image_dir / parts[0]
                            if image_path.exists():
                                samples.append({
                                    'path': str(image_path),
                                    'identity': int(parts[1]),
                                })
        
        return samples
    
    def _split_dataset(self, split: str, split_ratio: Tuple, seed: int) -> list:
        """
        Subject-disjoint split: all samples of an identity go to exactly one partition.

        Procedure:
        1. Collect unique identity IDs from self.samples.
        2. Shuffle identity list with fixed seed.
        3. Assign identity groups to train/val/test by ratio.
        4. Map back to sample indices.
        5. Assert zero identity overlap across partitions.

        Raises RuntimeError if any identity ID appears in more than one partition.
        """
        set_seed(seed)

        # Collect all unique identity IDs and group sample indices by identity
        identity_to_indices: dict = {}
        for idx, s in enumerate(self.samples):
            uid = s['identity']
            identity_to_indices.setdefault(uid, []).append(idx)

        unique_ids = list(identity_to_indices.keys())
        random.shuffle(unique_ids)

        n_ids = len(unique_ids)
        train_end = int(n_ids * split_ratio[0])
        val_end = train_end + int(n_ids * split_ratio[1])

        train_ids = set(unique_ids[:train_end])
        val_ids   = set(unique_ids[train_end:val_end])
        test_ids  = set(unique_ids[val_end:])

        # --- Identity-overlap assertion ---
        train_test_overlap = train_ids & test_ids
        train_val_overlap  = train_ids & val_ids
        val_test_overlap   = val_ids   & test_ids
        if train_test_overlap:
            raise RuntimeError(
                f"Identity leakage detected: {len(train_test_overlap)} identities "
                f"appear in both train and test partitions."
            )
        if train_val_overlap:
            raise RuntimeError(
                f"Identity leakage detected: {len(train_val_overlap)} identities "
                f"appear in both train and val partitions."
            )
        if val_test_overlap:
            raise RuntimeError(
                f"Identity leakage detected: {len(val_test_overlap)} identities "
                f"appear in both val and test partitions."
            )

        logger.info(
            f"Subject-disjoint split (seed={seed}): "
            f"train_ids={len(train_ids)}, val_ids={len(val_ids)}, test_ids={len(test_ids)}"
        )

        if split == 'train':
            selected_ids = train_ids
        elif split == 'val':
            selected_ids = val_ids
        else:
            selected_ids = test_ids

        indices = []
        for uid in selected_ids:
            indices.extend(identity_to_indices[uid])
        return indices
    
    def _precompute_quality_scores(self):
        """Precompute quality scores for all images."""
        logger.info("Precomputing quality scores...")
        for idx in self.indices:
            sample = self.samples[idx]
            image = cv2.imread(sample['path'])
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
                q = compute_quality_score(image_tensor)
                self.quality_scores[idx] = q
        logger.info(f"Computed quality scores for {len(self.quality_scores)} images")
    
    def _apply_degradation(self, image: np.ndarray) -> np.ndarray:
        """Apply synthetic degradation to simulate low-light, blur, occlusion."""
        for deg_type in self.degradation_types:
            if random.random() > 0.5:  # 50% chance
                if deg_type == 'blur':
                    kernel_size = random.choice([3, 5, 7])
                    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
                elif deg_type == 'brightness':
                    alpha = random.uniform(0.5, 1.5)
                    image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
                elif deg_type == 'occlusion':
                    # Random rectangular occlusion
                    h, w = image.shape[:2]
                    x1 = random.randint(0, w // 3)
                    y1 = random.randint(0, h // 3)
                    x2 = random.randint(2 * w // 3, w)
                    y2 = random.randint(2 * h // 3, h)
                    image[y1:y2, x1:x2] = 128
        return image
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with keys:
            - 'image': Preprocessed image tensor
            - 'identity': Identity label (integer)
            - 'quality': Quality score in [0, 1]
            - 'path': Original image path (for logging)
        """
        sample_idx = self.indices[idx]
        sample = self.samples[sample_idx]
        
        # Load image
        image = cv2.imread(sample['path'])
        if image is None:
            logger.warning(f"Failed to load image: {sample['path']}")
            # Return zeros on failure
            return {
                'image': torch.zeros(3, self.image_size, self.image_size),
                'identity': 0,
                'quality': 0.0,
                'path': sample['path'],
            }
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply degradation
        if self.add_degradation:
            image = self._apply_degradation(image)
        
        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Get quality score
        if sample_idx in self.quality_scores:
            quality = self.quality_scores[sample_idx]
        else:
            quality = compute_quality_score(image_tensor)
        
        # Normalize if needed
        if self.normalize:
            image_tensor = (image_tensor - 0.5) / 0.5
        
        return {
            'image': image_tensor,
            'identity': sample['identity'],
            'quality': quality,
            'path': sample['path'],
        }


def build_face_dataloader(
    config: Dict[str, Any],
    split: str = 'train',
) -> DataLoader:
    """
    Build a PyTorch DataLoader for face data.
    
    Args:
        config: Configuration dictionary with keys:
            - data.face_data_dir: Directory with face images
            - data.face_label_file: Path to label file
            - data.image_size: Image size (default 112)
            - data.train_split, data.val_split, data.test_split: Split ratios
            - data.add_quality_degradation: Enable synthetic degradation
            - data.degradation_types: List of degradation types
            - training.batch_size: Batch size
            - training.num_workers: Number of data loading workers
        split: 'train', 'val', or 'test'
        
    Returns:
        PyTorch DataLoader
    """
    data_config = config.get('data', {})
    train_config = config.get('training', {})
    
    image_dir = data_config.get('face_data_dir', 'data/face')
    label_file = data_config.get('face_label_file', 'data/face/labels.txt')
    image_size = data_config.get('image_size', 112)
    split_ratio = (
        data_config.get('train_split', 0.7),
        data_config.get('val_split', 0.15),
        data_config.get('test_split', 0.15),
    )
    add_degradation = data_config.get('add_quality_degradation', False)
    degradation_types = data_config.get('degradation_types', ['blur', 'brightness'])
    
    batch_size = train_config.get('batch_size', 128)
    num_workers = train_config.get('num_workers', 4)
    
    dataset = FaceDataset(
        image_dir=image_dir,
        label_file=label_file,
        image_size=image_size,
        split=split,
        split_ratio=split_ratio,
        add_degradation=add_degradation,
        degradation_types=degradation_types,
        normalize=True,
        precompute_quality=True,
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
