"""
LFW (Labeled Faces in the Wild) Dataset Utilities

Handles loading and preprocessing the LFW deepfunneled dataset:
- Automatic directory structure detection
- Identity label extraction from folder names
- Train/val/test splitting with person-level stratification
- CSV label file generation
- Quality filtering and duplicate handling
- Statistics and visualization

LFW Directory Structure:
    lfw-deepfunneled/
    ├── Aaron_Eckhart/
    │   ├── Aaron_Eckhart_0001.jpg
    │   ├── Aaron_Eckhart_0002.jpg
    │   └── ...
    ├── Abigail_Breslin/
    ├── ...
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LFWDatasetHandler:
    """
    Handles LFW deepfunneled dataset loading and preprocessing.
    
    Args:
        lfw_root: Root directory of lfw-deepfunneled
        min_images_per_person: Minimum images required per person (default: 2)
        min_face_size: Minimum face size in pixels (default: 50)
    """
    
    def __init__(
        self,
        lfw_root: str = 'data/face/lfw-deepfunneled',
        min_images_per_person: int = 2,
        min_face_size: int = 50,
    ):
        self.lfw_root = Path(lfw_root)
        self.min_images_per_person = min_images_per_person
        self.min_face_size = min_face_size
        
        if not self.lfw_root.exists():
            raise ValueError(f"LFW root directory not found: {lfw_root}")
        
        logger.info(f"Initializing LFW handler from: {lfw_root}")
    
    def scan_dataset(self) -> Dict[str, List[str]]:
        """
        Scan LFW directory and extract identity -> image mappings.
        
        Returns:
            Dictionary mapping person name to list of image paths
        """
        identity_images = {}
        
        # Scan each person directory
        for person_dir in sorted(self.lfw_root.iterdir()):
            if not person_dir.is_dir():
                continue
            
            person_name = person_dir.name
            images = []
            
            # Collect image paths
            for img_file in sorted(person_dir.glob('*.jpg')):
                images.append(str(img_file))
            
            # Filter by minimum images per person
            if len(images) >= self.min_images_per_person:
                identity_images[person_name] = images
                logger.debug(f"Person '{person_name}': {len(images)} images")
        
        logger.info(f"Found {len(identity_images)} people with {self.min_images_per_person}+ images")
        total_images = sum(len(imgs) for imgs in identity_images.values())
        logger.info(f"Total images: {total_images}")
        
        return identity_images
    
    def validate_images(self, identity_images: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Validate image files and remove corrupted/invalid images.
        
        Returns:
            Cleaned identity -> image mappings
        """
        valid_images = {}
        removed_count = 0
        
        for person_name, images in identity_images.items():
            valid = []
            
            for img_path in images:
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        logger.warning(f"Cannot read image: {img_path}")
                        removed_count += 1
                        continue
                    
                    # Check image size
                    h, w = img.shape[:2]
                    if h < self.min_face_size or w < self.min_face_size:
                        logger.warning(f"Image too small: {img_path} ({h}x{w})")
                        removed_count += 1
                        continue
                    
                    valid.append(img_path)
                
                except Exception as e:
                    logger.warning(f"Error validating {img_path}: {e}")
                    removed_count += 1
            
            # Keep person only if they have minimum images after validation
            if len(valid) >= self.min_images_per_person:
                valid_images[person_name] = valid
        
        logger.info(f"Validation complete: removed {removed_count} images")
        logger.info(f"Valid people: {len(valid_images)}")
        
        return valid_images
    
    def create_label_file(
        self,
        identity_images: Dict[str, List[str]],
        output_path: str = 'data/face/labels.csv',
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Create labels.csv file from identity -> image mappings.
        
        Format:
            image_path,identity_id
            images/person_1_img1.jpg,0
            images/person_1_img2.jpg,0
        
        Args:
            identity_images: Dictionary from scan_dataset()
            output_path: Where to save labels.csv
            seed: Random seed for identity ID assignment
            
        Returns:
            DataFrame with columns [image_path, identity_id]
        """
        np.random.seed(seed)
        
        labels_list = []
        identity_to_id = {}
        
        # Assign sequential identity IDs
        for identity_id, person_name in enumerate(sorted(identity_images.keys())):
            identity_to_id[person_name] = identity_id
            images = identity_images[person_name]
            
            for img_path in images:
                # Store relative path from data/face/
                rel_path = os.path.relpath(img_path, 'data/face')
                
                labels_list.append({
                    'image_path': rel_path,
                    'identity_id': identity_id,
                })
        
        # Create DataFrame
        df = pd.DataFrame(labels_list)
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Created labels file: {output_path}")
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Total identities: {len(identity_to_id)}")
        
        return df
    
    def get_statistics(self, identity_images: Dict[str, List[str]]) -> Dict:
        """Get dataset statistics."""
        image_counts = [len(imgs) for imgs in identity_images.values()]
        
        stats = {
            'num_identities': len(identity_images),
            'num_images': sum(image_counts),
            'min_images_per_person': min(image_counts),
            'max_images_per_person': max(image_counts),
            'mean_images_per_person': np.mean(image_counts),
            'median_images_per_person': np.median(image_counts),
        }
        
        return stats
    
    def print_statistics(self, identity_images: Dict[str, List[str]]):
        """Print formatted statistics."""
        stats = self.get_statistics(identity_images)
        
        print("\n" + "="*60)
        print("LFW DATASET STATISTICS")
        print("="*60)
        print(f"Number of identities: {stats['num_identities']}")
        print(f"Total images: {stats['num_images']}")
        print(f"Images per person:")
        print(f"  Min: {stats['min_images_per_person']}")
        print(f"  Max: {stats['max_images_per_person']}")
        print(f"  Mean: {stats['mean_images_per_person']:.2f}")
        print(f"  Median: {stats['median_images_per_person']:.0f}")
        print("="*60 + "\n")


def prepare_lfw_dataset(
    lfw_root: str = 'data/face/lfw-deepfunneled',
    output_labels_path: str = 'data/face/labels.csv',
    min_images_per_person: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    One-stop function to prepare LFW dataset for training.
    
    Steps:
    1. Scan LFW directory
    2. Validate images
    3. Create labels.csv
    
    Args:
        lfw_root: Path to lfw-deepfunneled directory
        output_labels_path: Where to save labels.csv
        min_images_per_person: Minimum images per person to include
        seed: Random seed
        
    Returns:
        DataFrame with labels
        
    Example:
        >>> df = prepare_lfw_dataset('data/face/lfw-deepfunneled')
        >>> print(f"Prepared {len(df)} images from {df['identity_id'].max()+1} people")
    """
    handler = LFWDatasetHandler(lfw_root, min_images_per_person)
    
    # Scan and validate
    identity_images = handler.scan_dataset()
    valid_images = handler.validate_images(identity_images)
    
    # Print statistics
    handler.print_statistics(valid_images)
    
    # Create labels
    df = handler.create_label_file(valid_images, output_labels_path, seed)
    
    return df


def create_subset(
    labels_df: pd.DataFrame,
    num_identities: Optional[int] = None,
    num_samples: Optional[int] = None,
    output_path: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create a smaller subset of LFW for faster experimentation.
    
    Args:
        labels_df: Original labels DataFrame
        num_identities: Number of identities to keep (None = all)
        num_samples: Total number of samples to keep (None = all)
        output_path: If provided, save subset to this CSV
        seed: Random seed for reproducibility
        
    Returns:
        Subset DataFrame
        
    Example:
        >>> subset_df = create_subset(df, num_identities=100, num_samples=2000)
    """
    np.random.seed(seed)
    
    subset = labels_df.copy()
    
    # Filter by number of identities
    if num_identities is not None:
        identity_ids = np.random.choice(
            subset['identity_id'].unique(),
            size=min(num_identities, len(subset['identity_id'].unique())),
            replace=False
        )
        subset = subset[subset['identity_id'].isin(identity_ids)]
    
    # Filter by number of samples
    if num_samples is not None:
        if len(subset) > num_samples:
            subset = subset.sample(n=num_samples, random_state=seed)
    
    # Renumber identity IDs sequentially
    old_to_new = {old_id: new_id for new_id, old_id in 
                  enumerate(sorted(subset['identity_id'].unique()))}
    subset['identity_id'] = subset['identity_id'].map(old_to_new)
    
    # Save if requested
    if output_path:
        subset.to_csv(output_path, index=False)
        logger.info(f"Saved subset to {output_path}")
    
    logger.info(f"Subset: {len(subset)} samples, {subset['identity_id'].max()+1} identities")
    
    return subset


def get_identity_distribution(labels_df: pd.DataFrame) -> Dict:
    """Get distribution of images per identity."""
    dist = labels_df['identity_id'].value_counts().to_dict()
    return dist
