"""
LFW Dataset Preparation Script

Quick setup to prepare LFW dataset for training:
1. Scans lfw-deepfunneled directory
2. Validates images
3. Creates labels.csv
4. (Optional) Creates smaller subset for testing

Usage:
    python scripts/prepare_lfw.py
    python scripts/prepare_lfw.py --subset 100 2000  # 100 people, 2000 images
"""

import argparse
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.lfw_utils import prepare_lfw_dataset, create_subset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare LFW dataset for training'
    )
    parser.add_argument(
        '--lfw-root',
        type=str,
        default='data/face/lfw-deepfunneled',
        help='Path to lfw-deepfunneled directory'
    )
    parser.add_argument(
        '--output-labels',
        type=str,
        default='data/face/labels.csv',
        help='Output path for labels.csv'
    )
    parser.add_argument(
        '--min-images',
        type=int,
        default=2,
        help='Minimum images per person'
    )
    parser.add_argument(
        '--subset',
        type=int,
        nargs=2,
        metavar=('NUM_IDENTITIES', 'NUM_SAMPLES'),
        help='Create subset with NUM_IDENTITIES people and NUM_SAMPLES images'
    )
    parser.add_argument(
        '--subset-output',
        type=str,
        default='data/face/labels_subset.csv',
        help='Output path for subset labels'
    )
    
    args = parser.parse_args()
    
    logger.info("Starting LFW dataset preparation...")
    
    # Prepare full dataset
    df = prepare_lfw_dataset(
        lfw_root=args.lfw_root,
        output_labels_path=args.output_labels,
        min_images_per_person=args.min_images,
    )
    
    logger.info(f"\n✓ Full dataset prepared:")
    logger.info(f"  Labels: {args.output_labels}")
    logger.info(f"  Samples: {len(df)}")
    logger.info(f"  Identities: {df['identity_id'].max() + 1}")
    
    # Create subset if requested
    if args.subset:
        num_identities, num_samples = args.subset
        logger.info(f"\nCreating subset: {num_identities} identities, {num_samples} samples...")
        
        subset_df = create_subset(
            df,
            num_identities=num_identities,
            num_samples=num_samples,
            output_path=args.subset_output,
        )
        
        logger.info(f"\n✓ Subset created:")
        logger.info(f"  Labels: {args.subset_output}")
        logger.info(f"  Samples: {len(subset_df)}")
        logger.info(f"  Identities: {subset_df['identity_id'].max() + 1}")
    
    logger.info("\n" + "="*60)
    logger.info("LFW dataset preparation complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
