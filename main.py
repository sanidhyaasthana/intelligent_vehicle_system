"""
Main entry point for the intelligent vehicle authorization system.

Usage:
    python main.py --config config/face_baseline.yaml --mode train_face
    python main.py --config config/fusion_full_system.yaml --mode eval_system
"""
import sys
sys.path.append(".")
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

from src.utils.config_utils import load_config, save_config
from src.utils.logger import setup_logging
from src.utils.seed_utils import set_seed

# Import training functions
from src.training import (
    train_face, eval_face,
    train_geo, eval_geo,
    train_fusion, eval_system
)

# Import simulation
from src.simulation import generate_geo_dataset, generate_system_events


def setup_argparse() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Risk-Aware Vehicle Authorization Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train face model with fixed margin:
    python main.py --config config/face_baseline.yaml --mode train_face
  
  Train face model with adaptive margin:
    python main.py --config config/face_adaptive.yaml --mode train_face
  
  Evaluate trained face model:
    python main.py --config config/face_baseline.yaml --mode eval_face
  
  Train geofence model:
    python main.py --config config/geo_prob.yaml --mode train_geo
  
  Train and evaluate full fusion system:
    python main.py --config config/fusion_full_system.yaml --mode train_fusion
    python main.py --config config/fusion_full_system.yaml --mode eval_system
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=[
            'train_face', 'eval_face',
            'train_geo', 'eval_geo',
            'train_fusion', 'eval_system',
            'gen_geo_data', 'gen_system_events'
        ],
        help='Operation mode to execute'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = setup_argparse()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("Risk-Aware Vehicle Authorization Framework")
    logger.info("="*80)
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(str(config_path))
    logger.info(f"Loaded config from: {config_path}")
    
    # Override with command line arguments
    config['device'] = args.device
    config['seed'] = args.seed
    
    set_seed(args.seed)
    
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Seed: {args.seed}")
    
    # Execute requested mode
    try:
        if args.mode == 'train_face':
            logger.info("Starting face model training...")
            result = train_face(config)
            logger.info(f"Face training complete: {result}")
        
        elif args.mode == 'eval_face':
            logger.info("Starting face model evaluation...")
            result = eval_face(config)
            logger.info(f"Face evaluation complete: {result}")
        
        elif args.mode == 'train_geo':
            logger.info("Starting geofence model training...")
            result = train_geo(config)
            logger.info(f"Geo training complete: {result}")
        
        elif args.mode == 'eval_geo':
            logger.info("Starting geofence model evaluation...")
            result = eval_geo(config)
            logger.info(f"Geo evaluation complete: {result}")
        
        elif args.mode == 'train_fusion':
            logger.info("Starting fusion model training...")
            result = train_fusion(config)
            logger.info(f"Fusion training complete: {result}")
        
        elif args.mode == 'eval_system':
            logger.info("Starting end-to-end system evaluation...")
            result = eval_system(config)
            logger.info(f"System evaluation complete: {result}")
        
        elif args.mode == 'gen_geo_data':
            logger.info("Generating synthetic geofence data...")
            data_config = config.get('data', {})
            output_path = data_config.get('geo_data_file', 'data/geo/location_data.csv')
            generate_geo_dataset(output_path)
            logger.info(f"Geofence data saved to {output_path}")
        
        elif args.mode == 'gen_system_events':
            logger.info("Generating synthetic system events...")
            data_config = config.get('data', {})
            output_path = data_config.get('system_events_path') \
                or data_config.get('system_events_file')
            if not output_path:
                raise RuntimeError(
                    "Config must define data.system_events_path for gen_system_events mode."
                )
            generate_system_events(output_path)
            logger.info(f"System events saved to {output_path}")
        
        logger.info("="*80)
        logger.info("Execution complete!")
        logger.info("="*80)
    
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
