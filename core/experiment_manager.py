import os
import yaml
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Experiment:
    name: str
    experiment_dir: Path
    config_path: Path
    logs_path: Path


def create_experiment(config: dict) -> Experiment:
    """Create a timestamped experiment directory, save config, and init logging.

    Directory format: results/YYYY-MM-DD_HH-MM-SS_<experiment_name>/
    """
    base_results = Path(config.get('results_dir', 'results'))
    name = config.get('experiment_name', config.get('name', 'experiment'))
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_dir = base_results / f"{ts}_{name}"
    exp_dir.mkdir(parents=True, exist_ok=False)

    # Save config snapshot
    cfg_path = exp_dir / 'config.yaml'
    with open(cfg_path, 'w') as f:
        yaml.safe_dump(config, f)

    # Initialize simple file logger
    logs_path = exp_dir / 'logs.txt'
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # file handler
    fh = logging.FileHandler(str(logs_path))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Also keep a minimal stdout logger
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # Register artifact subpaths
    for sub in ['artifacts', 'plots', 'predictions']:
        (exp_dir / sub).mkdir(exist_ok=True)

    return Experiment(name=name, experiment_dir=exp_dir, config_path=cfg_path, logs_path=logs_path)
