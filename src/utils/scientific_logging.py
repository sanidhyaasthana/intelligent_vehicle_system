"""
Scientific Logging Utilities.

Provides structured logging for reproducible research experiments.
"""

import logging
import json
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class PhaseLogger:
    """Log distinct phases of an experiment pipeline."""

    def __init__(self, experiment_name: str = "experiment"):
        self.experiment_name = experiment_name
        self.phases = []
        self.current_phase = None

    def start_phase(self, name: str):
        self.current_phase = name
        self.phases.append(name)
        logger.info(f"[{self.experiment_name}] Phase started: {name}")

    def end_phase(self, name: str, status: str = "OK"):
        logger.info(f"[{self.experiment_name}] Phase ended: {name} — {status}")
        self.current_phase = None

    def log(self, message: str):
        prefix = f"[{self.current_phase}]" if self.current_phase else ""
        logger.info(f"{prefix} {message}")


class FailureAssertion:
    """Collect and report assertion failures without immediately crashing."""

    def __init__(self):
        self.failures = []

    def check(self, condition: bool, message: str):
        if not condition:
            self.failures.append(message)
            logger.error(f"ASSERTION FAILED: {message}")

    def has_failures(self) -> bool:
        return len(self.failures) > 0

    def raise_if_failed(self):
        if self.failures:
            msg = "\n".join(f"  - {f}" for f in self.failures)
            raise RuntimeError(f"Pipeline failed {len(self.failures)} assertions:\n{msg}")


class ReproducibilityLogger:
    """Log reproducibility-relevant information."""

    def __init__(self):
        self.info = {}

    def log_seed(self, seed: int):
        self.info['seed'] = seed
        logger.info(f"Seed: {seed}")

    def log_data_hash(self, data_path: str, hash_val: str):
        self.info[f'data_hash_{data_path}'] = hash_val

    def log_config(self, config: Dict):
        self.info['config'] = config

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.info, f, indent=2, default=str)


class DiagnosticReport:
    """Generate a diagnostic report for debugging failed runs."""

    def __init__(self):
        self.entries = []

    def add(self, key: str, value: Any):
        self.entries.append((key, value))

    def to_string(self) -> str:
        lines = ["=== Diagnostic Report ==="]
        for key, val in self.entries:
            lines.append(f"  {key}: {val}")
        return "\n".join(lines)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.to_string())


class ExpectedMetricRanges:
    """Define and check expected metric ranges for sanity checking."""

    DEFAULT_RANGES = {
        'roc_auc': (0.5, 1.0),
        'eer': (0.0, 0.5),
        'accuracy': (0.4, 1.0),
        'far': (0.0, 1.0),
        'frr': (0.0, 1.0),
    }

    def __init__(self, ranges: Optional[Dict] = None):
        self.ranges = ranges or self.DEFAULT_RANGES

    def check(self, metrics: Dict[str, float]) -> List[str]:
        warnings = []
        for key, (low, high) in self.ranges.items():
            if key in metrics:
                val = metrics[key]
                if val < low or val > high:
                    msg = f"{key}={val:.4f} outside expected [{low}, {high}]"
                    warnings.append(msg)
                    logger.warning(msg)
        return warnings
