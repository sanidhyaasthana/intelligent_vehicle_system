"""
Logging Utilities
=================

Provides a comprehensive logging system for research experiments.
Supports console output, CSV logging, and optional TensorBoard/W&B integration.

Design Choices:
- CSV format for easy analysis in pandas/spreadsheets
- Automatic directory creation
- Thread-safe for multi-process training
- Rich formatting for console output
"""

import os
import sys
import csv
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import threading

import numpy as np


class Logger:
    """
    Comprehensive logger for research experiments.

    Handles multiple output destinations:
    - Console (stdout)
    - CSV file for metrics
    - JSON file for configuration and summaries
    - Optional TensorBoard integration
    - Optional Weights & Biases integration

    Attributes:
        experiment_name: Name of the current experiment.
        log_dir: Directory for log files.
        csv_path: Path to the metrics CSV file.

    Example:
        >>> logger = Logger('face_baseline', 'results/face/baseline')
        >>> logger.log_metrics({'loss': 0.5, 'accuracy': 0.9}, step=100)
        >>> logger.log_text('Training completed successfully')
        >>> logger.close()
    """

    def __init__(self,
                 experiment_name: str,
                 log_dir: str,
                 tensorboard: bool = False,
                 wandb: bool = False,
                 wandb_project: str = 'vehicle-auth',
                 console: bool = True):
        """
        Initialize the logger.

        Args:
            experiment_name: Name of the experiment.
            log_dir: Directory to save logs.
            tensorboard: Enable TensorBoard logging.
            wandb: Enable Weights & Biases logging.
            wandb_project: W&B project name.
            console: Enable console output.
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.console = console
        self._lock = threading.Lock()

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CSV logger
        self.csv_path = self.log_dir / 'metrics.csv'
        self._csv_file = None
        self._csv_writer = None
        self._csv_fields = None

        # Initialize text log
        self.text_log_path = self.log_dir / 'log.txt'
        self._text_file = open(self.text_log_path, 'a', encoding='utf-8')

        # Optional integrations
        self._tb_writer = None
        self._wandb_run = None

        if tensorboard:
            self._init_tensorboard()

        if wandb:
            self._init_wandb(wandb_project)

        # Log start time
        self.start_time = time.time()
        self.log_text(f"Experiment '{experiment_name}' started at "
                      f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def _init_tensorboard(self):
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = self.log_dir / 'tensorboard'
            self._tb_writer = SummaryWriter(log_dir=str(tb_dir))
            self.log_text(f"TensorBoard logging enabled: {tb_dir}")
        except ImportError:
            self.log_text("WARNING: TensorBoard not available, skipping")

    def _init_wandb(self, project: str):
        """Initialize Weights & Biases."""
        try:
            import wandb
            self._wandb_run = wandb.init(
                project=project,
                name=self.experiment_name,
                dir=str(self.log_dir)
            )
            self.log_text(f"Weights & Biases logging enabled: {project}")
        except ImportError:
            self.log_text("WARNING: wandb not available, skipping")

    def log_metrics(self,
                    metrics: Dict[str, Any],
                    step: Optional[int] = None,
                    epoch: Optional[int] = None,
                    prefix: str = '') -> None:
        """
        Log metrics to all enabled destinations.

        Args:
            metrics: Dictionary of metric names and values.
            step: Current training step.
            epoch: Current epoch.
            prefix: Prefix for metric names (e.g., 'train_', 'val_').

        Example:
            >>> logger.log_metrics({'loss': 0.5, 'acc': 0.9}, step=100, prefix='train_')
        """
        with self._lock:
            # Add prefix to metric names
            if prefix:
                metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

            # Add step/epoch info
            if step is not None:
                metrics['step'] = step
            if epoch is not None:
                metrics['epoch'] = epoch
            metrics['timestamp'] = time.time() - self.start_time

            # Log to CSV
            self._log_to_csv(metrics)

            # Log to console
            if self.console:
                self._log_to_console(metrics, step, epoch)

            # Log to TensorBoard
            if self._tb_writer is not None:
                for name, value in metrics.items():
                    if isinstance(value, (int, float)) and name not in ['step', 'epoch', 'timestamp']:
                        global_step = step if step is not None else epoch
                        self._tb_writer.add_scalar(name, value, global_step)

            # Log to W&B
            if self._wandb_run is not None:
                import wandb
                wandb.log(metrics, step=step)

    def _log_to_csv(self, metrics: Dict[str, Any]) -> None:
        """Write metrics to CSV file."""
        if self._csv_file is None:
            # Initialize CSV with headers
            self._csv_fields = list(metrics.keys())
            self._csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fields)
            self._csv_writer.writeheader()

        # Handle new fields
        new_fields = [k for k in metrics.keys() if k not in self._csv_fields]
        if new_fields:
            # Need to rewrite CSV with new columns
            self._csv_file.close()
            self._csv_fields.extend(new_fields)

            # Read existing data
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_data = list(reader)

            # Rewrite with new columns
            self._csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fields)
            self._csv_writer.writeheader()
            self._csv_writer.writerows(existing_data)

        # Write current row
        self._csv_writer.writerow(metrics)
        self._csv_file.flush()

    def _log_to_console(self,
                        metrics: Dict[str, Any],
                        step: Optional[int],
                        epoch: Optional[int]) -> None:
        """Print metrics to console."""
        parts = []

        if epoch is not None:
            parts.append(f"Epoch {epoch}")
        if step is not None:
            parts.append(f"Step {step}")

        for name, value in metrics.items():
            if name in ['step', 'epoch', 'timestamp']:
                continue
            if isinstance(value, float):
                parts.append(f"{name}: {value:.4f}")
            else:
                parts.append(f"{name}: {value}")

        print(" | ".join(parts))

    def log_text(self, message: str, also_print: bool = True) -> None:
        """
        Log a text message.

        Args:
            message: Text message to log.
            also_print: Also print to console.
        """
        with self._lock:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_line = f"[{timestamp}] {message}\n"
            self._text_file.write(log_line)
            self._text_file.flush()

            if also_print and self.console:
                print(f"[{timestamp}] {message}")

    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Save configuration to a JSON file.

        Args:
            config: Configuration dictionary.
        """
        config_path = self.log_dir / 'config.json'
        # Remove non-serializable items
        serializable_config = self._make_serializable(config)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_config, f, indent=2)
        self.log_text(f"Configuration saved to {config_path}")

    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()
                    if not k.startswith('_')}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    def log_distribution(self,
                         name: str,
                         values: Union[np.ndarray, list],
                         step: Optional[int] = None) -> None:
        """
        Log distribution statistics of values.

        Args:
            name: Name of the distribution.
            values: Array of values.
            step: Current step.
        """
        values = np.array(values)
        stats = {
            f'{name}_mean': float(np.mean(values)),
            f'{name}_std': float(np.std(values)),
            f'{name}_min': float(np.min(values)),
            f'{name}_max': float(np.max(values)),
            f'{name}_median': float(np.median(values))
        }
        self.log_metrics(stats, step=step)

        if self._tb_writer is not None:
            self._tb_writer.add_histogram(name, values, global_step=step)

    def log_figure(self,
                   name: str,
                   figure,
                   step: Optional[int] = None,
                   save: bool = True) -> None:
        """
        Log a matplotlib figure.

        Args:
            name: Name for the figure.
            figure: Matplotlib figure object.
            step: Current step.
            save: Save to file.
        """
        if save:
            figure_path = self.log_dir / f'{name}.png'
            figure.savefig(figure_path, dpi=150, bbox_inches='tight')
            self.log_text(f"Figure saved to {figure_path}")

        if self._tb_writer is not None:
            self._tb_writer.add_figure(name, figure, global_step=step)

        if self._wandb_run is not None:
            import wandb
            wandb.log({name: wandb.Image(figure)}, step=step)

    def log_table(self,
                  name: str,
                  data: List[Dict[str, Any]],
                  save: bool = True) -> None:
        """
        Log a table of data.

        Args:
            name: Name for the table.
            data: List of dictionaries (rows).
            save: Save to CSV file.
        """
        if save:
            table_path = self.log_dir / f'{name}.csv'
            if data:
                fields = list(data[0].keys())
                with open(table_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(data)
                self.log_text(f"Table saved to {table_path}")

    def save_summary(self, summary: Dict[str, Any]) -> None:
        """
        Save experiment summary to JSON.

        Args:
            summary: Summary dictionary with final results.
        """
        summary_path = self.log_dir / 'summary.json'
        summary['experiment_name'] = self.experiment_name
        summary['total_time_seconds'] = time.time() - self.start_time

        serializable_summary = self._make_serializable(summary)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_summary, f, indent=2)

        self.log_text(f"Summary saved to {summary_path}")

    def close(self) -> None:
        """Close all log files and writers."""
        total_time = time.time() - self.start_time
        self.log_text(f"Experiment '{self.experiment_name}' finished. "
                      f"Total time: {total_time:.2f}s")

        if self._csv_file is not None:
            self._csv_file.close()

        self._text_file.close()

        if self._tb_writer is not None:
            self._tb_writer.close()

        if self._wandb_run is not None:
            import wandb
            wandb.finish()


# Global logger instance
_GLOBAL_LOGGER: Optional[Logger] = None


def get_logger() -> Optional[Logger]:
    """Get the global logger instance."""
    return _GLOBAL_LOGGER


def set_logger(logger: Logger) -> None:
    """Set the global logger instance."""
    global _GLOBAL_LOGGER
    _GLOBAL_LOGGER = logger


def create_logger(config: Dict[str, Any]) -> Logger:
    """
    Create and set a logger from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Created Logger instance.
    """
    from .config_utils import get_config_value

    exp_name = get_config_value(config, 'experiment.name', 'unnamed')
    log_dir = get_config_value(config, 'results.dir', f'results/{exp_name}')
    tensorboard = get_config_value(config, 'logging.tensorboard', False)
    wandb = get_config_value(config, 'logging.wandb', False)

    logger = Logger(
        experiment_name=exp_name,
        log_dir=log_dir,
        tensorboard=tensorboard,
        wandb=wandb
    )

    set_logger(logger)
    logger.log_config(config)

    return logger


def setup_logging(level: int = logging.INFO,
                  log_file: Optional[str] = None,
                  console: bool = True) -> None:
    """
    Configure the root Python logger for console and optional file output.

    Args:
        level: Logging level (e.g., logging.INFO).
        log_file: Optional path to a file to write logs to.
        console: Whether to enable console (stdout) logging.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers to avoid duplicate logs
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if console:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path))
        fh.setLevel(level)
        fh.setFormatter(fmt)
        root.addHandler(fh)
