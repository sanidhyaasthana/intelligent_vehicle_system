"""
Configuration Utilities
=======================

Provides functions for loading, saving, and managing YAML configuration files.
Supports hierarchical configuration with default values and overrides.

Design Choices:
- YAML format for human readability and research reproducibility
- Deep merge capability for config inheritance
- Type preservation for numerical values
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
import copy


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is malformed.

    Example:
        >>> config = load_config('config/face_baseline.yaml')
        >>> print(config['model']['backbone'])
        'resnet50'
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Add metadata about config source
    config['_meta'] = {
        'config_path': str(config_path.absolute()),
        'config_name': config_path.stem
    }

    return config


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """
    Save a configuration dictionary to a YAML file.

    Args:
        config: Configuration dictionary to save.
        save_path: Path where to save the configuration.

    Note:
        Creates parent directories if they don't exist.
        Removes internal metadata fields before saving.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a copy without metadata
    config_to_save = {k: v for k, v in config.items() if not k.startswith('_')}

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_to_save, f, default_flow_style=False, sort_keys=False)


def merge_configs(base_config: Dict[str, Any],
                  override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.

    Values in override_config take precedence over base_config.
    Nested dictionaries are merged recursively.

    Args:
        base_config: Base configuration dictionary.
        override_config: Override configuration dictionary.

    Returns:
        Merged configuration dictionary.

    Example:
        >>> base = {'model': {'backbone': 'resnet50', 'dim': 512}}
        >>> override = {'model': {'backbone': 'mobilenet'}}
        >>> merged = merge_configs(base, override)
        >>> print(merged['model'])
        {'backbone': 'mobilenet', 'dim': 512}
    """
    result = copy.deepcopy(base_config)

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def get_config_value(config: Dict[str, Any],
                     key_path: str,
                     default: Any = None) -> Any:
    """
    Get a value from a nested configuration using dot notation.

    Args:
        config: Configuration dictionary.
        key_path: Dot-separated path to the value (e.g., 'model.backbone').
        default: Default value if path doesn't exist.

    Returns:
        Value at the specified path, or default if not found.

    Example:
        >>> config = {'model': {'backbone': 'resnet50'}}
        >>> get_config_value(config, 'model.backbone')
        'resnet50'
        >>> get_config_value(config, 'model.missing', 'default')
        'default'
    """
    keys = key_path.split('.')
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def set_config_value(config: Dict[str, Any],
                     key_path: str,
                     value: Any) -> Dict[str, Any]:
    """
    Set a value in a nested configuration using dot notation.

    Args:
        config: Configuration dictionary (modified in place).
        key_path: Dot-separated path to the value.
        value: Value to set.

    Returns:
        Modified configuration dictionary.
    """
    keys = key_path.split('.')
    current = config

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value
    return config


def validate_config(config: Dict[str, Any],
                    required_fields: list) -> tuple[bool, list]:
    """
    Validate that a configuration contains all required fields.

    Args:
        config: Configuration dictionary to validate.
        required_fields: List of required field paths (dot notation).

    Returns:
        Tuple of (is_valid, missing_fields).

    Example:
        >>> config = {'model': {'backbone': 'resnet50'}}
        >>> valid, missing = validate_config(config, ['model.backbone', 'model.dim'])
        >>> print(valid, missing)
        False ['model.dim']
    """
    missing = []

    for field in required_fields:
        if get_config_value(config, field) is None:
            missing.append(field)

    return len(missing) == 0, missing


def create_experiment_dir(config: Dict[str, Any],
                          base_dir: str = 'results') -> str:
    """
    Create an experiment directory based on configuration.

    Args:
        config: Configuration dictionary.
        base_dir: Base results directory.

    Returns:
        Path to the created experiment directory.
    """
    exp_name = get_config_value(config, 'experiment.name', 'unnamed')
    results_dir = get_config_value(config, 'results.dir',
                                   os.path.join(base_dir, exp_name))

    # Create directory structure
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Save a copy of the config used
    config_save_path = os.path.join(results_dir, 'config_used.yaml')
    save_config(config, config_save_path)

    return results_dir


def print_config(config: Dict[str, Any], indent: int = 0) -> None:
    """
    Pretty print a configuration dictionary.

    Args:
        config: Configuration dictionary to print.
        indent: Current indentation level.
    """
    for key, value in config.items():
        if key.startswith('_'):
            continue
        prefix = '  ' * indent
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_config(value, indent + 1)
        else:
            print(f"{prefix}{key}: {value}")
