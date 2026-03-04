"""
Seed Utilities
==============

Provides functions for setting and managing random seeds for reproducibility.
Ensures consistent behavior across NumPy, PyTorch, and Python's random module.

Design Choices:
- Single function to set all relevant seeds
- Support for deterministic cuDNN operations
- Global seed tracking for logging purposes
"""

import os
import random
from typing import Optional

import numpy as np
import torch


# Global variable to track the current seed
_GLOBAL_SEED: Optional[int] = None


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - cuDNN (for deterministic GPU operations)

    Args:
        seed: Integer seed value.
        deterministic: If True, configure cuDNN for deterministic behavior.
                       DISABLED BY DEFAULT due to CUDA compatibility.
                       When disabled, training runs faster but is not deterministic.

    Example:
        >>> set_seed(42)
        >>> torch.rand(3)
        tensor([0.8823, 0.9150, 0.3829])
        >>> set_seed(42)
        >>> torch.rand(3)
        tensor([0.8823, 0.9150, 0.3829])

    Note:
        Deterministic mode is disabled by default as it can cause CUDA errors.
        Enable only if reproducibility is critical and GPU memory is available.
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuDNN determinism settings
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set CUBLAS environment variable for reproducibility with CUDA >= 10.2
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # For PyTorch >= 1.8
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                # Some operations don't have deterministic implementations
                pass

    # Environment variable for complete reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_seed() -> Optional[int]:
    """
    Get the currently set global seed.

    Returns:
        The current global seed, or None if not set.

    Example:
        >>> set_seed(42)
        >>> get_seed()
        42
    """
    return _GLOBAL_SEED


def seed_worker(worker_id: int) -> None:
    """
    Worker initialization function for DataLoader reproducibility.

    Use this with DataLoader's worker_init_fn parameter to ensure
    reproducible data loading with multiple workers.

    Args:
        worker_id: Worker process ID (passed by DataLoader).

    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, worker_init_fn=seed_worker)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: Optional[int] = None) -> torch.Generator:
    """
    Create a PyTorch Generator with a specific seed.

    Useful for reproducible random operations like data shuffling.

    Args:
        seed: Seed for the generator. Uses global seed if None.

    Returns:
        Seeded PyTorch Generator.

    Example:
        >>> g = get_generator(42)
        >>> torch.randperm(5, generator=g)
        tensor([3, 4, 0, 2, 1])
    """
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    elif _GLOBAL_SEED is not None:
        g.manual_seed(_GLOBAL_SEED)
    return g


def random_split_indices(n: int,
                         ratios: list,
                         seed: Optional[int] = None) -> list:
    """
    Generate random split indices for dataset partitioning.

    Args:
        n: Total number of samples.
        ratios: List of split ratios (should sum to 1.0).
        seed: Random seed for reproducibility.

    Returns:
        List of index arrays for each split.

    Example:
        >>> indices = random_split_indices(100, [0.7, 0.15, 0.15], seed=42)
        >>> len(indices[0]), len(indices[1]), len(indices[2])
        (70, 15, 15)
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    elif _GLOBAL_SEED is not None:
        rng = np.random.RandomState(_GLOBAL_SEED)
    else:
        rng = np.random.RandomState()

    # Shuffle indices
    indices = rng.permutation(n)

    # Calculate split points
    split_sizes = [int(r * n) for r in ratios[:-1]]
    split_sizes.append(n - sum(split_sizes))  # Ensure all samples included

    # Split indices
    splits = []
    start = 0
    for size in split_sizes:
        splits.append(indices[start:start + size])
        start += size

    return splits


def print_seed_info() -> None:
    """Print information about current seed settings."""
    print(f"Global seed: {_GLOBAL_SEED}")
    print(f"PyTorch seed: {torch.initial_seed()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuDNN deterministic: {torch.backends.cudnn.deterministic}")
        print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
