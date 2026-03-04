"""
System Event Dataset for Fusion Model Training/Evaluation.

Loads pre-generated system events (T_face, T_geo, context features, labels)
from CSV and provides PyTorch Dataset/DataLoader for fusion model training
and evaluation.

Each row represents an authentication event with:
- T_face: Face trust score in [0, 1]
- T_geo: Geofence trust score in [0, 1]
- quality_score: Image quality estimate
- is_inside_geofence: Binary/continuous geofence flag
- time_of_day: Hour of day
- label: 1 = genuine (legitimate user), 0 = impostor (attack)

LABEL CONVENTION (consistent across the entire pipeline):
    1 = genuine (legitimate user)
    0 = impostor (attack)

SCORE CONVENTION:
    T_face, T_geo are trust scores: higher = more genuine.
"""

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def make_subject_disjoint_split(
    dataframe: pd.DataFrame,
    id_col: str = 'identity_id',
    seed: int = 42,
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    split_report_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create subject-disjoint train/val/test splits.

    All rows for an identity go to exactly one partition.
    Uses np.random.default_rng for reproducibility (no global state mutation).

    Args:
        dataframe: DataFrame with at least id_col column.
        id_col: Column name for identity/subject ID.
        seed: RNG seed for reproducibility.
        split_ratio: (train, val, test) fractions summing to 1.
        split_report_path: If given, save JSON split report here.

    Returns:
        (train_ids, val_ids, test_ids) as sorted numpy arrays of identity IDs.

    Raises:
        RuntimeError if identity leakage is detected.
    """
    unique_ids = sorted(dataframe[id_col].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)

    n = len(shuffled)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train:n_train + n_val]
    test_ids = shuffled[n_train + n_val:]

    # Strict disjointness checks
    for a_name, a_set, b_name, b_set in [
        ('train', set(train_ids), 'test', set(test_ids)),
        ('train', set(train_ids), 'val', set(val_ids)),
        ('val', set(val_ids), 'test', set(test_ids)),
    ]:
        overlap = a_set & b_set
        if overlap:
            raise RuntimeError(
                f"Identity leakage detected: {len(overlap)} identities appear "
                f"in both {a_name} and {b_name}. Seeds/logic error."
            )

    logger.info(
        f"Subject-disjoint split (seed={seed}): "
        f"total_ids={n}, train={len(train_ids)}, "
        f"val={len(val_ids)}, test={len(test_ids)}"
    )

    # Count samples per partition
    id_set_train = set(train_ids.tolist())
    id_set_val = set(val_ids.tolist())
    id_set_test = set(test_ids.tolist())

    n_rows_train = int((dataframe[id_col].isin(id_set_train)).sum())
    n_rows_val = int((dataframe[id_col].isin(id_set_val)).sum())
    n_rows_test = int((dataframe[id_col].isin(id_set_test)).sum())

    if split_report_path is not None:
        report = {
            'seed': seed,
            'id_col': id_col,
            'split_ratio': list(split_ratio),
            'total_identities': n,
            'train_identities': len(train_ids),
            'val_identities': len(val_ids),
            'test_identities': len(test_ids),
            'train_rows': n_rows_train,
            'val_rows': n_rows_val,
            'test_rows': n_rows_test,
            'train_ids': sorted([int(x) for x in train_ids]),
            'val_ids': sorted([int(x) for x in val_ids]),
            'test_ids': sorted([int(x) for x in test_ids]),
        }
        Path(split_report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(split_report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved split report to {split_report_path}")

    return train_ids, val_ids, test_ids


class SystemEventDataset(Dataset):
    """
    PyTorch Dataset for system-level authentication events.

    Args:
        csv_path: Path to CSV with system events
        split: 'train', 'val', or 'test'
        split_ratio: (train, val, test) ratios
        seed: Random seed for deterministic splitting
        split_report_path: Optional path to save split_report.json
    """

    def __init__(
        self,
        csv_path: str,
        split: str = 'train',
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
        split_report_path: Optional[str] = None,
    ):
        self.csv_path = Path(csv_path)
        self.split = split
        self.seed = seed

        self.data = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.data)} system events from {csv_path}")

        # Subject-disjoint split when an identity column is present.
        id_col = None
        for candidate in ('user_id', 'identity_id', 'identity', 'subject_id'):
            if candidate in self.data.columns:
                id_col = candidate
                break

        if id_col is not None:
            self.indices = self._subject_disjoint_split(
                id_col, split, split_ratio, seed, split_report_path)
        else:
            logger.warning(
                "SystemEventDataset: no identity column found. "
                "Falling back to row-wise permutation split. "
                "Subject-disjointness must be guaranteed upstream."
            )
            self.indices = self._rowwise_split(split, split_ratio, seed)

        logger.info(f"SystemEventDataset split '{split}': {len(self.indices)} samples")

    # ------------------------------------------------------------------
    # Split helpers
    # ------------------------------------------------------------------

    def _subject_disjoint_split(
        self,
        id_col: str,
        split: str,
        split_ratio: tuple,
        seed: int,
        split_report_path: Optional[str] = None,
    ) -> np.ndarray:
        """Identity-based split using make_subject_disjoint_split."""
        train_ids, val_ids, test_ids = make_subject_disjoint_split(
            self.data,
            id_col=id_col,
            seed=seed,
            split_ratio=split_ratio,
            split_report_path=split_report_path,
        )

        if split == 'train':
            selected = set(train_ids.tolist())
        elif split == 'val':
            selected = set(val_ids.tolist())
        else:
            selected = set(test_ids.tolist())

        row_indices = self.data.index[self.data[id_col].isin(selected)].values
        return np.array(row_indices, dtype=np.int64)

    def _rowwise_split(
        self,
        split: str,
        split_ratio: tuple,
        seed: int,
    ) -> np.ndarray:
        """Row-wise permutation split (legacy fallback, no identity column)."""
        rng = np.random.default_rng(seed)
        n = len(self.data)
        indices = rng.permutation(n)
        train_size = int(n * split_ratio[0])
        val_size = int(n * split_ratio[1])
        if split == 'train':
            return indices[:train_size]
        elif split == 'val':
            return indices[train_size:train_size + val_size]
        else:
            return indices[train_size + val_size:]

    def get_all_indices(self) -> np.ndarray:
        """Return the indices used for this split."""
        return self.indices.copy()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_idx = self.indices[idx]
        row = self.data.iloc[sample_idx]

        T_face = float(row['T_face'])
        T_geo = float(row['T_geo'])
        quality_score = float(row.get('quality_score', 0.5))
        is_inside = float(row.get('is_inside_geofence', 0.0))
        time_of_day = float(row.get('time_of_day', 12.0))

        # Time cyclical encoding
        time_sin = float(np.sin(2 * np.pi * time_of_day / 24.0))
        time_cos = float(np.cos(2 * np.pi * time_of_day / 24.0))

        features = torch.tensor([
            T_face,
            T_geo,
            quality_score,
            is_inside,
            time_sin,
            time_cos,
        ], dtype=torch.float32)

        label = int(row['label'])

        return {
            'features': features,
            'T_face': torch.tensor(T_face, dtype=torch.float32),
            'T_geo': torch.tensor(T_geo, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
        }


def get_dataloaders(
    csv_path: str,
    batch_size: int = 64,
    seed: int = 42,
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    num_workers: int = 0,
    split_report_path: Optional[str] = None,
) -> Dict[str, DataLoader]:
    """Create train/val/test DataLoaders from the system events CSV."""
    loaders = {}
    for split in ('train', 'val', 'test'):
        ds = SystemEventDataset(
            csv_path=csv_path,
            split=split,
            split_ratio=split_ratio,
            seed=seed,
            split_report_path=(split_report_path if split == 'train' else None),
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
        )
    return loaders


def build_system_event_dataloader(
    config: Dict[str, Any],
    split: str = 'train',
) -> DataLoader:
    """
    Build a single DataLoader for the given split using project config dict.

    Reads:
        config['data']['system_events_path']
        config['training']['batch_size']
        config['experiment']['seed'] or config['seed']
        config['dataset']['train_ratio'], ['val_ratio'], ['test_ratio']

    Args:
        config: Full experiment config dictionary.
        split: One of 'train', 'val', 'test'.

    Returns:
        DataLoader for the requested split.
    """
    data_cfg = config.get('data', {})
    csv_path = data_cfg.get('system_events_path', 'data/geo/system_events.csv')

    training_cfg = config.get('training', {})
    batch_size = training_cfg.get('batch_size', 64)

    seed = config.get('seed', config.get('experiment', {}).get('seed', 42))

    dataset_cfg = config.get('dataset', {})
    train_ratio = dataset_cfg.get('train_ratio', 0.7)
    val_ratio = dataset_cfg.get('val_ratio', 0.15)
    test_ratio = dataset_cfg.get('test_ratio', 0.15)
    split_ratio = (train_ratio, val_ratio, test_ratio)

    ds = SystemEventDataset(
        csv_path=csv_path,
        split=split,
        split_ratio=split_ratio,
        seed=seed,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=0,
    )
