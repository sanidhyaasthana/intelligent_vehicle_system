"""
Multi-Seed Evaluator.

Runs evaluation across multiple seeds and aggregates metrics
with mean +/- std for reproducibility reporting.

STEP 10: Multi-seed aggregation.
"""

import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Callable

logger = logging.getLogger(__name__)


def aggregate_metrics(
    per_seed_metrics: List[Dict[str, Any]],
    metric_keys: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-seed metrics into mean +/- std.

    Args:
        per_seed_metrics: List of metric dicts, one per seed.
        metric_keys: Which keys to aggregate.

    Returns:
        Dict mapping metric_name -> {'mean': float, 'std': float, 'values': list}
    """
    aggregated = {}
    for key in metric_keys:
        values = []
        for m in per_seed_metrics:
            if key in m and isinstance(m[key], (int, float)):
                values.append(float(m[key]))
        if values:
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values,
            }
    return aggregated


def run_multi_seed_evaluation(
    seeds: List[int],
    eval_fn: Callable,
    config: Dict[str, Any],
    metric_keys: List[str],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Run evaluation function across multiple seeds and aggregate results.

    Args:
        seeds: List of random seeds.
        eval_fn: Function(config, seed) -> Dict[str, Any] returning per-seed metrics.
        config: Configuration dictionary.
        metric_keys: Metric keys to aggregate.
        output_dir: Directory to save per-seed and aggregated results.

    Returns:
        Aggregated results dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_seed_metrics = []

    for seed in seeds:
        logger.info(f"\n=== Evaluating seed {seed} ===")
        metrics = eval_fn(config, seed)
        all_seed_metrics.append(metrics)

        # Save per-seed
        seed_dir = output_dir / f'seed_{seed}'
        seed_dir.mkdir(parents=True, exist_ok=True)
        save_metrics = {k: v for k, v in metrics.items()
                        if not k.startswith('_') and isinstance(v, (int, float, str, bool))}
        with open(seed_dir / 'metrics.json', 'w') as f:
            json.dump(save_metrics, f, indent=2)

    # Aggregate
    aggregated = aggregate_metrics(all_seed_metrics, metric_keys)

    logger.info("\n" + "=" * 60)
    logger.info("MULTI-SEED AGGREGATION")
    logger.info("=" * 60)
    for key, val in aggregated.items():
        logger.info(f"  {key}: {val['mean']:.4f} +/- {val['std']:.4f}")

    # Save aggregated
    save_agg = {k: {'mean': v['mean'], 'std': v['std']} for k, v in aggregated.items()}
    with open(output_dir / 'aggregated_metrics.json', 'w') as f:
        json.dump(save_agg, f, indent=2)

    # Save system_comparison.csv (STEP 10 format)
    rows = []
    for key in metric_keys:
        if key in aggregated:
            rows.append({
                'metric': key,
                'mean': aggregated[key]['mean'],
                'std': aggregated[key]['std'],
                'formatted': f"{aggregated[key]['mean']:.4f} +/- {aggregated[key]['std']:.4f}",
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'system_comparison.csv', index=False)
    logger.info(f"Saved system_comparison.csv to {output_dir}")

    # Save final_metrics.csv
    df.to_csv(output_dir / 'final_metrics.csv', index=False)

    return {
        'aggregated': save_agg,
        'per_seed': all_seed_metrics,
        'seeds': seeds,
    }
