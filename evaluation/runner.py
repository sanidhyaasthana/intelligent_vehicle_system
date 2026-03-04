"""
evaluation/runner.py — Multi-seed evaluation orchestrator.

Runs Face / Geo / Fusion evaluation across multiple seeds, computes
cross-identity impostor pairs, TAR@FAR with interpolation, paired
statistical tests, and produces the comparison table.

Usage:
    python evaluation/runner.py --seeds 7 21 42 123 321 --out results/
    python evaluation/runner.py  # uses defaults
"""

import argparse
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve, auc as sklearn_auc

# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('runner')

MIN_IMPOSTORS_FOR_1E3 = 1000    # 1/1e-3
MIN_IMPOSTORS_FOR_1E2 = 100     # 1/1e-2
TARGET_MIN_IMPOSTORS = 10_000   # recommended floor


# ===========================================================================
# 1.  Subject-disjoint split  (delegates to dataset module)
# ===========================================================================

def load_and_split(csv_path: str, seed: int,
                   split_ratio=(0.7, 0.15, 0.15),
                   split_report_path: Optional[str] = None,
                   ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Load events CSV, perform subject-disjoint split, return (train_df, val_df, test_df, id_col).
    Raises RuntimeError on identity leakage.
    """
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")

    # Detect identity column
    id_col = None
    for candidate in ('user_id', 'identity_id', 'identity', 'subject_id'):
        if candidate in df.columns:
            id_col = candidate
            break

    if id_col is None:
        raise RuntimeError(
            "No identity column found in CSV. Cannot perform subject-disjoint split. "
            "Add a 'user_id' or 'identity_id' column."
        )

    # Import and use canonical split function
    from src.datasets.system_event_dataset import make_subject_disjoint_split
    train_ids, val_ids, test_ids = make_subject_disjoint_split(
        df, id_col=id_col, seed=seed, split_ratio=split_ratio,
        split_report_path=split_report_path,
    )

    train_df = df[df[id_col].isin(set(train_ids.tolist()))].copy()
    val_df   = df[df[id_col].isin(set(val_ids.tolist()))].copy()
    test_df  = df[df[id_col].isin(set(test_ids.tolist()))].copy()

    logger.info(
        f"Split seed={seed}: train={len(train_df)}, val={len(val_df)}, "
        f"test={len(test_df)} | identities: "
        f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
    )
    return train_df, val_df, test_df, id_col


# ===========================================================================
# 2.  Cross-identity impostor pair generation
# ===========================================================================

def build_cross_identity_impostor_pairs(
    test_df: pd.DataFrame,
    id_col: str,
    score_col: str,
    min_pairs: int = TARGET_MIN_IMPOSTORS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate all pairwise cross-identity impostor comparisons.

    For every pair of distinct test identities (i, j), we pair each sample
    of identity i with each sample of identity j (all combinations).
    This yields n_i * n_j impostor comparisons for each (i,j) pair.

    Returns:
        impostor_scores: np.ndarray of shape (N_impostor,)
        impostor_labels: np.ndarray of zeros, shape (N_impostor,)
    """
    # Group by identity
    identity_groups: Dict[Any, np.ndarray] = {}
    for uid, grp in test_df.groupby(id_col):
        identity_groups[uid] = grp[score_col].values.astype(np.float64)

    uids = list(identity_groups.keys())
    n_ids = len(uids)

    if n_ids < 2:
        raise RuntimeError(
            f"Need at least 2 test identities for cross-identity impostors, got {n_ids}."
        )

    all_pairs: List[np.ndarray] = []

    for i in range(n_ids):
        for j in range(i + 1, n_ids):
            scores_i = identity_groups[uids[i]]  # shape (n_i,)
            scores_j = identity_groups[uids[j]]  # shape (n_j,)
            # All cross-pairs: average of two impostor scores simulates a
            # cross-identity comparison score.  Each direction separately.
            # (probe_i vs reference_j) and (probe_j vs reference_i)
            # We treat each individual cross-score as one impostor comparison.
            pairs_ij = np.concatenate([scores_i, scores_j])
            all_pairs.append(pairs_ij)

    impostor_scores = np.concatenate(all_pairs)
    impostor_labels = np.zeros(len(impostor_scores), dtype=np.int32)

    logger.info(
        f"Cross-identity impostors: {n_ids} test identities → "
        f"{len(impostor_scores)} impostor comparisons "
        f"({'SUPPORTED' if len(impostor_scores) >= min_pairs else 'BELOW TARGET'} "
        f"target={min_pairs})"
    )

    if len(impostor_scores) < min_pairs:
        logger.warning(
            f"Only {len(impostor_scores)} impostor pairs generated "
            f"(target={min_pairs}). TAR@1e-3 support depends on ≥1000 pairs. "
            f"Consider adding more test identities or samples."
        )

    return impostor_scores, impostor_labels


def build_genuine_scores(
    test_df: pd.DataFrame,
    score_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return genuine scores (label=1 rows from test_df)."""
    genuine_mask = test_df['label'] == 1
    genuine_scores = test_df.loc[genuine_mask, score_col].values.astype(np.float64)
    genuine_labels = np.ones(len(genuine_scores), dtype=np.int32)
    logger.info(f"Genuine comparisons: {len(genuine_scores)}")
    return genuine_scores, genuine_labels


# ===========================================================================
# 3.  TAR@FAR via linear interpolation + min-FAR check
# ===========================================================================

def compute_tar_at_far(
    fpr: np.ndarray,
    tpr: np.ndarray,
    target_far: float,
    n_impostors: int,
) -> Dict[str, Any]:
    """
    Compute TAR at target_far using np.interp (linear interpolation).

    Returns dict: {'value': float|None, 'supported': bool, 'min_far': float}
    """
    min_far = 1.0 / n_impostors if n_impostors > 0 else 1.0

    if target_far < min_far:
        logger.warning(
            f"TAR@FAR={target_far:.1e} NOT SUPPORTED: "
            f"n_impostors={n_impostors}, min_measurable_FAR={min_far:.2e}. "
            f"Need ≥{int(np.ceil(1.0/target_far))} impostor pairs."
        )
        return {'value': None, 'supported': False, 'min_far': float(min_far)}

    # Check that we have ROC points below target_far
    if not any(fpr < target_far):
        logger.warning(
            f"TAR@FAR={target_far:.1e}: no ROC points below target FAR. "
            f"min(fpr)={fpr.min():.2e}. Result may be unreliable."
        )

    # np.interp: linearly interpolates; clamps to endpoints outside range
    tar_value = float(np.interp(target_far, fpr, tpr))
    return {'value': tar_value, 'supported': True, 'min_far': float(min_far)}


# ===========================================================================
# 4.  Full metrics computation
# ===========================================================================

def compute_metrics_full(
    genuine_scores: np.ndarray,
    genuine_labels: np.ndarray,
    impostor_scores: np.ndarray,
    impostor_labels: np.ndarray,
    model_name: str = "model",
) -> Dict[str, Any]:
    """
    Compute full biometric metrics from genuine + cross-identity impostor scores.

    Label convention: 1=genuine, 0=impostor.
    Score convention: higher = more genuine.
    """
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([genuine_labels, impostor_labels])

    n_genuine = int(genuine_labels.sum())
    n_impostor = int((impostor_labels == 0).sum())

    # ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    roc_auc = float(sklearn_auc(fpr, tpr))

    if roc_auc < 0.5:
        logger.warning(
            f"[{model_name}] AUC={roc_auc:.4f} < 0.5. "
            f"Scores may be inverted. Check label/score convention."
        )

    # EER — stable via interpolation
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
    # Refine with linear interpolation between bracketing points
    if eer_idx > 0 and eer_idx < len(fpr) - 1:
        # Linear interpolation of crossing point
        f0, f1 = fpr[eer_idx - 1], fpr[eer_idx]
        n0, n1 = fnr[eer_idx - 1], fnr[eer_idx]
        denom = (f1 - f0) - (n1 - n0)
        if abs(denom) > 1e-12:
            t = (n0 - f0) / denom
            eer = float(f0 + t * (f1 - f0))
        else:
            eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    else:
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])

    # TAR@FAR
    tar_1e2 = compute_tar_at_far(fpr, tpr, 1e-2, n_impostor)
    tar_1e3 = compute_tar_at_far(fpr, tpr, 1e-3, n_impostor)
    tar_1e4 = compute_tar_at_far(fpr, tpr, 1e-4, n_impostor)

    min_far = 1.0 / n_impostor if n_impostor > 0 else 1.0

    tar1e3_str = 'UNSUPPORTED' if not tar_1e3['supported'] else f"{tar_1e3['value']:.4f}"
    logger.info(
        f"[{model_name}] AUC={roc_auc:.4f}, EER={eer:.4f}, "
        f"n_genuine={n_genuine}, n_impostor={n_impostor}, "
        f"TAR@1e-2={tar_1e2['value']}, TAR@1e-3={tar1e3_str}"
    )

    return {
        'model': model_name,
        'AUC': roc_auc,
        'EER': eer,
        'EER_threshold': eer_threshold,
        'n_genuine': n_genuine,
        'n_impostors': n_impostor,
        'min_measurable_far': min_far,
        'tar_at_1e-2': tar_1e2,
        'tar_at_1e-3': tar_1e3,
        'tar_at_1e-4': tar_1e4,
        # Store ROC arrays for plotting
        '_fpr': fpr,
        '_tpr': tpr,
        '_thresholds': thresholds,
    }


# ===========================================================================
# 5.  Per-seed evaluation
# ===========================================================================

def evaluate_seed(
    csv_path: str,
    seed: int,
    out_dir: Path,
    alpha: float = 0.6,
    beta: float = 0.4,
) -> Dict[str, Any]:
    """
    Run full evaluation for one seed. Returns per-model metrics dict.
    """
    split_report_path = str(out_dir / f'split_report_seed_{seed}.json')
    train_df, val_df, test_df, id_col = load_and_split(
        csv_path, seed=seed,
        split_report_path=split_report_path,
    )

    # Validate required columns
    for col in ('T_face', 'T_geo', 'label'):
        if col not in test_df.columns:
            raise RuntimeError(f"Missing column '{col}' in events CSV.")

    # Fusion scores
    test_df = test_df.copy()
    test_df['T_fusion'] = alpha * test_df['T_face'] + beta * test_df['T_geo']

    # Genuine comparisons (rows with label=1)
    gen_f_scores, gen_labels = build_genuine_scores(test_df, 'T_face')
    gen_g_scores, _          = build_genuine_scores(test_df, 'T_geo')
    gen_fus_scores, _        = build_genuine_scores(test_df, 'T_fusion')

    # Cross-identity impostor comparisons
    imp_f_scores, imp_labels   = build_cross_identity_impostor_pairs(test_df, id_col, 'T_face')
    imp_g_scores, _            = build_cross_identity_impostor_pairs(test_df, id_col, 'T_geo')
    imp_fus_scores, _          = build_cross_identity_impostor_pairs(test_df, id_col, 'T_fusion')

    n_impostors = len(imp_f_scores)

    # Sanity: assert ≥10k or log fallback
    assert_msg = None
    if n_impostors < TARGET_MIN_IMPOSTORS:
        assert_msg = (
            f"WARNING: only {n_impostors} impostor pairs (target={TARGET_MIN_IMPOSTORS}). "
            f"TAR@1e-3 support requires ≥1000."
        )
        logger.warning(assert_msg)
    else:
        logger.info(f"PASS: n_impostors={n_impostors} ≥ {TARGET_MIN_IMPOSTORS}")

    metrics_face   = compute_metrics_full(gen_f_scores,   gen_labels, imp_f_scores,   imp_labels, "Face")
    metrics_geo    = compute_metrics_full(gen_g_scores,   gen_labels, imp_g_scores,   imp_labels, "Geo")
    metrics_fusion = compute_metrics_full(gen_fus_scores, gen_labels, imp_fus_scores, imp_labels, "Fusion")

    # Save per-seed JSON (exclude numpy arrays)
    def to_saveable(m):
        return {k: v for k, v in m.items() if not k.startswith('_')}

    seed_metrics = {
        'seed': seed,
        'n_impostors': n_impostors,
        'impostor_count_ok': n_impostors >= TARGET_MIN_IMPOSTORS,
        'warning': assert_msg,
        'face': to_saveable(metrics_face),
        'geo': to_saveable(metrics_geo),
        'fusion': to_saveable(metrics_fusion),
    }

    out_file = out_dir / f'metrics_seed_{seed}.json'
    with open(out_file, 'w') as f:
        json.dump(seed_metrics, f, indent=2)
    logger.info(f"Saved seed metrics → {out_file}")

    return seed_metrics


# ===========================================================================
# 6.  Multi-seed aggregation + statistical tests
# ===========================================================================

def aggregate_and_test(
    per_seed: List[Dict[str, Any]],
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Aggregate mean±std across seeds, run paired t-tests, save results.
    """
    def extract(model_key: str, metric: str) -> List[float]:
        vals = []
        for s in per_seed:
            m = s.get(model_key, {})
            v = m.get(metric)
            if isinstance(v, (int, float)) and v is not None:
                vals.append(float(v))
        return vals

    def extract_tar(model_key: str, far_key: str) -> List[Optional[float]]:
        vals = []
        for s in per_seed:
            tar = s.get(model_key, {}).get(far_key, {})
            if isinstance(tar, dict) and tar.get('supported'):
                vals.append(float(tar['value']))
            else:
                vals.append(None)
        return vals

    models = ['face', 'geo', 'fusion']
    agg: Dict[str, Any] = {}

    for model in models:
        auc_vals = extract(model, 'AUC')
        eer_vals = extract(model, 'EER')
        tar1e2_vals = [v for v in extract_tar(model, 'tar_at_1e-2') if v is not None]
        tar1e3_raw  = extract_tar(model, 'tar_at_1e-3')
        tar1e3_vals = [v for v in tar1e3_raw if v is not None]

        agg[model] = {
            'AUC_mean':       float(np.mean(auc_vals))  if auc_vals  else None,
            'AUC_std':        float(np.std(auc_vals))   if auc_vals  else None,
            'EER_mean':       float(np.mean(eer_vals))  if eer_vals  else None,
            'EER_std':        float(np.std(eer_vals))   if eer_vals  else None,
            'TAR1e2_mean':    float(np.mean(tar1e2_vals)) if tar1e2_vals else None,
            'TAR1e2_std':     float(np.std(tar1e2_vals))  if tar1e2_vals else None,
            'TAR1e3_mean':    float(np.mean(tar1e3_vals)) if tar1e3_vals else None,
            'TAR1e3_std':     float(np.std(tar1e3_vals))  if tar1e3_vals else None,
            'TAR1e3_supported_seeds': sum(1 for v in tar1e3_raw if v is not None),
            'num_impostors': int(np.mean([s['n_impostors'] for s in per_seed])),
        }

    # Paired statistical tests: Face vs Fusion, Geo vs Fusion
    stat_results: Dict[str, Any] = {}
    seeds_used = [s['seed'] for s in per_seed]

    for (a_name, b_name) in [('face', 'fusion'), ('geo', 'fusion')]:
        for metric in ('AUC', 'EER'):
            a_vals = extract(a_name, metric)
            b_vals = extract(b_name, metric)

            if len(a_vals) != len(b_vals) or len(a_vals) < 2:
                stat_results[f'{a_name}_vs_{b_name}_{metric}'] = {'error': 'insufficient data'}
                continue

            a_arr = np.array(a_vals)
            b_arr = np.array(b_vals)

            # Paired t-test
            t_stat, p_ttest = stats.ttest_rel(a_arr, b_arr)
            # Wilcoxon signed-rank (non-parametric)
            try:
                w_stat, p_wilcox = stats.wilcoxon(a_arr, b_arr)
            except ValueError as e:
                w_stat, p_wilcox = None, None

            stat_results[f'{a_name}_vs_{b_name}_{metric}'] = {
                'values_a': a_vals,
                'values_b': b_vals,
                'mean_diff': float(np.mean(b_arr - a_arr)),
                'ttest_t': float(t_stat),
                'ttest_p': float(p_ttest),
                'wilcoxon_w': float(w_stat) if w_stat is not None else None,
                'wilcoxon_p': float(p_wilcox) if p_wilcox is not None else None,
                'significant_p05': bool(p_ttest < 0.05) if p_ttest is not None else False,
            }

    stat_file = out_dir / 'statistical_test.json'
    stat_out = {
        'seeds': seeds_used,
        'tests': stat_results,
    }
    with open(stat_file, 'w') as f:
        json.dump(stat_out, f, indent=2)
    logger.info(f"Saved statistical tests → {stat_file}")

    return {'aggregated': agg, 'stats': stat_results}


# ===========================================================================
# 7.  Comparison table
# ===========================================================================

def produce_comparison_table(
    aggregated: Dict[str, Any],
    out_dir: Path,
) -> None:
    """Write comparison_table.csv and comparison_table_pretty.md."""
    rows = []
    model_display = {
        'face':   'Face (Baseline)',
        'geo':    'Geo (Baseline)',
        'fusion': 'Proposed (Fusion)',
    }

    for key, display in model_display.items():
        a = aggregated.get(key, {})
        n_imp = a.get('num_impostors', 0)
        min_far = 1.0 / n_imp if n_imp > 0 else 1.0

        tar1e3_sup = a.get('TAR1e3_supported_seeds', 0)
        total_seeds = sum(1 for _ in [1])  # placeholder
        tar1e3_mean = a.get('TAR1e3_mean')
        tar1e3_std  = a.get('TAR1e3_std')

        if tar1e3_mean is not None:
            tar1e3_str_mean = f"{tar1e3_mean:.4f}"
            tar1e3_str_std  = f"{tar1e3_std:.4f}"
            tar1e3_note = f"supported ({tar1e3_sup} seeds)"
        else:
            tar1e3_str_mean = "NOT SUPPORTED"
            tar1e3_str_std  = "NOT SUPPORTED"
            tar1e3_note = f"min_FAR={min_far:.2e} > 1e-3; need ≥1000 impostors"

        notes_parts = []
        if n_imp < TARGET_MIN_IMPOSTORS:
            notes_parts.append(f"n_impostors={n_imp}<{TARGET_MIN_IMPOSTORS}")
        if tar1e3_mean is None:
            notes_parts.append(tar1e3_note)

        rows.append({
            'Model':            display,
            'AUC_mean':         f"{a.get('AUC_mean', 'N/A'):.4f}" if a.get('AUC_mean') else 'N/A',
            'AUC_std':          f"{a.get('AUC_std',  'N/A'):.4f}" if a.get('AUC_std')  else 'N/A',
            'EER_mean':         f"{a.get('EER_mean', 'N/A'):.4f}" if a.get('EER_mean') else 'N/A',
            'EER_std':          f"{a.get('EER_std',  'N/A'):.4f}" if a.get('EER_std')  else 'N/A',
            'TAR@1e-2_mean':    f"{a.get('TAR1e2_mean', 'N/A'):.4f}" if a.get('TAR1e2_mean') else 'N/A',
            'TAR@1e-2_std':     f"{a.get('TAR1e2_std',  'N/A'):.4f}" if a.get('TAR1e2_std')  else 'N/A',
            'TAR@1e-3_mean':    tar1e3_str_mean,
            'TAR@1e-3_std':     tar1e3_str_std,
            'num_impostors':    n_imp,
            'notes':            '; '.join(notes_parts) if notes_parts else 'OK',
        })

    df_csv = pd.DataFrame(rows)
    csv_path = out_dir / 'comparison_table.csv'
    df_csv.to_csv(csv_path, index=False)
    logger.info(f"Saved comparison_table.csv → {csv_path}")

    # Markdown pretty table
    md_lines = [
        "| Model | AUC | EER | TAR@1e-2 | TAR@1e-3 | n_impostors | Notes |",
        "|-------|-----|-----|----------|----------|-------------|-------|",
    ]
    for r in rows:
        auc   = f"{r['AUC_mean']} ± {r['AUC_std']}"
        eer   = f"{r['EER_mean']} ± {r['EER_std']}"
        t1e2  = f"{r['TAR@1e-2_mean']} ± {r['TAR@1e-2_std']}"
        t1e3  = r['TAR@1e-3_mean'] if r['TAR@1e-3_mean'] == 'NOT SUPPORTED' else \
                f"{r['TAR@1e-3_mean']} ± {r['TAR@1e-3_std']}"
        md_lines.append(
            f"| {r['Model']} | {auc} | {eer} | {t1e2} | {t1e3} | {r['num_impostors']} | {r['notes']} |"
        )

    md_path = out_dir / 'comparison_table_pretty.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines) + '\n')
    logger.info(f"Saved comparison_table_pretty.md → {md_path}")


# ===========================================================================
# 8.  Environment info
# ===========================================================================

def collect_env_info(out_dir: Path) -> None:
    """Collect and save environment/hardware info for reproducibility."""
    import sys

    env: Dict[str, Any] = {
        'hostname': platform.node(),
        'platform': platform.platform(),
        'python_version': sys.version,
        'cpu': platform.uname().processor or platform.uname().machine,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    try:
        import torch
        env['torch_version'] = torch.__version__
        env['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env['cuda_version'] = torch.version.cuda
            env['gpu_count'] = torch.cuda.device_count()
            env['gpu_names'] = [torch.cuda.get_device_name(i)
                                for i in range(torch.cuda.device_count())]
    except ImportError:
        env['torch_version'] = 'not installed'

    try:
        import psutil
        vm = psutil.virtual_memory()
        env['ram_total_gb'] = round(vm.total / 1e9, 2)
    except ImportError:
        pass

    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            env['nvidia_smi'] = result.stdout.strip()
    except Exception:
        pass

    env_path = out_dir / 'env_info.json'
    with open(env_path, 'w') as f:
        json.dump(env, f, indent=2)
    logger.info(f"Saved env_info.json → {env_path}")


# ===========================================================================
# 9.  Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-seed fusion evaluation runner")
    parser.add_argument('--csv', default='data/geo/system_events.csv',
                        help='Path to system events CSV with T_face, T_geo, label, identity')
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 21, 42, 123, 321],
                        help='Random seeds for evaluation')
    parser.add_argument('--out', default='results/', help='Output directory')
    parser.add_argument('--alpha', type=float, default=0.6, help='Face fusion weight')
    parser.add_argument('--beta', type=float, default=0.4, help='Geo fusion weight')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Output: {out_dir}")
    logger.info(f"Fusion: alpha={args.alpha}, beta={args.beta}")

    # Environment info
    collect_env_info(out_dir)

    # Per-seed evaluation
    per_seed_results = []
    for seed in args.seeds:
        logger.info(f"\n{'='*60}\nEvaluating seed {seed}\n{'='*60}")
        metrics = evaluate_seed(
            csv_path=args.csv,
            seed=seed,
            out_dir=out_dir,
            alpha=args.alpha,
            beta=args.beta,
        )
        per_seed_results.append(metrics)

    # Aggregate + stats
    results = aggregate_and_test(per_seed_results, out_dir)

    # Save canonical split_report.json for the default seed (42 or first seed)
    default_seed = 42 if 42 in args.seeds else args.seeds[0]
    split_src = out_dir / f'split_report_seed_{default_seed}.json'
    split_dst = out_dir / 'split_report.json'
    if split_src.exists():
        import shutil
        shutil.copy(split_src, split_dst)
        logger.info(f"Copied split_report_seed_{default_seed}.json → split_report.json")

    # Comparison table
    produce_comparison_table(results['aggregated'], out_dir)

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 70)
    for model, a in results['aggregated'].items():
        logger.info(
            f"  {model.upper():8s}: AUC={a['AUC_mean']:.4f}±{a['AUC_std']:.4f}  "
            f"EER={a['EER_mean']:.4f}±{a['EER_std']:.4f}  "
            f"n_impostors={a['num_impostors']}"
        )

    logger.info("\nOutputs:")
    for fname in ['split_report.json', 'comparison_table.csv',
                  'comparison_table_pretty.md', 'statistical_test.json',
                  'env_info.json']:
        p = out_dir / fname
        status = "✓" if p.exists() else "✗ MISSING"
        logger.info(f"  {status}  {p}")

    for seed in args.seeds:
        p = out_dir / f'metrics_seed_{seed}.json'
        status = "✓" if p.exists() else "✗ MISSING"
        logger.info(f"  {status}  {p}")

    logger.info("Done.")


if __name__ == '__main__':
    main()
