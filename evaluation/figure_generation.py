"""
Figure Generation for Publication-Grade Evaluation.

Generates:
- ROC curves (single-seed and multi-seed overlay)
- DET curves
- Score distribution histograms
- Confusion matrix heatmaps

All figures are saved at 1200 DPI for publication quality.
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    save_path: Path,
    title: str = "ROC Curve",
    label: str = "",
):
    """Plot a single ROC curve."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    lbl = label if label else f'AUC = {roc_auc:.4f}'
    ax.plot(fpr, tpr, label=lbl, lw=2)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=1200)
    plt.close(fig)
    logger.info(f"Saved ROC curve to {save_path}")


def plot_roc_multi_seed(
    per_seed_data: List[Dict],
    seeds: List[int],
    save_path: Path,
    title: str = "ROC Curve — Multi-Seed Overlay",
):
    """Plot multi-seed ROC overlay."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    for data, seed in zip(per_seed_data, seeds):
        fpr = np.array(data['fpr'])
        tpr = np.array(data['tpr'])
        auc_val = data['roc_auc']
        ax.plot(fpr, tpr, label=f'Seed {seed} (AUC={auc_val:.4f})', lw=1.5)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=1200)
    plt.close(fig)
    logger.info(f"Saved multi-seed ROC to {save_path}")


def plot_det(
    fpr: np.ndarray,
    tpr: np.ndarray,
    save_path: Path,
    title: str = "DET Curve",
    log_scale: bool = False,
):
    """Plot DET curve (FPR vs FNR)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fnr = 1.0 - np.array(tpr)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, fnr, lw=2, color='darkblue')
    ax.set_xlabel('False Positive Rate (FAR)', fontsize=12)
    ax.set_ylabel('False Negative Rate (FRR)', fontsize=12)
    ax.set_title(title, fontsize=14)
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=1200)
    plt.close(fig)
    logger.info(f"Saved DET curve to {save_path}")


def plot_score_distribution(
    scores: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
    title: str = "Score Distribution by Class",
):
    """
    STEP 8: Plot histogram of scores for positive and negative classes.

    Saves to results/fusion/system_eval/figures/score_distribution.png
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    legitimate_scores = scores[labels == 0]
    attack_scores = scores[labels == 1]

    ax.hist(legitimate_scores, bins=50, alpha=0.6, label=f'Legitimate (n={len(legitimate_scores)})',
            color='green', density=True, edgecolor='darkgreen', linewidth=0.5)
    ax.hist(attack_scores, bins=50, alpha=0.6, label=f'Attack (n={len(attack_scores)})',
            color='red', density=True, edgecolor='darkred', linewidth=0.5)

    ax.set_xlabel('Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=1200)
    plt.close(fig)
    logger.info(f"Saved score distribution to {save_path}")


def plot_confusion_matrix(
    TP: int, FP: int, TN: int, FN: int,
    save_path: Path,
    title: str = "Confusion Matrix at EER Threshold",
    normalize: bool = True,
):
    """Plot confusion matrix as heatmap."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = np.array([[TN, FP], [FN, TP]])
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm.astype(float) / row_sums
    else:
        cm_norm = cm.astype(float)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Neg', 'Predicted Pos'])
    ax.set_yticklabels(['Actual Neg', 'Actual Pos'])
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title(title, fontsize=12)

    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = cm_norm[i, j]
            ax.text(j, i, f'{count}\n({pct:.2f})', ha='center', va='center', fontsize=11)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=1200)
    plt.close(fig)
    logger.info(f"Saved confusion matrix to {save_path}")
