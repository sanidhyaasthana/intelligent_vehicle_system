"""
Evaluation Metrics
==================

Comprehensive metrics for face verification, geofence evaluation, and system-level
performance assessment. All metrics are designed for research paper reporting.

Metrics Implemented:
- Accuracy, FAR (False Acceptance Rate), FRR (False Rejection Rate)
- TAR (True Acceptance Rate), EER (Equal Error Rate)
- ROC curve and AUC
- Confusion matrices
- Threshold optimization

Design Choices:
- All functions work with NumPy arrays for compatibility
- Vectorized implementations for efficiency
- Support for both binary and multi-class settings
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Union, Any
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix as sklearn_confusion_matrix,
    precision_recall_curve,
    f1_score,
    accuracy_score
)
import matplotlib.pyplot as plt
from pathlib import Path


def compute_accuracy(y_true: np.ndarray,
                     y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Accuracy as a float in [0, 1].

    Example:
        >>> y_true = np.array([1, 0, 1, 1, 0])
        >>> y_pred = np.array([1, 0, 0, 1, 0])
        >>> compute_accuracy(y_true, y_pred)
        0.8
    """
    return float(np.mean(y_true == y_pred))


def compute_far(y_true: np.ndarray,
                y_pred: np.ndarray,
                positive_label: int = 1) -> float:
    """
    Compute False Acceptance Rate (FAR).

    FAR = FP / (FP + TN) = False Positives / Total Negatives

    In security context: Rate of unauthorized users accepted.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        positive_label: Label considered as "authorized/legitimate".

    Returns:
        FAR as a float in [0, 1].

    Example:
        >>> y_true = np.array([0, 0, 0, 1, 1])  # 3 negatives, 2 positives
        >>> y_pred = np.array([1, 0, 0, 1, 1])  # 1 false positive
        >>> compute_far(y_true, y_pred)
        0.333...
    """
    # Negative samples (unauthorized)
    negatives = y_true != positive_label
    num_negatives = np.sum(negatives)

    if num_negatives == 0:
        return 0.0

    # False positives: predicted positive but actually negative
    false_positives = np.sum((y_pred == positive_label) & negatives)

    return float(false_positives / num_negatives)


def compute_frr(y_true: np.ndarray,
                y_pred: np.ndarray,
                positive_label: int = 1) -> float:
    """
    Compute False Rejection Rate (FRR).

    FRR = FN / (FN + TP) = False Negatives / Total Positives

    In security context: Rate of legitimate users rejected.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        positive_label: Label considered as "authorized/legitimate".

    Returns:
        FRR as a float in [0, 1].

    Example:
        >>> y_true = np.array([1, 1, 1, 0, 0])  # 3 positives, 2 negatives
        >>> y_pred = np.array([0, 1, 1, 0, 0])  # 1 false negative
        >>> compute_frr(y_true, y_pred)
        0.333...
    """
    # Positive samples (legitimate)
    positives = y_true == positive_label
    num_positives = np.sum(positives)

    if num_positives == 0:
        return 0.0

    # False negatives: predicted negative but actually positive
    false_negatives = np.sum((y_pred != positive_label) & positives)

    return float(false_negatives / num_positives)


def compute_tar(y_true: np.ndarray,
                y_pred: np.ndarray,
                positive_label: int = 1) -> float:
    """
    Compute True Acceptance Rate (TAR).

    TAR = TP / (TP + FN) = 1 - FRR

    Also known as True Positive Rate (TPR) or Recall.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        positive_label: Label considered as "authorized/legitimate".

    Returns:
        TAR as a float in [0, 1].
    """
    return 1.0 - compute_frr(y_true, y_pred, positive_label)


def compute_eer(y_true: np.ndarray,
                scores: np.ndarray,
                positive_label: int = 1) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER).

    EER is the point where FAR = FRR on the ROC curve.
    Lower EER indicates better verification performance.

    Args:
        y_true: Ground truth labels.
        scores: Continuous prediction scores (higher = more likely positive).
        positive_label: Label considered as "authorized/legitimate".

    Returns:
        Tuple of (EER, threshold at EER).

    Example:
        >>> y_true = np.array([1, 1, 0, 0])
        >>> scores = np.array([0.9, 0.7, 0.3, 0.2])
        >>> eer, thresh = compute_eer(y_true, scores)
        >>> print(f"EER: {eer:.4f} at threshold {thresh:.4f}")
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=positive_label)

    # FRR = 1 - TPR
    fnr = 1 - tpr

    # Find intersection point (EER)
    # The point where FPR = FNR (equivalently, FAR = FRR)
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2

    return float(eer), float(thresholds[eer_index])


def compute_roc(y_true: np.ndarray,
                scores: np.ndarray,
                num_thresholds: int = 100,
                positive_label: int = 1) -> Dict[str, np.ndarray]:
    """
    Compute ROC curve data.

    Args:
        y_true: Ground truth labels.
        scores: Continuous prediction scores.
        num_thresholds: Number of threshold points.
        positive_label: Label considered as positive.

    Returns:
        Dictionary with 'fpr', 'tpr', 'thresholds', 'far', 'frr'.

    Note:
        In verification context: FAR = FPR, FRR = FNR = 1 - TPR
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=positive_label)

    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'far': fpr,  # FAR = FPR in binary classification
        'frr': 1 - tpr,  # FRR = FNR = 1 - TPR
        'tar': tpr  # TAR = TPR
    }


def compute_auc(y_true: np.ndarray,
                scores: np.ndarray,
                positive_label: int = 1) -> float:
    """
    Compute Area Under ROC Curve (AUC).

    Args:
        y_true: Ground truth labels.
        scores: Continuous prediction scores.
        positive_label: Label considered as positive.

    Returns:
        AUC as a float in [0, 1].
    """
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=positive_label)
    return float(auc(fpr, tpr))


def compute_confusion_matrix(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             labels: Optional[List] = None,
                             normalize: Optional[str] = None) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: List of labels to include.
        normalize: Normalization mode ('true', 'pred', 'all', or None).

    Returns:
        Confusion matrix as 2D numpy array.

    Layout:
        [[TN, FP],
         [FN, TP]]

    For binary: rows are actual classes, columns are predicted classes.
    """
    cm = sklearn_confusion_matrix(y_true, y_pred, labels=labels)

    if normalize == 'true':
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    elif normalize == 'pred':
        cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)
    elif normalize == 'all':
        cm = cm.astype('float') / cm.sum()

    return cm


def compute_tar_at_far(y_true: np.ndarray,
                       scores: np.ndarray,
                       target_far: float,
                       positive_label: int = 1) -> Tuple[float, float]:
    """
    Compute TAR at a specific FAR level.

    Common metric for face verification: TAR@FAR=0.1%, TAR@FAR=0.01%, etc.

    Args:
        y_true: Ground truth labels.
        scores: Continuous prediction scores.
        target_far: Target FAR level.
        positive_label: Label considered as positive.

    Returns:
        Tuple of (TAR at target FAR, corresponding threshold).
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=positive_label)

    # Find the largest threshold where FAR <= target_far
    valid_indices = np.where(fpr <= target_far)[0]

    if len(valid_indices) == 0:
        return 0.0, float(thresholds[0])

    best_idx = valid_indices[-1]
    return float(tpr[best_idx]), float(thresholds[best_idx])


def compute_verification_metrics(y_true: np.ndarray,
                                 scores: np.ndarray,
                                 threshold: Optional[float] = None,
                                 positive_label: int = 1) -> Dict[str, float]:
    """
    Compute comprehensive verification metrics.

    Args:
        y_true: Ground truth labels (1 for same identity, 0 for different).
        scores: Similarity scores between pairs.
        threshold: Decision threshold. If None, uses EER threshold.
        positive_label: Label for genuine pairs.

    Returns:
        Dictionary containing all relevant metrics.
    """
    # Compute EER and use its threshold if not provided
    eer, eer_thresh = compute_eer(y_true, scores, positive_label)

    if threshold is None:
        threshold = eer_thresh

    # Binary predictions at threshold
    y_pred = (scores >= threshold).astype(int)
    if positive_label != 1:
        y_pred = np.where(y_pred == 1, positive_label, 1 - positive_label)

    # Compute all metrics
    metrics = {
        'accuracy': compute_accuracy(y_true, y_pred),
        'far': compute_far(y_true, y_pred, positive_label),
        'frr': compute_frr(y_true, y_pred, positive_label),
        'tar': compute_tar(y_true, y_pred, positive_label),
        'eer': eer,
        'eer_threshold': eer_thresh,
        'auc': compute_auc(y_true, scores, positive_label),
        'threshold_used': threshold
    }

    # TAR at various FAR levels (common in face verification)
    for target_far in [0.1, 0.01, 0.001, 0.0001]:
        tar, _ = compute_tar_at_far(y_true, scores, target_far, positive_label)
        metrics[f'tar_at_far_{target_far}'] = tar

    return metrics


def compute_face_metrics(embeddings: np.ndarray,
                         labels: np.ndarray) -> Dict[str, float]:
    """
    Compute verification metrics for a set of embeddings and identity labels.

    Converts embeddings into pairwise similarity scores (cosine similarity)
    and computes verification metrics such as EER and TAR@FAR.

    Args:
        embeddings: Array of shape (N, D) with embedding vectors.
        labels: Array of shape (N,) with integer identity labels.

    Returns:
        Dictionary of verification metrics (see `compute_verification_metrics`).
    """
    # Ensure numpy arrays
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D array of shape (N, D)")

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs_norm = embeddings / norms

    # Pairwise cosine similarities (upper triangular, excluding diagonal)
    sims = embs_norm @ embs_norm.T
    idx = np.triu_indices(sims.shape[0], k=1)
    scores = sims[idx]

    # Ground-truth labels for pairs (1 = same identity, 0 = different)
    y_true = (labels[idx[0]] == labels[idx[1]]).astype(int)

    metrics = compute_verification_metrics(y_true, scores)

    # Convenience alias used in training logging
    metrics['tar_at_far'] = metrics.get('tar_at_far_0.1', 0.0)

    return metrics


def compute_geo_metrics(predictions: np.ndarray,
                        labels: np.ndarray,
                        threshold: Optional[float] = None,
                        positive_label: int = 1) -> Dict[str, float]:
    """
    Compute evaluation metrics for geofence trust model.

    Args:
        predictions: Trust scores from geofence model (shape (N,) or (N, 1)).
        labels: Ground truth labels (1 for inside/trusted, 0 for outside/untrusted).
        threshold: Decision threshold. If None, uses EER threshold.
        positive_label: Label for positive class (default: 1 for inside geofence).

    Returns:
        Dictionary containing geofence evaluation metrics:
        - accuracy: Overall correctness
        - far: False Acceptance Rate (untrusted locations accepted)
        - frr: False Rejection Rate (trusted locations rejected)
        - eer: Equal Error Rate
        - auc: Area Under ROC Curve
        - threshold_used: Decision threshold
    """
    # Ensure numpy arrays and flatten if needed
    predictions = np.asarray(predictions).flatten()
    labels = np.asarray(labels).flatten()

    if predictions.shape != labels.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs labels {labels.shape}")

    # Compute EER and use its threshold if not provided
    eer, eer_thresh = compute_eer(labels, predictions, positive_label)

    if threshold is None:
        threshold = eer_thresh

    # Binary predictions at threshold
    y_pred = (predictions >= threshold).astype(int)
    if positive_label != 1:
        y_pred = np.where(y_pred == 1, positive_label, 1 - positive_label)

    # Compute metrics
    metrics = {
        'accuracy': compute_accuracy(labels, y_pred),
        'far': compute_far(labels, y_pred, positive_label),
        'frr': compute_frr(labels, y_pred, positive_label),
        'eer': eer,
        'eer_threshold': eer_thresh,
        'auc': compute_auc(labels, predictions, positive_label),
        'threshold_used': threshold,
    }

    return metrics


def compute_system_metrics(y_true: np.ndarray,
                           decisions: np.ndarray,
                           attack_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute system-level authorization metrics.

    Args:
        y_true: Ground truth (1 for legitimate, 0 for attack).
        decisions: System decisions (encoded as integers).
                   0: BLOCK, 1: WARN, 2: ALLOW (or binary)
        attack_labels: Optional attack type labels for breakdown.

    Returns:
        Dictionary containing system metrics.
    """
    # For binary decisions
    if np.max(decisions) <= 1:
        y_pred = decisions
    else:
        # Multi-class: ALLOW (2) counts as acceptance
        y_pred = (decisions == 2).astype(int)

    metrics = {
        'accuracy': compute_accuracy(y_true, y_pred),
        'far': compute_far(y_true, y_pred),
        'frr': compute_frr(y_true, y_pred),
        'attack_success_rate': compute_far(y_true, y_pred),
        'legitimate_failure_rate': compute_frr(y_true, y_pred)
    }

    # F1 score
    metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))

    return metrics


def find_optimal_threshold(y_true: np.ndarray,
                           scores: np.ndarray,
                           criterion: str = 'eer',
                           positive_label: int = 1) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal threshold based on specified criterion.

    Args:
        y_true: Ground truth labels.
        scores: Prediction scores.
        criterion: Optimization criterion ('eer', 'f1', 'balanced').
        positive_label: Positive class label.

    Returns:
        Tuple of (optimal threshold, metrics at that threshold).
    """
    thresholds = np.linspace(np.min(scores), np.max(scores), 1000)

    if criterion == 'eer':
        eer, optimal_threshold = compute_eer(y_true, scores, positive_label)
    elif criterion == 'f1':
        best_f1 = 0
        optimal_threshold = 0.5
        for thresh in thresholds:
            y_pred = (scores >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                optimal_threshold = thresh
    elif criterion == 'balanced':
        # Minimize |FAR - FRR|
        min_diff = float('inf')
        optimal_threshold = 0.5
        for thresh in thresholds:
            y_pred = (scores >= thresh).astype(int)
            far = compute_far(y_true, y_pred, positive_label)
            frr = compute_frr(y_true, y_pred, positive_label)
            diff = abs(far - frr)
            if diff < min_diff:
                min_diff = diff
                optimal_threshold = thresh
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    # Compute metrics at optimal threshold
    y_pred = (scores >= optimal_threshold).astype(int)
    metrics = {
        'accuracy': compute_accuracy(y_true, y_pred),
        'far': compute_far(y_true, y_pred, positive_label),
        'frr': compute_frr(y_true, y_pred, positive_label),
        'threshold': optimal_threshold
    }

    return optimal_threshold, metrics


def save_metrics_to_csv(metrics: Dict[str, float],
                        save_path: Union[str, Path],
                        append: bool = False) -> None:
    """
    Save metrics dictionary to CSV file.

    Args:
        metrics: Dictionary of metric names and values.
        save_path: Path to save CSV.
        append: If True, append to existing file.
    """
    import csv

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    mode = 'a' if append else 'w'
    write_header = not (append and save_path.exists())

    with open(save_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)


def plot_roc_curve(y_true: np.ndarray,
                   scores: np.ndarray,
                   label: str = 'Model',
                   save_path: Optional[Union[str, Path]] = None,
                   show: bool = False) -> plt.Figure:
    """
    Plot ROC curve with AUC.

    Args:
        y_true: Ground truth labels.
        scores: Prediction scores.
        label: Legend label for this curve.
        save_path: Path to save the plot.
        show: If True, display the plot.

    Returns:
        Matplotlib Figure object.
    """
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'{label} (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

    ax.set_xlabel('False Positive Rate (FAR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TAR)', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def save_roc_curve(embeddings: np.ndarray,
                   labels: np.ndarray,
                   save_path: Optional[Union[str, Path]] = None,
                   show: bool = False) -> None:
    """
    Compute pairwise ROC from embeddings and labels and save the ROC plot.

    Args:
        embeddings: Array of shape (N, D) with embedding vectors.
        labels: Array of shape (N,) with integer identity labels.
        save_path: Path to save the ROC figure.
        show: If True, display the plot.
    """
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D array of shape (N, D)")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs_norm = embeddings / norms

    sims = embs_norm @ embs_norm.T
    idx = np.triu_indices(sims.shape[0], k=1)
    scores = sims[idx]
    y_true = (labels[idx[0]] == labels[idx[1]]).astype(int)

    plot_roc_curve(y_true, scores, save_path=save_path, show=show)


def plot_far_frr_curve(y_true: np.ndarray,
                       scores: np.ndarray,
                       save_path: Optional[Union[str, Path]] = None,
                       show: bool = False) -> plt.Figure:
    """
    Plot FAR and FRR vs threshold curve with EER point.

    Args:
        y_true: Ground truth labels.
        scores: Prediction scores.
        save_path: Path to save the plot.
        show: If True, display the plot.

    Returns:
        Matplotlib Figure object.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1 - tpr

    # Find EER
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_thresh = thresholds[eer_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, fpr, 'b-', linewidth=2, label='FAR')
    ax.plot(thresholds, fnr, 'r-', linewidth=2, label='FRR')

    # Mark EER point
    ax.axvline(x=eer_thresh, color='g', linestyle='--', linewidth=1,
               label=f'EER = {eer:.4f}')
    ax.scatter([eer_thresh], [eer], color='g', s=100, zorder=5)

    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Error Rate', fontsize=12)
    ax.set_title('FAR and FRR vs Threshold', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_confusion_matrix(cm: np.ndarray,
                          labels: List[str] = ['Negative', 'Positive'],
                          save_path: Optional[Union[str, Path]] = None,
                          show: bool = False,
                          title: str = 'Confusion Matrix') -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix array.
        labels: Class labels.
        save_path: Path to save the plot.
        show: If True, display the plot.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels,
           yticklabels=labels,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.2f' if cm.dtype == float else 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def compare_models(results: List[Dict[str, Any]],
                   save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Create comparison bar chart for multiple models.

    Args:
        results: List of dictionaries with 'name' and metrics.
        save_path: Path to save the plot.

    Returns:
        Matplotlib Figure object.
    """
    metrics_to_plot = ['accuracy', 'far', 'frr', 'eer', 'auc']
    available_metrics = [m for m in metrics_to_plot if m in results[0]]

    fig, axes = plt.subplots(1, len(available_metrics), figsize=(4 * len(available_metrics), 5))
    if len(available_metrics) == 1:
        axes = [axes]

    names = [r['name'] for r in results]
    x = np.arange(len(names))

    for ax, metric in zip(axes, available_metrics):
        values = [r.get(metric, 0) for r in results]
        bars = ax.bar(x, values, color=plt.cm.Set2(np.arange(len(names))))
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig

def compute_metrics_binary(y_true: np.ndarray,
                          scores: np.ndarray,
                          thresholds: np.ndarray,
                          positive_label: int = 1) -> Dict[str, float]:
    """
    Compute binary classification metrics: FAR, FRR, EER, AUC.
    
    Computes biometric metrics for fusion model evaluation.
    All metrics computed on validation set (no test set leakage).
    
    Args:
        y_true: Ground truth binary labels (0/1).
        scores: Continuous scores from model (0-1 range).
        thresholds: Array of threshold values to evaluate.
        positive_label: Label considered as positive (default: 1).
    
    Returns:
        Dictionary with:
        - 'far': False Acceptance Rate
        - 'frr': False Rejection Rate  
        - 'eer': Equal Error Rate
        - 'auc': Area Under ROC Curve
        - 'tar': True Acceptance Rate
    
    Example:
        >>> y_true = np.array([1, 1, 0, 0, 1, 0])
        >>> scores = np.array([0.9, 0.7, 0.3, 0.1, 0.8, 0.2])
        >>> thresholds = np.linspace(0, 1, 100)
        >>> metrics = compute_metrics_binary(y_true, scores, thresholds)
        >>> print(f"EER: {metrics['eer']:.4f}, AUC: {metrics['auc']:.4f}")
    """
    # Compute EER and threshold
    eer, eer_threshold = compute_eer(y_true, scores, positive_label=positive_label)
    
    # Compute AUC
    auc_value = compute_auc(y_true, scores, positive_label=positive_label)
    
    # Compute FAR and FRR at EER threshold
    y_pred_eer = (scores >= eer_threshold).astype(int)
    
    # FAR = False Positives / Negatives
    negatives = y_true != positive_label
    false_positives = np.sum((y_pred_eer == 1) & negatives)
    far = false_positives / np.sum(negatives) if np.sum(negatives) > 0 else 0.0
    
    # FRR = False Negatives / Positives
    positives = y_true == positive_label
    false_negatives = np.sum((y_pred_eer == 0) & positives)
    frr = false_negatives / np.sum(positives) if np.sum(positives) > 0 else 0.0
    
    # TAR = True Acceptance Rate
    tar = 1.0 - frr
    
    return {
        'far': float(far),
        'frr': float(frr),
        'eer': float(eer),
        'auc': float(auc_value),
        'tar': float(tar),
    }