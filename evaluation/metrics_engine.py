import numpy as np
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_auc_score


def compute_metrics(labels: np.ndarray, scores: np.ndarray):
    """Compute a strict set of metrics using T_final (scores).

    Returns a dict with ROC, AUC, EER, FAR@1%, FAR@0.1%, FRR, accuracy, precision, recall
    """
    assert labels.shape[0] == scores.shape[0]
    assert not np.isnan(scores).any(), "scores contain NaN"

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # EER
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])

    # FAR at specific operating points (using thresholds sweep)
    def far_at_rate(target_far):
        # FAR = FPR; find threshold where FPR <= target_far and take TPR at that point
        idx = np.where(fpr <= target_far)[0]
        if idx.size == 0:
            return 0.0
        i = idx[-1]
        return float(fpr[i])

    far_1pct = far_at_rate(0.01)
    far_01pct = far_at_rate(0.001)

    # Use EER threshold for classification metrics
    preds = (scores >= eer_threshold).astype(int)

    accuracy = float(accuracy_score(labels, preds))
    precision = float(precision_score(labels, preds, zero_division=0))
    recall = float(recall_score(labels, preds, zero_division=0))
    fr = float(np.mean((labels == 1) & (preds == 0)))  # simple FRR

    return {
        'roc_auc': float(roc_auc),
        'eer': eer,
        'eer_threshold': eer_threshold,
        'far_1pct': far_1pct,
        'far_01pct': far_01pct,
        'frr': fr,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
    }
