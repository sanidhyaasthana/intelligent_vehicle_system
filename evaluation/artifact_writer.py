import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_metrics_json(metrics: dict, path: Path):
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)


def save_eer_txt(eer: float, path: Path):
    with open(path, 'w') as f:
        f.write(f"EER: {eer:.6f}\n")


def save_latency_json(latency: dict, path: Path):
    with open(path, 'w') as f:
        json.dump(latency, f, indent=2)


def save_fusion_weights(weights: dict, path: Path):
    with open(path, 'w') as f:
        json.dump(weights, f, indent=2)


def plot_roc(labels, scores, outpath: Path):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}', lw=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(outpath), dpi=300)
    plt.close()


def plot_det(labels, scores, outpath: Path):
    from sklearn.metrics import det_curve
    fpr, fnr, _ = det_curve(labels, scores, pos_label=1)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, fnr, lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.title('DET Curve')
    plt.grid(True)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(outpath), dpi=300)
    plt.close()


def save_predictions_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
