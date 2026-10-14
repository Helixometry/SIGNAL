"""Evaluation metrics for attribution and open-set detection"""

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from typing import Dict


def compute_attribution_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute attribution accuracy metrics"""
    return {
        "accuracy": accuracy_score(targets, predictions),
        "f1": f1_score(targets, predictions, average="weighted"),
    }


def compute_open_set_metrics(is_unknown: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute open-set detection metrics"""
    return {
        "auroc": roc_auc_score(targets, is_unknown),
    }
