"""Classification metrics and validation-based cost-sensitive thresholds."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


FALSE_NEGATIVE_COST = 5.0
FALSE_POSITIVE_COST = 1.0


def find_cost_sensitive_threshold(
    target: np.ndarray,
    probabilities: np.ndarray,
    false_negative_cost: float = FALSE_NEGATIVE_COST,
    false_positive_cost: float = FALSE_POSITIVE_COST,
) -> tuple[float, float]:
    """Find the validation threshold minimizing total weighted error cost."""
    target = np.asarray(target, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if target.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("target and probabilities must be one-dimensional.")
    if target.size == 0 or target.size != probabilities.size:
        raise ValueError("target and probabilities must have equal non-zero length.")
    if not np.isin(target, [0, 1]).all():
        raise ValueError("target must contain only binary values 0 and 1.")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0) | (probabilities > 1)
    ).any():
        raise ValueError("probabilities must be finite values between 0 and 1.")
    if false_negative_cost <= 0 or false_positive_cost <= 0:
        raise ValueError("Misclassification costs must be positive.")

    thresholds = np.unique(np.concatenate(([0.0, 1.0], probabilities)))
    best_threshold = float(thresholds[0])
    best_cost = float("inf")
    for threshold in thresholds:
        predictions = probabilities >= threshold
        false_negatives = np.count_nonzero((target == 1) & ~predictions)
        false_positives = np.count_nonzero((target == 0) & predictions)
        cost = (
            false_negative_cost * false_negatives
            + false_positive_cost * false_positives
        )
        if cost < best_cost:
            best_cost = float(cost)
            best_threshold = float(threshold)
    return best_threshold, best_cost


def evaluate_predictions(
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate discrimination, calibration, and cost metrics."""
    target = np.asarray(target, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if target.size != probabilities.size or target.size == 0:
        raise ValueError("target and probabilities must have equal non-zero length.")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "average_precision": float(average_precision_score(target, probabilities)),
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(target, probabilities)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "weighted_error_cost": float(
            FALSE_NEGATIVE_COST * fn + FALSE_POSITIVE_COST * fp
        ),
    }
