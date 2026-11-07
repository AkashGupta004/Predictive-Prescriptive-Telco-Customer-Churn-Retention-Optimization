import numpy as np

from src.evaluation import evaluate_predictions, find_cost_sensitive_threshold


def test_cost_sensitive_threshold_reduces_missed_churn_cost() -> None:
    target = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.3, 0.4, 0.9])

    threshold, cost = find_cost_sensitive_threshold(target, probabilities)

    assert 0.0 <= threshold <= 1.0
    assert cost <= 5.0


def test_evaluation_reports_brier_and_weighted_error_cost() -> None:
    target = np.array([0, 1, 1, 0])
    probabilities = np.array([0.1, 0.8, 0.3, 0.6])

    metrics = evaluate_predictions(target, probabilities, threshold=0.5)

    assert metrics["brier_score"] >= 0.0
    assert metrics["false_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["weighted_error_cost"] == 6.0
