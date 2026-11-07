"""Probability calibration helpers."""

from __future__ import annotations

from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold


CALIBRATION_METHODS = ("isotonic", "sigmoid")


def calibrate_classifier(
    estimator: Any,
    features: Any,
    target: Any,
    method: str,
    random_state: int = 42,
    n_splits: int = 3,
) -> CalibratedClassifierCV:
    """Fit isotonic or Platt/sigmoid calibration using stratified CV."""
    if method not in CALIBRATION_METHODS:
        raise ValueError(
            f"method must be one of {CALIBRATION_METHODS}; received {method!r}."
        )
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    calibrated = CalibratedClassifierCV(
        estimator=estimator,
        method=method,
        cv=StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        ),
        ensemble=True,
    )
    calibrated.fit(features, target)
    return calibrated
