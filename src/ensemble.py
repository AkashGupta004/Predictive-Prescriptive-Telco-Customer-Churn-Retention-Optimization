"""Soft-voting ensemble implementation."""

from typing import Any

import pandas as pd
from sklearn.ensemble import VotingClassifier

from src.models import build_model_pipeline, create_classifiers


def create_soft_voting_ensemble(
    features: pd.DataFrame,
    random_state: int = 42,
) -> Any:
    """Create an equal-weight soft-voting classifier from all seven models."""
    estimators = [
        (
            name,
            build_model_pipeline(classifier, features, random_state),
        )
        for name, classifier in create_classifiers(random_state).items()
    ]
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)


__all__ = ["create_soft_voting_ensemble"]
