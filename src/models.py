"""Classifier definitions, leakage-safe pipelines, and Optuna tuning."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.preprocessing import build_preprocessor


def create_classifiers(random_state: int = 42) -> dict[str, Any]:
    """Create seven diverse, probability-capable classifiers."""
    return {
        "logistic_regression": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2_000,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            l2_regularization=1.0,
            random_state=random_state,
        ),
        "svc": SVC(
            C=1.0,
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
        "knn": KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1),
    }


def build_model_pipeline(
    classifier: Any,
    features: pd.DataFrame,
    random_state: int = 42,
) -> Pipeline:
    """Build a training-only preprocessing and SMOTE pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            ("smote", SMOTE(random_state=random_state)),
            ("classifier", classifier),
        ]
    )


def _search_space(trial: Any, model_name: str) -> dict[str, Any]:
    """Return model-specific Optuna parameters using pipeline parameter names."""
    if model_name == "logistic_regression":
        return {
            "classifier__C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        }
    if model_name in {"random_forest", "extra_trees"}:
        return {
            "classifier__n_estimators": trial.suggest_int(
                "n_estimators", 150, 500, step=50
            ),
            "classifier__max_depth": trial.suggest_int("max_depth", 4, 24),
            "classifier__min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 8
            ),
        }
    if model_name in {"gradient_boosting", "hist_gradient_boosting"}:
        parameters: dict[str, Any] = {
            "classifier__learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
        }
        if model_name == "gradient_boosting":
            parameters["classifier__n_estimators"] = trial.suggest_int(
                "n_estimators", 75, 300, step=25
            )
            parameters["classifier__max_depth"] = trial.suggest_int(
                "max_depth", 1, 4
            )
        else:
            parameters["classifier__max_iter"] = trial.suggest_int(
                "max_iter", 75, 300, step=25
            )
            parameters["classifier__l2_regularization"] = trial.suggest_float(
                "l2_regularization", 1e-5, 10.0, log=True
            )
        return parameters
    if model_name == "svc":
        return {
            "classifier__C": trial.suggest_float("C", 1e-2, 100.0, log=True),
            "classifier__gamma": trial.suggest_categorical(
                "gamma", ["scale", "auto"]
            ),
        }
    if model_name == "knn":
        return {
            "classifier__n_neighbors": trial.suggest_int(
                "n_neighbors", 5, 35, step=2
            ),
            "classifier__weights": trial.suggest_categorical(
                "weights", ["uniform", "distance"]
            ),
        }
    raise ValueError(f"Unsupported classifier for tuning: {model_name}")


def tune_classifier(
    model_name: str,
    features: pd.DataFrame,
    target: pd.Series,
    n_trials: int = 20,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """Tune one classifier by stratified cross-validated average precision.

    Returns the fitted best pipeline and its Optuna parameter dictionary.
    """
    try:
        import optuna
    except ImportError as error:
        raise ImportError(
            "Optuna is required for tuning. Install the project requirements."
        ) from error

    classifiers = create_classifiers(random_state)
    if model_name not in classifiers:
        raise ValueError(f"Unknown classifier: {model_name}")
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    def objective(trial: Any) -> float:
        pipeline = build_model_pipeline(
            classifiers[model_name], features, random_state
        )
        pipeline.set_params(**_search_space(trial, model_name))
        scores = cross_val_score(
            pipeline,
            features,
            target,
            scoring="average_precision",
            cv=cv,
            n_jobs=1,
            error_score="raise",
        )
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials)

    best_pipeline = build_model_pipeline(
        classifiers[model_name], features, random_state
    )
    best_pipeline.set_params(
        **{
            f"classifier__{name}": value
            for name, value in study.best_params.items()
        }
    )
    best_pipeline.fit(features, target)
    return best_pipeline, dict(study.best_params)
