"""Data cleaning, feature engineering, and leakage-safe preprocessing."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Churn"
IDENTIFIER_COLUMN = "customerID"
ENGINEERED_COLUMN = "is_new_customer"


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return cleaned predictors and a binary churn target."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain a {TARGET_COLUMN!r} column.")

    frame = data.copy()
    target = frame.pop(TARGET_COLUMN).astype("string").str.strip()
    target_map = {"No": 0, "Yes": 1}
    unknown_targets = set(target.dropna().unique()) - set(target_map)
    if unknown_targets:
        raise ValueError(f"Unexpected churn labels: {sorted(unknown_targets)}")

    if IDENTIFIER_COLUMN in frame.columns:
        frame = frame.drop(columns=IDENTIFIER_COLUMN)

    if "TotalCharges" in frame.columns:
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")

    if "tenure" in frame.columns:
        tenure = pd.to_numeric(frame["tenure"], errors="coerce")
        frame[ENGINEERED_COLUMN] = (tenure <= 12).astype("int8")

    if target.isna().any():
        raise ValueError("Churn target contains missing values.")

    return frame, target.map(target_map).astype("int8")


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create numeric and categorical transformers for the given columns."""
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return transformed feature names after the preprocessor has been fitted."""
    try:
        return preprocessor.get_feature_names_out().tolist()
    except (AttributeError, ValueError) as error:
        raise RuntimeError(
            "The preprocessor must be fitted before requesting feature names."
        ) from error


def as_float_array(values: Any) -> Any:
    """Convert transformed values to a dense float array for downstream models."""
    return values.astype("float64", copy=False)
