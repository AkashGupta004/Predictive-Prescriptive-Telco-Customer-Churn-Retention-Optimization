from pathlib import Path

import pandas as pd

from src.preprocessing import prepare_features, build_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = (
    PROJECT_ROOT
    / "data"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


def test_telco_preprocessing_produces_46_features() -> None:
    data = pd.read_csv(DATASET)
    features, target = prepare_features(data)
    preprocessor = build_preprocessor(features)
    transformed = preprocessor.fit_transform(features)

    assert transformed.shape == (7043, 46)
    assert target.value_counts().to_dict() == {0: 5174, 1: 1869}
    assert "customerID" not in features
    assert features["TotalCharges"].isna().sum() == 11
