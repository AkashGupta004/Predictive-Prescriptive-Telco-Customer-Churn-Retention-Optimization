"""SHAP explainers for tree pipelines and arbitrary probability models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def explain_with_shap(
    model: Any,
    features: pd.DataFrame,
    output_path: str | Path,
    sample_size: int = 100,
    background_size: int = 40,
    random_state: int = 42,
) -> Path:
    """Create a SHAP summary plot, using TreeExplainer or KernelExplainer."""
    try:
        import shap
    except ImportError as error:
        raise ImportError(
            "SHAP is required for explanations. Install the project requirements."
        ) from error

    import matplotlib.pyplot as plt

    if features.empty:
        raise ValueError("Cannot explain an empty feature dataset.")
    if sample_size < 1 or background_size < 1:
        raise ValueError("sample_size and background_size must be positive.")

    rng = np.random.default_rng(random_state)
    sample_count = min(sample_size, len(features))
    background_count = min(background_size, len(features))
    sample = features.iloc[
        rng.choice(len(features), size=sample_count, replace=False)
    ]
    background = features.iloc[
        rng.choice(len(features), size=background_count, replace=False)
    ]

    pipeline_steps = getattr(model, "named_steps", {})
    preprocessor = pipeline_steps.get("preprocessor")
    classifier = pipeline_steps.get("classifier")
    tree_estimators = (
        "RandomForestClassifier",
        "ExtraTreesClassifier",
        "GradientBoostingClassifier",
        "HistGradientBoostingClassifier",
    )

    if preprocessor is not None and classifier is not None:
        transformed_sample = preprocessor.transform(sample)
        transformed_background = preprocessor.transform(background)
        feature_names = preprocessor.get_feature_names_out()
        if classifier.__class__.__name__ in tree_estimators:
            explainer = shap.TreeExplainer(classifier, transformed_background)
            values = explainer.shap_values(transformed_sample)
            plot_data = transformed_sample
        else:
            transformed_model = lambda values: classifier.predict_proba(values)[:, 1]
            explainer = shap.KernelExplainer(
                transformed_model,
                transformed_background,
                link="logit",
            )
            values = explainer.shap_values(transformed_sample, nsamples=100)
            plot_data = transformed_sample
    else:
        explainer = shap.KernelExplainer(
            lambda values: model.predict_proba(
                pd.DataFrame(values, columns=features.columns)
            )[:, 1],
            background,
            link="logit",
        )
        values = explainer.shap_values(sample, nsamples=100)
        feature_names = features.columns.to_numpy()
        plot_data = sample

    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(
        values,
        plot_data,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close()
    return destination
