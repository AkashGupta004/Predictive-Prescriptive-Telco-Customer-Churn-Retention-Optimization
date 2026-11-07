"""Run the Telco churn training, calibration, evaluation, and prescriptive retention optimization pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, brier_score_loss

from src.calibration import CALIBRATION_METHODS, calibrate_classifier
from src.ensemble import create_soft_voting_ensemble
from src.evaluation import evaluate_predictions, find_cost_sensitive_threshold
from src.models import (
    build_model_pipeline,
    create_classifiers,
    tune_classifier,
)
from src.preprocessing import IDENTIFIER_COLUMN, prepare_features
from src.segmentation import AutoencoderKMeansSegmenter
from src.retention_optimization import (
    DEFAULT_OFFERS,
    DEFAULT_SENSITIVITY_BUDGETS,
    prepare_retention_candidates,
    run_baseline_comparisons,
    run_sensitivity_analysis,
    save_optimization_outputs,
    summarize_comparisons,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"
RANDOM_STATE = 42


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the IBM Telco churn CSV.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Tune all seven base classifiers with Optuna before fitting.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Optuna trials per classifier when --tune is enabled.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate a SHAP summary plot for the selected base classifier.",
    )
    parser.add_argument(
        "--skip-segmentation",
        action="store_true",
        help="Skip the 46-to-16 autoencoder and K-Means segmentation.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run prescriptive customer retention optimization using MILP and baseline comparisons.",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=50000.0,
        help="Campaign budget in INR (₹) for retention optimization (default: 50000.0).",
    )
    parser.add_argument(
        "--horizon-years",
        type=float,
        default=5.0,
        help="Customer Lifetime Value horizon in years for value proxy calculation (default: 5.0).",
    )
    parser.add_argument(
        "--run-sensitivity",
        action="store_true",
        help="Run sensitivity analysis across standard budget grid (₹10k to ₹100k).",
    )
    parser.add_argument(
        "--allow-negative-net-benefit",
        action="store_true",
        help="Allow interventions with negative net benefit (for budget exhaustion testing).",
    )
    return parser.parse_args()


def _fit_candidates(
    features_train: pd.DataFrame,
    target_train: pd.Series,
    tune: bool,
    trials: int,
) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for name, classifier in create_classifiers(RANDOM_STATE).items():
        logging.info("Fitting %s", name)
        if tune:
            pipeline, best_params = tune_classifier(
                name,
                features_train,
                target_train,
                n_trials=trials,
                random_state=RANDOM_STATE,
            )
            logging.info("%s best parameters: %s", name, best_params)
        else:
            pipeline = build_model_pipeline(
                classifier, features_train, RANDOM_STATE
            )
            pipeline.fit(features_train, target_train)
        candidates[name] = pipeline

    logging.info("Fitting soft-voting ensemble")
    ensemble = create_soft_voting_ensemble(features_train, RANDOM_STATE)
    ensemble.fit(features_train, target_train)
    candidates["soft_voting"] = ensemble
    return candidates


def run_pipeline(
    dataset_path: Path,
    tune: bool = False,
    trials: int = 20,
    explain: bool = False,
    segment: bool = True,
    optimize: bool = False,
    budget: float = 50000.0,
    horizon_years: float = 5.0,
    run_sensitivity: bool = False,
    allow_negative_net_benefit: bool = False,
) -> dict[str, Any]:
    """Execute the predictive, descriptive, and prescriptive retention optimization pipeline."""
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if tune and trials < 1:
        raise ValueError("trials must be positive when tuning is enabled.")

    data = pd.read_csv(dataset_path)
    features, target = prepare_features(data)
    identifiers = (
        data[IDENTIFIER_COLUMN].copy()
        if IDENTIFIER_COLUMN in data.columns
        else pd.Series(data.index.astype(str), index=data.index, name="customerID")
    )

    features_train, features_holdout, target_train, target_holdout = (
        train_test_split(
            features,
            target,
            test_size=0.4,
            stratify=target,
            random_state=RANDOM_STATE,
        )
    )
    features_validation, features_test, target_validation, target_test = (
        train_test_split(
            features_holdout,
            target_holdout,
            test_size=0.5,
            stratify=target_holdout,
            random_state=RANDOM_STATE,
        )
    )

    # Check if a trained model already exists to avoid unnecessary retraining
    churn_model_path = OUTPUT_DIRECTORY / "churn_model.joblib"
    if churn_model_path.exists() and not tune:
        logging.info("Loading existing calibrated model from %s", churn_model_path)
        saved_bundle = joblib.load(churn_model_path)
        final_model = saved_bundle["model"]
        selected_name = saved_bundle.get("selected_base_model", "logistic_regression")
        calibration_method = saved_bundle.get("calibration_method", "isotonic")
        threshold = saved_bundle.get("threshold", 0.2341)
        validation_cost = 0.0
        test_probabilities = final_model.predict_proba(features_test)[:, 1]
        test_metrics = evaluate_predictions(
            target_test.to_numpy(), test_probabilities, threshold
        )
    else:
        logging.info("Fitting candidate models and calibrating...")
        candidates = _fit_candidates(features_train, target_train, tune, trials)
        validation_scores: list[dict[str, float | str]] = []
        for name, model in candidates.items():
            probabilities = model.predict_proba(features_validation)[:, 1]
            validation_scores.append(
                {
                    "model": name,
                    "average_precision": float(
                        average_precision_score(target_validation, probabilities)
                    ),
                }
            )
        comparison = pd.DataFrame(validation_scores).sort_values(
            "average_precision", ascending=False
        )
        selected_name = str(comparison.iloc[0]["model"])
        selected_model = candidates[selected_name]
        logging.info("Selected %s using validation average precision", selected_name)

        calibrated_candidates: dict[str, Any] = {}
        calibration_scores: list[tuple[str, float, Any]] = []
        for method in CALIBRATION_METHODS:
            logging.info("Calibrating selected model with %s", method)
            calibrated = calibrate_classifier(
                selected_model,
                features_train,
                target_train,
                method=method,
                random_state=RANDOM_STATE,
            )
            validation_probabilities = calibrated.predict_proba(
                features_validation
            )[:, 1]
            score = float(brier_score_loss(target_validation, validation_probabilities))
            calibrated_candidates[method] = calibrated
            calibration_scores.append((method, score, calibrated))

        calibration_method, _, final_model = min(
            calibration_scores, key=lambda candidate: candidate[1]
        )
        validation_probabilities = final_model.predict_proba(features_validation)[:, 1]
        threshold, validation_cost = find_cost_sensitive_threshold(
            target_validation.to_numpy(), validation_probabilities
        )
        test_probabilities = final_model.predict_proba(features_test)[:, 1]
        test_metrics = evaluate_predictions(
            target_test.to_numpy(), test_probabilities, threshold
        )

        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(OUTPUT_DIRECTORY / "model_comparison.csv", index=False)
        pd.DataFrame(
            [
                {"method": method, "validation_brier_score": score}
                for method, score, _ in calibration_scores
            ]
        ).to_csv(OUTPUT_DIRECTORY / "calibration_comparison.csv", index=False)
        pd.DataFrame([test_metrics]).to_csv(
            OUTPUT_DIRECTORY / "test_metrics.csv", index=False
        )
        pd.DataFrame(
            {
                "customerID": identifiers.loc[features_test.index].to_numpy(),
                "actual_churn": target_test.to_numpy(),
                "churn_probability": test_probabilities,
                "predicted_churn": (test_probabilities >= threshold).astype(int),
            }
        ).to_csv(OUTPUT_DIRECTORY / "test_predictions.csv", index=False)
        joblib.dump(
            {
                "model": final_model,
                "selected_base_model": selected_name,
                "calibration_method": calibration_method,
                "threshold": threshold,
                "false_negative_cost": 5.0,
                "false_positive_cost": 1.0,
                "feature_columns": features.columns.tolist(),
            },
            OUTPUT_DIRECTORY / "churn_model.joblib",
        )

    segment_assignments = None
    if segment:
        segmenter_path = OUTPUT_DIRECTORY / "segmenter.joblib"
        if segmenter_path.exists():
            logging.info("Loading existing segmenter from %s", segmenter_path)
            segmenter = joblib.load(segmenter_path)
            segment_labels = segmenter.predict(features)
            segment_assignments = pd.DataFrame(
                {
                    "customerID": identifiers.to_numpy(),
                    "segment": segment_labels,
                }
            )
        else:
            logging.info("Training autoencoder and K-Means segmentation")
            segmenter = AutoencoderKMeansSegmenter(
                n_clusters=3,
                random_state=RANDOM_STATE,
            )
            segment_assignments, _ = segmenter.fit_predict(features, identifiers)
            segment_assignments.to_csv(
                OUTPUT_DIRECTORY / "customer_segments.csv", index=False
            )
            joblib.dump(segmenter, OUTPUT_DIRECTORY / "segmenter.joblib")

    if explain:
        from src.explainability import explain_with_shap

        explain_with_shap(
            final_model,
            features_test,
            OUTPUT_DIRECTORY / "shap_summary.png",
            random_state=RANDOM_STATE,
        )

    # PRESCRIPTIVE RETENTION OPTIMIZATION (Stages 3, 4, 5)
    optimization_summary = None
    if optimize or run_sensitivity:
        logging.info("Running Prescriptive Retention Optimization (Budget: ₹%.2f, Horizon: %.1fy)", budget, horizon_years)
        # Compute calibrated churn probabilities for all customers
        all_probabilities = final_model.predict_proba(features)[:, 1]
        
        segments_series = None
        if segment_assignments is not None and "segment" in segment_assignments.columns:
            segments_series = segment_assignments["segment"]

        candidates_df = prepare_retention_candidates(
            customer_ids=identifiers,
            monthly_charges=data["MonthlyCharges"],
            churn_probabilities=all_probabilities,
            segments=segments_series,
            profit_margin=0.40,
            horizon_months=12,
            time_horizon_years=horizon_years,
        )

        # Baseline Strategy Comparisons
        baseline_results = run_baseline_comparisons(
            candidates_df=candidates_df,
            budget=budget,
            offers=DEFAULT_OFFERS,
            allow_negative_net_benefit=allow_negative_net_benefit,
            random_state=RANDOM_STATE,
        )
        milp_result = baseline_results["MILP Optimization"]
        baseline_summary_df = summarize_comparisons(baseline_results)

        # Sensitivity Analysis (if requested)
        sensitivity_df, sensitivity_summary = None, None
        if run_sensitivity:
            logging.info("Running Sensitivity Analysis across budget grid: %s", DEFAULT_SENSITIVITY_BUDGETS)
            sensitivity_df, sensitivity_summary = run_sensitivity_analysis(
                candidates_df=candidates_df,
                budgets=DEFAULT_SENSITIVITY_BUDGETS,
                offers=DEFAULT_OFFERS,
                allow_negative_net_benefit=allow_negative_net_benefit,
                random_state=RANDOM_STATE,
            )

        saved_opt_files = save_optimization_outputs(
            milp_result=milp_result,
            baseline_summary_df=baseline_summary_df,
            output_dir=OUTPUT_DIRECTORY,
            sensitivity_df=sensitivity_df,
            sensitivity_summary=sensitivity_summary,
        )
        logging.info("Saved prescriptive optimization files: %s", list(saved_opt_files.keys()))
        optimization_summary = {
            "primary_strategy": milp_result.to_dict(),
            "baseline_comparison": baseline_summary_df.to_dict(orient="records"),
            "saved_files": [str(p) for p in saved_opt_files.values()],
        }

    result = {
        "selected_model": selected_name,
        "calibration_method": calibration_method,
        "threshold": threshold,
        "validation_weighted_error_cost": validation_cost,
        "test_metrics": test_metrics,
        "output_directory": OUTPUT_DIRECTORY,
        "optimization_summary": optimization_summary,
    }
    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    arguments = _parse_args()
    result = run_pipeline(
        dataset_path=arguments.data,
        tune=arguments.tune,
        trials=arguments.trials,
        explain=arguments.explain,
        segment=not arguments.skip_segmentation,
        optimize=arguments.optimize,
        budget=arguments.budget,
        horizon_years=arguments.horizon_years,
        run_sensitivity=arguments.run_sensitivity,
        allow_negative_net_benefit=arguments.allow_negative_net_benefit,
    )
    logging.info(
        "Finished: model=%s calibration=%s threshold=%.4f metrics=%s",
        result["selected_model"],
        result["calibration_method"],
        result["threshold"],
        result["test_metrics"],
    )
    if result["optimization_summary"]:
        primary = result["optimization_summary"]["primary_strategy"]
        logging.info(
            "Optimization: Strategy=%s Budget=₹%.2f Spend=₹%.2f Retained=₹%.2f Net=₹%.2f ROI=%.2f%% Targeted=%d",
            primary["strategy"],
            primary["total_budget"],
            primary["total_cost"],
            primary["total_retained_value"],
            primary["total_net_benefit"],
            primary["roi_percentage"],
            primary["customers_targeted"],
        )
    logging.info("Artifacts written to %s", result["output_directory"])


if __name__ == "__main__":
    main()
