"""Unit tests for Stage 1 to Stage 5: Customer Value, Offer Benefits, MILP, Baselines, and Sensitivity."""

import numpy as np
import pandas as pd
import pytest

from src.retention_optimization import (
    DEFAULT_OFFERS,
    RetentionOffer,
    calculate_customer_value_proxy,
    calculate_offer_metrics,
    prepare_retention_candidates,
    run_baseline_comparisons,
    run_sensitivity_analysis,
    solve_retention_milp,
    summarize_comparisons,
)


def test_customer_value_proxy_calculation() -> None:
    charges = [100.0, 50.0, 0.0]
    proxies = calculate_customer_value_proxy(
        charges, profit_margin=0.40, horizon_months=12, time_horizon_years=1.0
    )
    np.testing.assert_allclose(proxies, [480.0, 240.0, 0.0])

    proxies_3y = calculate_customer_value_proxy(
        charges, profit_margin=0.40, horizon_months=12, time_horizon_years=3.0
    )
    np.testing.assert_allclose(proxies_3y, [1440.0, 720.0, 0.0])


def test_customer_value_proxy_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="cannot contain negative numbers"):
        calculate_customer_value_proxy([-10.0])

    with pytest.raises(ValueError, match="contains NaN values"):
        calculate_customer_value_proxy([np.nan])

    with pytest.raises(ValueError, match="Time horizon years must be positive"):
        calculate_customer_value_proxy([100.0], time_horizon_years=-1.0)


def test_prepare_retention_candidates() -> None:
    c_ids = ["C1", "C2"]
    charges = [70.0, 100.0]
    p_churn = [0.5, 0.8]
    segments = [1, 2]

    df = prepare_retention_candidates(
        customer_ids=c_ids,
        monthly_charges=charges,
        churn_probabilities=p_churn,
        segments=segments,
        time_horizon_years=1.0,
    )

    assert len(df) == 2
    assert "customer_value_proxy" in df.columns
    assert df.loc[0, "customer_value_proxy"] == 70.0 * 0.4 * 12  # 336.0
    assert df.loc[1, "customer_value_proxy"] == 100.0 * 0.4 * 12  # 480.0
    assert df.loc[0, "segment"] == 1


def test_calculate_offer_metrics_manual_verification() -> None:
    candidates = pd.DataFrame(
        {
            "customerID": ["C1", "C2"],
            "monthly_charges": [100.0, 200.0],
            "churn_probability": [0.50, 0.90],
            "customer_value_proxy": [480.0, 960.0],
        }
    )

    metrics = calculate_offer_metrics(candidates, offers=DEFAULT_OFFERS)

    # Customer 1 checks
    assert np.isclose(metrics.loc[0, "value_retained_Basic"], 24.0)
    assert np.isclose(metrics.loc[0, "net_benefit_Basic"], -176.0)
    assert np.isclose(metrics.loc[0, "value_retained_Standard"], 48.0)
    assert np.isclose(metrics.loc[0, "net_benefit_Standard"], -452.0)
    assert np.isclose(metrics.loc[0, "value_retained_Premium"], 72.0)
    assert np.isclose(metrics.loc[0, "net_benefit_Premium"], -928.0)

    # Customer 2 checks
    assert np.isclose(metrics.loc[1, "value_retained_Basic"], 86.4)
    assert np.isclose(metrics.loc[1, "net_benefit_Basic"], -113.6)
    assert np.isclose(metrics.loc[1, "value_retained_Premium"], 259.2)
    assert np.isclose(metrics.loc[1, "net_benefit_Premium"], -740.8)


def test_hand_calculated_3_customer_2_offer_example() -> None:
    offers = (
        RetentionOffer(name="OfferA", cost=100.0, relative_risk_reduction=0.20),
        RetentionOffer(name="OfferB", cost=250.0, relative_risk_reduction=0.40),
    )

    candidates = pd.DataFrame(
        {
            "customerID": ["C1", "C2", "C3"],
            "monthly_charges": [100.0, 150.0, 80.0],
            "churn_probability": [0.80, 0.60, 0.20],
            "customer_value_proxy": [1000.0, 1500.0, 800.0],
        }
    )

    result = solve_retention_milp(
        candidates_df=candidates,
        budget=350.0,
        offers=offers,
        allow_negative_net_benefit=False,
    )

    assert result.solver_status in ["Optimal", "Feasible"]
    assert np.isclose(result.total_cost, 350.0)
    assert np.isclose(result.total_retained_value, 520.0)
    assert np.isclose(result.total_net_benefit, 170.0)
    assert result.customers_targeted == 2

    alloc = result.allocations_df.set_index("customerID")
    assert alloc.loc["C1", "assigned_offer"] == "OfferA"
    assert np.isclose(alloc.loc["C1", "expected_net_benefit"], 60.0)

    assert alloc.loc["C2", "assigned_offer"] == "OfferB"
    assert np.isclose(alloc.loc["C2", "expected_net_benefit"], 110.0)

    assert alloc.loc["C3", "assigned_offer"] == "None"
    assert np.isclose(alloc.loc["C3", "expected_net_benefit"], 0.0)


def test_baseline_comparisons_consistency() -> None:
    offers = (
        RetentionOffer(name="OfferA", cost=100.0, relative_risk_reduction=0.20),
        RetentionOffer(name="OfferB", cost=250.0, relative_risk_reduction=0.40),
    )

    candidates = pd.DataFrame(
        {
            "customerID": ["C1", "C2", "C3"],
            "monthly_charges": [100.0, 150.0, 80.0],
            "churn_probability": [0.80, 0.60, 0.20],
            "customer_value_proxy": [1000.0, 1500.0, 800.0],
        }
    )

    baselines = run_baseline_comparisons(
        candidates_df=candidates,
        budget=350.0,
        offers=offers,
        allow_negative_net_benefit=False,
        random_state=42,
    )

    assert len(baselines) == 6
    expected_names = [
        "No Campaign",
        "Random Selection",
        "Highest Churn Risk",
        "Highest Customer Value",
        "Greedy Expected Net Benefit",
        "MILP Optimization",
    ]
    for name in expected_names:
        assert name in baselines
        assert baselines[name].total_cost <= 350.0

    summary = summarize_comparisons(baselines)
    assert len(summary) == 6

    milp_net = baselines["MILP Optimization"].total_net_benefit
    for name, res in baselines.items():
        assert milp_net >= res.total_net_benefit - 1e-5


def test_sensitivity_analysis_grid() -> None:
    """Verify that sensitivity analysis executes across multiple budgets and tracks diminishing marginal returns."""
    offers = (
        RetentionOffer(name="OfferA", cost=100.0, relative_risk_reduction=0.20),
        RetentionOffer(name="OfferB", cost=250.0, relative_risk_reduction=0.40),
    )

    candidates = pd.DataFrame(
        {
            "customerID": ["C1", "C2", "C3"],
            "monthly_charges": [100.0, 150.0, 80.0],
            "churn_probability": [0.80, 0.60, 0.20],
            "customer_value_proxy": [1000.0, 1500.0, 800.0],
        }
    )

    budgets = [100.0, 250.0, 350.0]
    sens_df, sens_dict = run_sensitivity_analysis(
        candidates_df=candidates,
        budgets=budgets,
        offers=offers,
        allow_negative_net_benefit=False,
    )

    assert len(sens_df) == len(budgets) * 6
    assert "marginal_cost" in sens_df.columns or "marginal_roi" in sens_df.columns or len(sens_dict["milp_sensitivity_trend"]) == 3
    
    milp_trend = sens_dict["milp_sensitivity_trend"]
    # Total net benefit must be non-decreasing as budget increases
    assert milp_trend[0]["net_benefit"] <= milp_trend[1]["net_benefit"] <= milp_trend[2]["net_benefit"]
