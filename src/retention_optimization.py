"""Customer value proxy, retention offer evaluation, MILP optimization, baselines, and sensitivity analysis.

Stage 1: Input preparation & customer value proxy calculation.
Stage 2: Offer definitions and incremental net benefit calculations.
Stage 3: Mixed Integer Linear Programming (MILP) using PuLP and CBC.
Stage 4: Baseline strategy comparisons (No Campaign, Random, High Risk, High Value, Greedy, MILP).
Stage 5: Sensitivity analysis over a budget grid.

IMPORTANT NOTICE:
- Customer Value Proxy is an illustrative financial proxy:
  Customer Value Proxy = MonthlyCharges * profit_margin * 12 * time_horizon_years
  It is not actual audited accounting profit or full-lifecycle CLV.
- Assumed offer effectiveness rates are hypothetical scenario parameters for decision
  support and prescriptive optimization, not empirically observed causal uplifts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pulp


# Default margin and annualization parameters
DEFAULT_PROFIT_MARGIN: float = 0.40
DEFAULT_HORIZON_MONTHS: int = 12
DEFAULT_TIME_HORIZON_YEARS: float = 5.0
DEFAULT_SENSITIVITY_BUDGETS: tuple[float, ...] = (
    10000.0,
    25000.0,
    50000.0,
    75000.0,
    100000.0,
)


@dataclass(frozen=True)
class RetentionOffer:
    """Specification of a hypothetical retention offer."""

    name: str
    cost: float  # In INR (₹)
    relative_risk_reduction: float  # Fraction reduction in churn probability (0.0 to 1.0)
    capacity: int | None = None  # Maximum number of customers that can receive this offer

    def __post_init__(self) -> None:
        if self.cost < 0:
            raise ValueError(f"Offer cost must be non-negative; got {self.cost}.")
        if not 0.0 <= self.relative_risk_reduction <= 1.0:
            raise ValueError(
                f"Relative risk reduction must be between 0.0 and 1.0; got {self.relative_risk_reduction}."
            )
        if self.capacity is not None and self.capacity < 0:
            raise ValueError(f"Capacity must be non-negative if specified; got {self.capacity}.")


# Default scenario retention offers as specified by business requirements
DEFAULT_OFFERS: tuple[RetentionOffer, ...] = (
    RetentionOffer(name="Basic", cost=200.0, relative_risk_reduction=0.10),
    RetentionOffer(name="Standard", cost=500.0, relative_risk_reduction=0.20),
    RetentionOffer(name="Premium", cost=1000.0, relative_risk_reduction=0.30),
)


@dataclass
class OptimizationResult:
    """Encapsulates allocation decisions and aggregate business metrics."""

    strategy_name: str
    allocations_df: pd.DataFrame
    total_budget: float
    total_cost: float
    total_retained_value: float
    total_net_benefit: float
    roi_percentage: float
    customers_targeted: int
    offer_counts: dict[str, int]
    solver_status: str = "N/A"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a summary dictionary."""
        return {
            "strategy": self.strategy_name,
            "total_budget": round(self.total_budget, 2),
            "total_cost": round(self.total_cost, 2),
            "total_retained_value": round(self.total_retained_value, 2),
            "total_net_benefit": round(self.total_net_benefit, 2),
            "roi_percentage": round(self.roi_percentage, 2),
            "customers_targeted": self.customers_targeted,
            "solver_status": self.solver_status,
            **{f"count_{k}": v for k, v in self.offer_counts.items()},
        }


def calculate_customer_value_proxy(
    monthly_charges: pd.Series | np.ndarray | Sequence[float],
    profit_margin: float = DEFAULT_PROFIT_MARGIN,
    horizon_months: int = DEFAULT_HORIZON_MONTHS,
    time_horizon_years: float = DEFAULT_TIME_HORIZON_YEARS,
) -> np.ndarray:
    """Calculate the illustrative Customer Value Proxy.

    Formula:
        Customer Value Proxy = MonthlyCharges * profit_margin * horizon_months * time_horizon_years

    Note:
        This is an illustrative proxy representing operational contribution margin over
        the specified horizon, not full-lifecycle CLV or accounting net profit.
    """
    if profit_margin <= 0:
        raise ValueError(f"Profit margin must be positive; got {profit_margin}.")
    if horizon_months <= 0:
        raise ValueError(f"Horizon months must be positive; got {horizon_months}.")
    if time_horizon_years <= 0:
        raise ValueError(f"Time horizon years must be positive; got {time_horizon_years}.")

    charges = np.asarray(monthly_charges, dtype=np.float64)
    if np.isnan(charges).any():
        raise ValueError("MonthlyCharges contains NaN values.")
    if (charges < 0).any():
        raise ValueError("MonthlyCharges cannot contain negative numbers.")

    return charges * profit_margin * horizon_months * time_horizon_years


def prepare_retention_candidates(
    customer_ids: pd.Series | np.ndarray | Sequence[str],
    monthly_charges: pd.Series | np.ndarray | Sequence[float],
    churn_probabilities: pd.Series | np.ndarray | Sequence[float],
    segments: pd.Series | np.ndarray | Sequence[int | str] | None = None,
    profit_margin: float = DEFAULT_PROFIT_MARGIN,
    horizon_months: int = DEFAULT_HORIZON_MONTHS,
    time_horizon_years: float = DEFAULT_TIME_HORIZON_YEARS,
) -> pd.DataFrame:
    """Prepare customer-level dataset with predictive probabilities and value proxies."""
    c_ids = np.asarray(customer_ids, dtype=object)
    p_churn = np.asarray(churn_probabilities, dtype=np.float64)
    charges = np.asarray(monthly_charges, dtype=np.float64)

    if not (len(c_ids) == len(p_churn) == len(charges)):
        raise ValueError("Length mismatch between customer_ids, churn_probabilities, and monthly_charges.")

    if ((p_churn < 0.0) | (p_churn > 1.0)).any():
        raise ValueError("churn_probabilities must be bounded within [0.0, 1.0].")

    val_proxy = calculate_customer_value_proxy(
        charges,
        profit_margin=profit_margin,
        horizon_months=horizon_months,
        time_horizon_years=time_horizon_years,
    )

    df = pd.DataFrame(
        {
            "customerID": c_ids,
            "monthly_charges": charges,
            "churn_probability": p_churn,
            "customer_value_proxy": val_proxy,
        }
    )

    if segments is not None:
        seg_array = np.asarray(segments)
        if len(seg_array) != len(df):
            raise ValueError("Length mismatch between segments and customer data.")
        df["segment"] = seg_array

    return df


def calculate_offer_metrics(
    candidates_df: pd.DataFrame,
    offers: Sequence[RetentionOffer] = DEFAULT_OFFERS,
) -> pd.DataFrame:
    """Calculate incremental retained value and net benefit for each customer-offer pair."""
    required_cols = {"customerID", "churn_probability", "customer_value_proxy"}
    missing = required_cols - set(candidates_df.columns)
    if missing:
        raise ValueError(f"candidates_df is missing required columns: {missing}")

    result = candidates_df.copy()

    for offer in offers:
        col_value = f"value_retained_{offer.name}"
        col_net = f"net_benefit_{offer.name}"

        # Expected Incremental Retained Value
        retained_val = (
            result["churn_probability"]
            * offer.relative_risk_reduction
            * result["customer_value_proxy"]
        )
        # Expected Net Benefit
        net_ben = retained_val - offer.cost

        result[col_value] = retained_val
        result[col_net] = net_ben

    return result


def solve_retention_milp(
    candidates_df: pd.DataFrame,
    budget: float,
    offers: Sequence[RetentionOffer] = DEFAULT_OFFERS,
    allow_negative_net_benefit: bool = False,
    segment_capacities: dict[int | str, int] | None = None,
    solver_time_limit: int = 60,
) -> OptimizationResult:
    """Solve the customer retention campaign allocation using Mixed Integer Linear Programming (MILP)."""
    if budget < 0:
        raise ValueError(f"Budget must be non-negative; got {budget}.")

    # Ensure net benefit columns exist
    df = candidates_df
    for offer in offers:
        if f"net_benefit_{offer.name}" not in df.columns:
            df = calculate_offer_metrics(df, offers)
            break

    customer_ids = df["customerID"].tolist()
    n_customers = len(customer_ids)

    # Initialize PuLP Problem
    prob = pulp.LpProblem("Customer_Retention_Optimization", pulp.LpMaximize)

    # Create Binary Decision Variables x_{i, k}
    x: dict[tuple[int, str], pulp.LpVariable] = {}
    eligible_pairs = []

    for i in range(n_customers):
        for offer in offers:
            nb = float(df.loc[i, f"net_benefit_{offer.name}"])
            if allow_negative_net_benefit or nb > 0:
                var = pulp.LpVariable(f"x_{i}_{offer.name}", cat=pulp.LpBinary)
                x[(i, offer.name)] = var
                eligible_pairs.append((i, offer.name, nb, offer.cost))

    # Objective Function: Maximize sum_{i, k} net_benefit_{i, k} * x_{i, k}
    prob += pulp.lpSum(nb * x[(i, k)] for i, k, nb, _ in eligible_pairs), "Total_Net_Benefit"

    # Constraint 1: Mutually exclusive offers (at most one offer per customer)
    for i in range(n_customers):
        assigned_vars = [x[(i, o.name)] for o in offers if (i, o.name) in x]
        if assigned_vars:
            prob += pulp.lpSum(assigned_vars) <= 1, f"One_Offer_Limit_{i}"

    # Constraint 2: Total budget constraint
    prob += (
        pulp.lpSum(cost * x[(i, k)] for i, k, _, cost in eligible_pairs) <= budget,
        "Total_Budget_Constraint",
    )

    # Constraint 3: Offer-specific capacity constraints
    for offer in offers:
        if offer.capacity is not None:
            offer_vars = [x[(i, offer.name)] for i in range(n_customers) if (i, offer.name) in x]
            if offer_vars:
                prob += (
                    pulp.lpSum(offer_vars) <= offer.capacity,
                    f"Offer_Capacity_{offer.name}",
                )

    # Constraint 4: Optional segment-level capacities
    if segment_capacities and "segment" in df.columns:
        for seg_id, cap in segment_capacities.items():
            seg_indices = df.index[df["segment"] == seg_id].tolist()
            seg_vars = [x[(i, o.name)] for i in seg_indices for o in offers if (i, o.name) in x]
            if seg_vars:
                prob += pulp.lpSum(seg_vars) <= cap, f"Segment_Capacity_{seg_id}"

    # Solve with CBC solver
    solver = pulp.PULP_CBC_CMD(timeLimit=solver_time_limit, msg=False)
    prob.solve(solver)

    status_str = pulp.LpStatus[prob.status]

    # Extract allocations
    allocated_offers = ["None"] * n_customers
    assigned_costs = np.zeros(n_customers, dtype=np.float64)
    assigned_retained_values = np.zeros(n_customers, dtype=np.float64)
    assigned_net_benefits = np.zeros(n_customers, dtype=np.float64)

    offer_counts = {o.name: 0 for o in offers}
    offer_counts["None"] = 0

    if prob.status == pulp.constants.LpStatusOptimal or prob.status == pulp.constants.LpStatusNotSolved:
        for i, k, nb, cost in eligible_pairs:
            var_val = pulp.value(x[(i, k)])
            if var_val is not None and np.isclose(var_val, 1.0):
                allocated_offers[i] = k
                assigned_costs[i] = cost
                val_ret = float(df.loc[i, f"value_retained_{k}"])
                assigned_retained_values[i] = val_ret
                assigned_net_benefits[i] = nb
                offer_counts[k] += 1

    offer_counts["None"] = n_customers - sum(offer_counts[o.name] for o in offers)

    allocations_df = df[
        ["customerID", "churn_probability", "customer_value_proxy"]
        + ([col for col in ["segment"] if col in df.columns])
    ].copy()
    allocations_df["assigned_offer"] = allocated_offers
    allocations_df["offer_cost"] = assigned_costs
    allocations_df["expected_retained_value"] = assigned_retained_values
    allocations_df["expected_net_benefit"] = assigned_net_benefits

    total_cost = float(np.sum(assigned_costs))
    total_retained = float(np.sum(assigned_retained_values))
    total_net = float(np.sum(assigned_net_benefits))
    targeted = int(np.count_nonzero(assigned_costs > 0))
    roi = (total_net / total_cost * 100.0) if total_cost > 0 else 0.0

    return OptimizationResult(
        strategy_name="MILP Optimization",
        allocations_df=allocations_df,
        total_budget=budget,
        total_cost=total_cost,
        total_retained_value=total_retained,
        total_net_benefit=total_net,
        roi_percentage=roi,
        customers_targeted=targeted,
        offer_counts=offer_counts,
        solver_status=status_str,
    )


def _evaluate_heuristic_assignment(
    candidates_df: pd.DataFrame,
    budget: float,
    offers: Sequence[RetentionOffer],
    strategy_name: str,
    selection_order: Sequence[int],
    offer_selection_func: Any,
) -> OptimizationResult:
    """Helper to evaluate heuristic policies under identical budget and capacity constraints."""
    n_customers = len(candidates_df)
    allocated_offers = ["None"] * n_customers
    assigned_costs = np.zeros(n_customers, dtype=np.float64)
    assigned_retained_values = np.zeros(n_customers, dtype=np.float64)
    assigned_net_benefits = np.zeros(n_customers, dtype=np.float64)

    remaining_budget = budget
    offer_capacities = {o.name: o.capacity for o in offers}
    offer_counts = {o.name: 0 for o in offers}
    offer_counts["None"] = 0

    for idx in selection_order:
        best_offer = offer_selection_func(idx, remaining_budget, offer_capacities)
        if best_offer is not None:
            allocated_offers[idx] = best_offer.name
            assigned_costs[idx] = best_offer.cost
            val_ret = float(candidates_df.loc[idx, f"value_retained_{best_offer.name}"])
            assigned_retained_values[idx] = val_ret
            nb = float(candidates_df.loc[idx, f"net_benefit_{best_offer.name}"])
            assigned_net_benefits[idx] = nb

            remaining_budget -= best_offer.cost
            offer_counts[best_offer.name] += 1
            if offer_capacities[best_offer.name] is not None:
                offer_capacities[best_offer.name] -= 1

    offer_counts["None"] = n_customers - sum(offer_counts[o.name] for o in offers)

    allocations_df = candidates_df[
        ["customerID", "churn_probability", "customer_value_proxy"]
        + ([col for col in ["segment"] if col in candidates_df.columns])
    ].copy()
    allocations_df["assigned_offer"] = allocated_offers
    allocations_df["offer_cost"] = assigned_costs
    allocations_df["expected_retained_value"] = assigned_retained_values
    allocations_df["expected_net_benefit"] = assigned_net_benefits

    total_cost = float(np.sum(assigned_costs))
    total_retained = float(np.sum(assigned_retained_values))
    total_net = float(np.sum(assigned_net_benefits))
    targeted = int(np.count_nonzero(assigned_costs > 0))
    roi = (total_net / total_cost * 100.0) if total_cost > 0 else 0.0

    return OptimizationResult(
        strategy_name=strategy_name,
        allocations_df=allocations_df,
        total_budget=budget,
        total_cost=total_cost,
        total_retained_value=total_retained,
        total_net_benefit=total_net,
        roi_percentage=roi,
        customers_targeted=targeted,
        offer_counts=offer_counts,
        solver_status="Feasible",
    )


def run_baseline_comparisons(
    candidates_df: pd.DataFrame,
    budget: float,
    offers: Sequence[RetentionOffer] = DEFAULT_OFFERS,
    allow_negative_net_benefit: bool = False,
    random_state: int = 42,
) -> dict[str, OptimizationResult]:
    """Evaluate 6 retention strategies under identical parameters, budget, and capacity constraints."""
    df = candidates_df
    for offer in offers:
        if f"net_benefit_{offer.name}" not in df.columns:
            df = calculate_offer_metrics(df, offers)
            break

    n_customers = len(df)
    results: dict[str, OptimizationResult] = {}

    # 1. No Campaign
    results["No Campaign"] = _evaluate_heuristic_assignment(
        candidates_df=df,
        budget=budget,
        offers=offers,
        strategy_name="No Campaign",
        selection_order=[],
        offer_selection_func=lambda idx, rem_b, caps: None,
    )

    # Helper function to pick the highest profitable/affordable offer for a customer
    def get_highest_viable_offer(idx: int, rem_b: float, caps: dict[str, int | None]) -> RetentionOffer | None:
        for offer in sorted(offers, key=lambda o: o.cost, reverse=True):
            if offer.cost <= rem_b:
                if caps[offer.name] is None or caps[offer.name] > 0:
                    nb = float(df.loc[idx, f"net_benefit_{offer.name}"])
                    if allow_negative_net_benefit or nb > 0:
                        return offer
        return None

    # 2. Random Selection (deterministic with random_state and customerID tie-break)
    rng = np.random.default_rng(random_state)
    random_order = list(range(n_customers))
    rng.shuffle(random_order)
    results["Random Selection"] = _evaluate_heuristic_assignment(
        candidates_df=df,
        budget=budget,
        offers=offers,
        strategy_name="Random Selection",
        selection_order=random_order,
        offer_selection_func=get_highest_viable_offer,
    )

    # 3. Highest Churn Risk (Deterministic sorting by churn_probability desc, customerID asc)
    risk_order = (
        df.sort_values(by=["churn_probability", "customerID"], ascending=[False, True])
        .index.tolist()
    )
    results["Highest Churn Risk"] = _evaluate_heuristic_assignment(
        candidates_df=df,
        budget=budget,
        offers=offers,
        strategy_name="Highest Churn Risk",
        selection_order=risk_order,
        offer_selection_func=get_highest_viable_offer,
    )

    # 4. Highest Customer Value (Deterministic sorting by customer_value_proxy desc, customerID asc)
    val_order = (
        df.sort_values(by=["customer_value_proxy", "customerID"], ascending=[False, True])
        .index.tolist()
    )
    results["Highest Customer Value"] = _evaluate_heuristic_assignment(
        candidates_df=df,
        budget=budget,
        offers=offers,
        strategy_name="Highest Customer Value",
        selection_order=val_order,
        offer_selection_func=get_highest_viable_offer,
    )

    # 5. Greedy Expected Net Benefit (Sort customer-offer candidate pairs)
    candidate_pairs = []
    for idx in range(n_customers):
        c_id = df.loc[idx, "customerID"]
        for offer in offers:
            nb = float(df.loc[idx, f"net_benefit_{offer.name}"])
            if allow_negative_net_benefit or nb > 0:
                candidate_pairs.append((nb, -offer.cost, str(c_id), idx, offer))

    # Sort descending by net benefit, breaking ties with lower cost, then customerID
    candidate_pairs.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

    greedy_assigned: set[int] = set()
    greedy_order: list[int] = []
    greedy_offer_for_customer: dict[int, RetentionOffer] = {}

    rem_b = budget
    caps = {o.name: o.capacity for o in offers}

    for nb, neg_cost, _, idx, offer in candidate_pairs:
        if idx not in greedy_assigned and offer.cost <= rem_b:
            if caps[offer.name] is None or caps[offer.name] > 0:
                greedy_assigned.add(idx)
                greedy_order.append(idx)
                greedy_offer_for_customer[idx] = offer
                rem_b -= offer.cost
                if caps[offer.name] is not None:
                    caps[offer.name] -= 1

    results["Greedy Expected Net Benefit"] = _evaluate_heuristic_assignment(
        candidates_df=df,
        budget=budget,
        offers=offers,
        strategy_name="Greedy Expected Net Benefit",
        selection_order=greedy_order,
        offer_selection_func=lambda idx, b, c: greedy_offer_for_customer.get(idx),
    )

    # 6. MILP Optimization
    results["MILP Optimization"] = solve_retention_milp(
        candidates_df=df,
        budget=budget,
        offers=offers,
        allow_negative_net_benefit=allow_negative_net_benefit,
    )

    return results


def summarize_comparisons(
    comparison_results: dict[str, OptimizationResult],
) -> pd.DataFrame:
    """Format multiple comparison results into a clean benchmark table."""
    records = [res.to_dict() for res in comparison_results.values()]
    return pd.DataFrame.from_records(records)


def run_sensitivity_analysis(
    candidates_df: pd.DataFrame,
    budgets: Sequence[float] = DEFAULT_SENSITIVITY_BUDGETS,
    offers: Sequence[RetentionOffer] = DEFAULT_OFFERS,
    allow_negative_net_benefit: bool = False,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate all 6 baseline strategies across a grid of campaign budgets.

    Parameters:
        candidates_df: Customer-level candidate DataFrame.
        budgets: Sequence of budget values in INR (₹).
        offers: Sequence of RetentionOffer objects.
        allow_negative_net_benefit: Whether to allow unprofitable assignments.
        random_state: Seed for reproducible random baseline.

    Returns:
        tuple (sensitivity_df, sensitivity_summary_dict)
    """
    records: list[dict[str, Any]] = []

    for b in budgets:
        baseline_results = run_baseline_comparisons(
            candidates_df=candidates_df,
            budget=float(b),
            offers=offers,
            allow_negative_net_benefit=allow_negative_net_benefit,
            random_state=random_state,
        )
        for strategy_name, result in baseline_results.items():
            row = result.to_dict()
            row["budget_scenario"] = float(b)
            records.append(row)

    sensitivity_df = pd.DataFrame.from_records(records)

    # Compute marginal returns for MILP
    milp_rows = sensitivity_df[sensitivity_df["strategy"] == "MILP Optimization"].sort_values("total_budget")
    milp_rows["marginal_cost"] = milp_rows["total_cost"].diff().fillna(milp_rows["total_cost"].iloc[0])
    milp_rows["marginal_net_benefit"] = milp_rows["total_net_benefit"].diff().fillna(milp_rows["total_net_benefit"].iloc[0])
    milp_rows["marginal_roi"] = np.where(
        milp_rows["marginal_cost"] > 0,
        milp_rows["marginal_net_benefit"] / milp_rows["marginal_cost"] * 100.0,
        0.0,
    )

    summary_dict = {
        "budgets_evaluated": [float(b) for b in budgets],
        "strategies_evaluated": [
            "No Campaign",
            "Random Selection",
            "Highest Churn Risk",
            "Highest Customer Value",
            "Greedy Expected Net Benefit",
            "MILP Optimization",
        ],
        "allow_negative_net_benefit": allow_negative_net_benefit,
        "milp_sensitivity_trend": [
            {
                "budget": float(row["total_budget"]),
                "cost": float(row["total_cost"]),
                "retained_value": float(row["total_retained_value"]),
                "net_benefit": float(row["total_net_benefit"]),
                "roi_percentage": float(row["roi_percentage"]),
                "customers_targeted": int(row["customers_targeted"]),
            }
            for _, row in milp_rows.iterrows()
        ],
    }

    return sensitivity_df, summary_dict


def save_optimization_outputs(
    milp_result: OptimizationResult,
    baseline_summary_df: pd.DataFrame,
    output_dir: Path,
    sensitivity_df: pd.DataFrame | None = None,
    sensitivity_summary: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Save all prescriptive optimization outputs into the specified directory.

    Saved files:
        - outputs/retention_policy.csv
        - outputs/optimization_summary.json
        - outputs/sensitivity_analysis.csv (optional)
        - outputs/sensitivity_summary.json (optional)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}

    # 1. retention_policy.csv
    policy_path = output_dir / "retention_policy.csv"
    milp_result.allocations_df.to_csv(policy_path, index=False)
    saved_paths["retention_policy"] = policy_path

    # 2. optimization_summary.json
    summary_path = output_dir / "optimization_summary.json"
    summary_payload = {
        "primary_strategy": milp_result.to_dict(),
        "baseline_comparison_table": baseline_summary_df.to_dict(orient="records"),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    saved_paths["optimization_summary"] = summary_path

    # 3. sensitivity files (if available)
    if sensitivity_df is not None:
        sens_csv_path = output_dir / "sensitivity_analysis.csv"
        sensitivity_df.to_csv(sens_csv_path, index=False)
        saved_paths["sensitivity_analysis"] = sens_csv_path

    if sensitivity_summary is not None:
        sens_json_path = output_dir / "sensitivity_summary.json"
        with open(sens_json_path, "w", encoding="utf-8") as f:
            json.dump(sensitivity_summary, f, indent=2)
        saved_paths["sensitivity_summary"] = sens_json_path

    return saved_paths
