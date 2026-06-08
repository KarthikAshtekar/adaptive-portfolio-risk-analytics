"""Sensitivity-analysis helpers for experiment result tables."""

from __future__ import annotations

import pandas as pd


OBJECTIVE_DIRECTIONS = {
    "cagr": False,
    "sharpe": False,
    "sortino": False,
    "calmar": False,
    "max_drawdown": False,
    "final_value": False,
}


def rank_experiments(
    experiment_results_df: pd.DataFrame,
    objective: str = "calmar",
) -> pd.DataFrame:
    """Sort successful experiments by the chosen objective metric."""
    if objective not in OBJECTIVE_DIRECTIONS:
        supported = ", ".join(sorted(OBJECTIVE_DIRECTIONS))
        raise ValueError(f"unsupported objective '{objective}'. Supported: {supported}")

    results = _successful_results(experiment_results_df)
    if objective not in results.columns:
        raise ValueError(f"objective '{objective}' not present in experiment_results_df")

    ascending = OBJECTIVE_DIRECTIONS[objective]
    return results.sort_values(by=objective, ascending=ascending).reset_index(drop=True)


def summarize_by_parameter(
    experiment_results_df: pd.DataFrame,
    parameter,
    metric,
) -> pd.DataFrame:
    """Summarize a metric grouped by the requested experiment parameter."""
    results = _successful_results(experiment_results_df)
    if parameter not in results.columns:
        raise ValueError(f"parameter '{parameter}' not present in experiment_results_df")
    if metric not in results.columns:
        raise ValueError(f"metric '{metric}' not present in experiment_results_df")

    summary = (
        results.groupby(parameter, dropna=False)[metric]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": f"{metric}_mean",
                "std": f"{metric}_std",
                "min": f"{metric}_min",
                "max": f"{metric}_max",
                "count": "num_runs",
            }
        )
    )
    return summary


def compute_parameter_sensitivity(
    experiment_results_df: pd.DataFrame,
    metric,
) -> pd.DataFrame:
    """Estimate how much the chosen metric moves across each parameter."""
    results = _successful_results(experiment_results_df)
    if metric not in results.columns:
        raise ValueError(f"metric '{metric}' not present in experiment_results_df")

    parameter_columns = [
        "strategy",
        "covariance_method",
        "rebalance_mode",
        "threshold",
        "transaction_cost_bps",
        "slippage_bps",
        "vol_targeting_enabled",
        "target_vol",
        "defensive_asset",
    ]

    rows = []
    for parameter in parameter_columns:
        if parameter not in results.columns:
            continue
        parameter_frame = results[[parameter, metric]].dropna(subset=[metric]).copy()
        if parameter_frame.empty:
            continue

        grouped = parameter_frame.groupby(parameter, dropna=False)[metric].mean()
        if grouped.empty:
            continue

        if metric == "max_drawdown":
            best_value = grouped.idxmax()
            worst_value = grouped.idxmin()
            metric_spread = float(grouped.max() - grouped.min())
        else:
            best_value = grouped.idxmax()
            worst_value = grouped.idxmin()
            metric_spread = float(grouped.max() - grouped.min())

        rows.append(
            {
                "parameter": parameter,
                "best_value": best_value,
                "worst_value": worst_value,
                "metric_spread": metric_spread,
                "metric_mean": float(grouped.mean()),
                "metric_std": float(grouped.std(ddof=0)),
            }
        )

    return pd.DataFrame(rows)


def _successful_results(experiment_results_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(experiment_results_df, pd.DataFrame):
        raise TypeError("experiment_results_df must be a pandas DataFrame")
    if experiment_results_df.empty:
        raise ValueError("experiment_results_df must not be empty")

    if "status" in experiment_results_df.columns:
        results = experiment_results_df[experiment_results_df["status"] == "success"].copy()
    else:
        results = experiment_results_df.copy()

    if results.empty:
        raise ValueError("experiment_results_df does not contain any successful runs")
    return results
