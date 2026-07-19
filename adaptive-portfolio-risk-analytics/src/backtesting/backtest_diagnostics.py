"""Diagnostics helpers for realistic backtesting outputs."""

from __future__ import annotations

import pandas as pd


def build_rebalance_summary(
    rebalance_log_df: pd.DataFrame,
) -> dict[str, float | int | dict[str, float | int]]:
    """Summarize rebalance activity from the rebalance log."""
    if not isinstance(rebalance_log_df, pd.DataFrame):
        raise TypeError("rebalance_log_df must be a pandas DataFrame")
    if rebalance_log_df.empty:
        return {
            "total_rebalances": 0,
            "total_turnover": 0.0,
            "average_turnover": 0.0,
            "total_transaction_cost": 0.0,
            "average_transaction_cost": 0.0,
            "max_weight_drift": 0.0,
            "average_max_weight_drift": 0.0,
            "calendar_rebalances": 0,
            "threshold_rebalances": 0,
            "calendar_or_threshold_rebalances": 0,
            "rebalance_reason_counts": {},
            "average_turnover_by_reason": {},
            "max_weight_drift_by_reason": {},
        }

    reason_counts = (
        rebalance_log_df["rebalance_reason"].value_counts().sort_index().to_dict()
        if "rebalance_reason" in rebalance_log_df
        else {}
    )
    average_turnover_by_reason = (
        rebalance_log_df.groupby("rebalance_reason")["turnover"].mean().sort_index().to_dict()
        if "rebalance_reason" in rebalance_log_df
        else {}
    )
    max_weight_drift_by_reason = (
        rebalance_log_df.groupby("rebalance_reason")["max_weight_drift"]
        .max()
        .sort_index()
        .to_dict()
        if "rebalance_reason" in rebalance_log_df
        else {}
    )

    return {
        "total_rebalances": int(len(rebalance_log_df)),
        "total_turnover": float(rebalance_log_df["turnover"].sum()),
        "average_turnover": float(rebalance_log_df["turnover"].mean()),
        "total_transaction_cost": float(rebalance_log_df["transaction_cost"].sum()),
        "average_transaction_cost": float(rebalance_log_df["transaction_cost"].mean()),
        "max_weight_drift": float(rebalance_log_df["max_weight_drift"].max()),
        "average_max_weight_drift": float(rebalance_log_df["max_weight_drift"].mean()),
        "calendar_rebalances": int(reason_counts.get("calendar", 0)),
        "threshold_rebalances": int(reason_counts.get("threshold", 0)),
        "calendar_or_threshold_rebalances": int(reason_counts.get("calendar_or_threshold", 0)),
        "rebalance_reason_counts": {str(key): int(value) for key, value in reason_counts.items()},
        "average_turnover_by_reason": {
            str(key): float(value) for key, value in average_turnover_by_reason.items()
        },
        "max_weight_drift_by_reason": {
            str(key): float(value) for key, value in max_weight_drift_by_reason.items()
        },
    }


def compare_cost_drag(
    gross_portfolio_values: pd.Series,
    net_portfolio_values: pd.Series,
) -> dict[str, float]:
    """Compare gross and net portfolio values to quantify cost drag."""
    if not isinstance(gross_portfolio_values, pd.Series) or not isinstance(
        net_portfolio_values, pd.Series
    ):
        raise TypeError("gross_portfolio_values and net_portfolio_values must be pandas Series")
    if gross_portfolio_values.empty or net_portfolio_values.empty:
        return {
            "gross_final_value": 0.0,
            "net_final_value": 0.0,
            "cost_drag": 0.0,
            "cost_drag_pct": 0.0,
        }

    gross_final = float(gross_portfolio_values.iloc[-1])
    net_final = float(net_portfolio_values.iloc[-1])
    cost_drag = gross_final - net_final
    cost_drag_pct = cost_drag / gross_final if gross_final != 0.0 else 0.0

    return {
        "gross_final_value": gross_final,
        "net_final_value": net_final,
        "cost_drag": float(cost_drag),
        "cost_drag_pct": float(cost_drag_pct),
    }
