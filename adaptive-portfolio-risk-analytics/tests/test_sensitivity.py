"""Tests for Phase 2D sensitivity analysis helpers."""

from __future__ import annotations

import pandas as pd

from src.experiments import (
    compute_parameter_sensitivity,
    rank_experiments,
    summarize_by_parameter,
)


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy": ["HRP", "HRP", "HERC", "HERC"],
            "covariance_method": ["sample", "ledoit_wolf", "sample", "ledoit_wolf"],
            "rebalance_mode": ["calendar", "threshold", "calendar", "threshold"],
            "threshold": [None, 0.05, None, 0.05],
            "transaction_cost_bps": [10.0, 10.0, 10.0, 10.0],
            "slippage_bps": [5.0, 5.0, 5.0, 5.0],
            "vol_targeting_enabled": [False, True, False, True],
            "target_vol": [None, 0.10, None, 0.10],
            "defensive_asset": [None, "Synthetic Risk-Free", None, "Synthetic Risk-Free"],
            "cagr": [0.12, 0.10, 0.14, 0.11],
            "sharpe": [1.10, 1.05, 1.25, 1.15],
            "sortino": [1.50, 1.40, 1.65, 1.55],
            "volatility": [0.14, 0.12, 0.15, 0.13],
            "max_drawdown": [-0.22, -0.18, -0.25, -0.16],
            "calmar": [0.55, 0.56, 0.57, 0.69],
            "final_value": [1_200_000, 1_180_000, 1_240_000, 1_210_000],
            "status": ["success", "success", "success", "success"],
        }
    )


def test_rank_experiments_sorts_correctly() -> None:
    ranked = rank_experiments(_results(), objective="calmar")

    assert ranked.iloc[0]["calmar"] == 0.69


def test_max_drawdown_ranking_handles_less_negative_is_better() -> None:
    ranked = rank_experiments(_results(), objective="max_drawdown")

    assert ranked.iloc[0]["max_drawdown"] == -0.16


def test_summarize_by_parameter_returns_expected_columns() -> None:
    summary = summarize_by_parameter(_results(), parameter="covariance_method", metric="calmar")

    assert {
        "covariance_method",
        "calmar_mean",
        "calmar_std",
        "calmar_min",
        "calmar_max",
        "num_runs",
    } == set(summary.columns)


def test_parameter_sensitivity_returns_expected_columns() -> None:
    sensitivity = compute_parameter_sensitivity(_results(), metric="calmar")

    expected = {
        "parameter",
        "best_value",
        "worst_value",
        "metric_spread",
        "metric_mean",
        "metric_std",
    }
    assert expected.issubset(set(sensitivity.columns))
