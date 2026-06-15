"""Tests for FRM stress-testing helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics import (
    apply_hypothetical_stress,
    calculate_correlation_stress,
    calculate_historical_stress_performance,
    calculate_stress_period_benchmark_comparison,
    classify_asset_for_stress,
    find_worst_periods,
)


def test_historical_stress_period_return_calculation() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.00], index=dates)
    periods = {"Test Stress": ("2024-01-02", "2024-01-04")}

    result = calculate_historical_stress_performance(returns, stress_periods=periods)

    expected_return = (1 - 0.02) * (1 + 0.03) * (1 - 0.01) - 1
    assert np.isclose(result.loc[0, "period_return"], expected_return)
    assert result.loc[0, "n_observations"] == 3


def test_missing_stress_period_data_returns_no_data_row() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.00], index=dates)
    periods = {"Missing Stress": ("2023-01-01", "2023-01-31")}

    result = calculate_historical_stress_performance(returns, stress_periods=periods)

    assert result.loc[0, "status"] == "no_data"
    assert np.isnan(result.loc[0, "period_return"])


def test_find_worst_periods_returns_expected_window() -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    returns = pd.Series([0.01, -0.10, -0.10, 0.05, 0.02, 0.01], index=dates)

    result = find_worst_periods(returns, windows=(2,))

    assert result.loc[0, "window_days"] == 2
    assert result.loc[0, "start_date"] == dates[1].date().isoformat()


def test_hypothetical_stress_return_equals_weighted_sum() -> None:
    weights = pd.Series({"HDFCBANK.NS": 0.60, "GOLDBEES.NS": 0.40})
    scenario = {"equity": -0.10, "gold": 0.00}

    stress_return = apply_hypothetical_stress(weights, scenario)

    assert np.isclose(stress_return, -0.06)


def test_asset_classifier_maps_simple_yahoo_tickers() -> None:
    assert classify_asset_for_stress("GOLDBEES.NS") == "gold"
    assert classify_asset_for_stress("SILVERBEES.NS") == "silver"
    assert classify_asset_for_stress("LIQUIDBEES.NS") == "defensive"
    assert classify_asset_for_stress("HDFCBANK.NS") == "equity"


def test_stress_period_benchmark_comparison_drawdown_reduction() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    strategy = pd.Series([0.00, -0.02, -0.01, 0.01], index=dates)
    benchmark = pd.Series([0.00, -0.05, -0.02, 0.01], index=dates)
    periods = {"Stress": ("2024-01-01", "2024-01-04")}

    result = calculate_stress_period_benchmark_comparison(strategy, benchmark, periods)

    assert result.loc[0, "drawdown_reduction"] > 0.0
    assert result.loc[0, "excess_stress_return"] > 0.0


def test_correlation_stress_increases_or_matches_volatility() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.03, -0.01, 0.00],
            "B": [-0.01, 0.02, -0.03, 0.01, 0.00],
        },
        index=dates,
    )
    weights = pd.Series({"A": 0.50, "B": 0.50})

    result = calculate_correlation_stress(weights, returns, stressed_correlation=0.8)

    assert result["correlation_stressed_volatility"] >= result["normal_volatility"]
