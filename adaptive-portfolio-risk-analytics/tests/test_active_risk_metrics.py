"""Tests for benchmark-relative and concentration metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics import (
    calculate_active_risk_metrics,
    calculate_beta,
    calculate_concentration_metrics,
    calculate_drawdown_durations,
    calculate_hit_ratio,
    calculate_information_ratio,
    calculate_jensens_alpha,
    calculate_simple_alpha,
    calculate_tracking_error,
)


def test_simple_alpha_uses_cagr_difference() -> None:
    alpha = calculate_simple_alpha(strategy_cagr=0.12, benchmark_cagr=0.08)
    assert np.isclose(alpha, 0.04)


def test_beta_handles_misaligned_dates_and_nans() -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    benchmark = pd.Series([0.01, 0.02, -0.01, 0.00, 0.015, 0.005], index=dates)
    strategy = pd.Series(
        [2.0 * benchmark.loc[date] + 0.001 for date in dates[1:]],
        index=dates[1:],
    )
    strategy.loc[dates[3]] = np.nan

    beta = calculate_beta(strategy, benchmark)

    assert np.isclose(beta, 2.0)


def test_zero_benchmark_variance_returns_nan() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    benchmark = pd.Series([0.001] * 5, index=dates)
    strategy = pd.Series([0.001, 0.002, 0.000, 0.003, 0.001], index=dates)

    assert np.isnan(calculate_beta(strategy, benchmark))
    assert np.isnan(calculate_jensens_alpha(strategy, benchmark)["annualized_jensen_alpha"])


def test_identical_strategy_and_benchmark_active_metrics_are_safe() -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    returns = pd.Series([0.01, -0.005, 0.002, 0.006, -0.001, 0.004], index=dates)

    assert np.isclose(calculate_beta(returns, returns), 1.0)
    assert np.isclose(calculate_tracking_error(returns, returns), 0.0)
    assert np.isnan(calculate_information_ratio(returns, returns))


def test_jensens_alpha_estimates_known_daily_alpha() -> None:
    dates = pd.date_range("2024-01-01", periods=7, freq="B")
    annual_rf = 0.02
    daily_rf = annual_rf / 252.0
    daily_alpha = 0.0002
    benchmark = pd.Series([0.003, -0.002, 0.004, 0.001, -0.001, 0.002, 0.000], index=dates)
    strategy = daily_rf + daily_alpha + 1.5 * (benchmark - daily_rf)

    result = calculate_jensens_alpha(
        strategy,
        benchmark,
        annual_risk_free_rate=annual_rf,
    )

    assert np.isclose(result["daily_jensen_alpha"], daily_alpha)
    assert np.isclose(result["annualized_jensen_alpha"], daily_alpha * 252.0)


def test_hit_ratio_is_one_when_strategy_always_beats_benchmark() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    benchmark = pd.Series([0.001, -0.002, 0.000, 0.003, -0.001], index=dates)
    strategy = benchmark + 0.001

    assert calculate_hit_ratio(strategy, benchmark) == 1.0


def test_drawdown_durations_with_recovery() -> None:
    values = pd.Series([100.0, 90.0, 80.0, 100.0, 110.0, 105.0, 111.0])

    durations = calculate_drawdown_durations(values)

    assert durations["max_drawdown_duration"] == 2
    assert durations["current_drawdown_duration"] == 0
    assert np.isclose(durations["average_drawdown_duration"], 1.5)


def test_drawdown_durations_with_ongoing_unrecovered_drawdown() -> None:
    values = pd.Series([100.0, 95.0, 90.0, 92.0])

    durations = calculate_drawdown_durations(values)

    assert durations["max_drawdown_duration"] == 3
    assert durations["current_drawdown_duration"] == 3
    assert durations["average_drawdown_duration"] == 3.0


def test_equal_weight_vector_concentration_metrics() -> None:
    weights = pd.Series([0.25, 0.25, 0.25, 0.25], index=list("ABCD"))

    metrics = calculate_concentration_metrics(weights)

    assert np.isclose(metrics["hhi"], 0.25)
    assert np.isclose(metrics["effective_n"], 4.0)
    assert np.isclose(metrics["max_weight"], 0.25)


def test_concentrated_weight_vector_has_low_effective_n() -> None:
    weights = pd.Series([0.90, 0.10, 0.0, 0.0], index=list("ABCD"))

    metrics = calculate_concentration_metrics(weights)

    assert metrics["effective_n"] < 2.0
    assert np.isclose(metrics["top_5_weight_sum"], 1.0)


def test_weight_history_returns_latest_and_average_concentration() -> None:
    weights = pd.DataFrame(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.70, 0.10, 0.10, 0.10],
        ],
        columns=list("ABCD"),
    )

    metrics = calculate_concentration_metrics(weights)

    assert metrics["latest_effective_n"] < metrics["average_effective_n"]
    assert np.isclose(metrics["average_hhi"], (0.25 + 0.52) / 2.0)


def test_combined_active_risk_metrics_contains_requested_keys() -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    benchmark = pd.Series([0.002, -0.001, 0.003, 0.000, 0.001, -0.002], index=dates)
    strategy = benchmark + 0.0005
    values = (1.0 + strategy).cumprod()
    weights = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])

    metrics = calculate_active_risk_metrics(
        strategy,
        benchmark,
        strategy_values=values,
        weights=weights,
        strategy_cagr=0.10,
        benchmark_cagr=0.08,
    )

    for key in (
        "simple_alpha",
        "jensen_alpha_annualized",
        "beta",
        "tracking_error",
        "information_ratio",
        "hit_ratio",
        "max_drawdown_duration",
        "hhi",
        "effective_n",
    ):
        assert key in metrics
    assert np.isclose(metrics["simple_alpha"], 0.02)
