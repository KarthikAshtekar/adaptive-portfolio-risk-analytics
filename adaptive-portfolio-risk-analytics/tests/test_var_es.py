"""Tests for historical VaR, ES, and VaR exceptions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics import (
    calculate_historical_es,
    calculate_historical_var,
    calculate_var_exceptions,
)


def test_historical_var_known_percentile_and_currency_amount() -> None:
    returns = pd.Series([-0.10, -0.05, 0.00, 0.05, 0.10])

    result = calculate_historical_var(
        returns,
        confidence_level=0.80,
        portfolio_value=1_000_000.0,
    )

    assert np.isclose(result["var_return"], 0.06)
    assert np.isclose(result["var_amount"], 60_000.0)
    assert np.isclose(result["tail_probability"], 0.20)


def test_historical_var_drops_nans() -> None:
    returns = pd.Series([-0.10, np.nan, -0.05, 0.00, 0.05, 0.10])

    result = calculate_historical_var(returns, confidence_level=0.80)

    assert np.isclose(result["var_return"], 0.06)


def test_empty_historical_var_returns_nan_safely() -> None:
    result = calculate_historical_var(pd.Series(dtype=float))

    assert np.isnan(result["var_return"])
    assert np.isnan(result["var_amount"])


def test_higher_confidence_generally_increases_var() -> None:
    returns = pd.Series(np.linspace(-0.10, 0.10, 101))

    var_95 = calculate_historical_var(returns, confidence_level=0.95)["var_return"]
    var_99 = calculate_historical_var(returns, confidence_level=0.99)["var_return"]

    assert var_99 >= var_95


def test_expected_shortfall_is_greater_than_or_equal_to_var() -> None:
    returns = pd.Series([-0.10, -0.05, 0.00, 0.05, 0.10])

    var_result = calculate_historical_var(returns, confidence_level=0.80)
    es_result = calculate_historical_es(returns, confidence_level=0.80)

    assert es_result["es_return"] >= var_result["var_return"]


def test_expected_shortfall_currency_amount() -> None:
    returns = pd.Series([-0.10, -0.05, 0.00, 0.05, 0.10])

    result = calculate_historical_es(
        returns,
        confidence_level=0.80,
        portfolio_value=1_000_000.0,
    )

    assert np.isclose(result["es_return"], 0.10)
    assert np.isclose(result["es_amount"], 100_000.0)


def test_static_var_exceptions_known_breaches() -> None:
    returns = pd.Series([-0.10, -0.05, 0.01, 0.02, 0.03])

    result = calculate_var_exceptions(
        returns,
        confidence_level=0.80,
        var_threshold=0.04,
    )

    assert result["actual_exceptions"] == 2
    assert np.isclose(result["expected_exceptions"], 1.0)
    assert np.isclose(result["exception_ratio"], 2.0)


def test_empty_var_exceptions_are_safe() -> None:
    result = calculate_var_exceptions(pd.Series(dtype=float))

    assert result["actual_exceptions"] == 0
    assert result["n_observations"] == 0
    assert np.isnan(result["exception_ratio"])


def test_rolling_var_exceptions_use_lagged_threshold() -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    returns = pd.Series([-0.01, -0.02, -0.03, -0.04, -0.05, -0.20], index=dates)

    result = calculate_var_exceptions(
        returns,
        confidence_level=0.80,
        rolling_window=3,
    )

    assert result["n_observations"] == 3
    assert result["actual_exceptions"] == 3
