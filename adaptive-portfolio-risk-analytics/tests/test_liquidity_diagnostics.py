"""Tests for liquidity diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics import calculate_liquidity_diagnostics, summarize_liquidity_diagnostics


def test_liquidity_adtv_and_participation_rate_calculation() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    prices = pd.DataFrame({"A": [100.0, 110.0, 120.0]}, index=dates)
    volumes = pd.DataFrame({"A": [1_000.0, 2_000.0, 3_000.0]}, index=dates)
    current_weights = pd.Series({"A": 0.10})
    target_weights = pd.Series({"A": 0.20})

    result = calculate_liquidity_diagnostics(
        prices,
        volumes,
        portfolio_value=1_000_000.0,
        current_weights=current_weights,
        target_weights=target_weights,
        lookback_days=2,
    )

    assert np.isclose(result.loc[0, "average_daily_volume"], 2_500.0)
    assert np.isclose(result.loc[0, "average_daily_traded_value"], 300_000.0)
    assert np.isclose(result.loc[0, "estimated_trade_value"], 100_000.0)
    assert np.isclose(result.loc[0, "participation_rate"], 1 / 3)


def test_missing_volume_returns_empty_frame_safely() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    prices = pd.DataFrame({"A": [100.0, 110.0, 120.0]}, index=dates)

    result = calculate_liquidity_diagnostics(prices, pd.DataFrame())

    assert result.empty


def test_high_participation_rate_warning_triggers() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    prices = pd.DataFrame({"A": [10.0, 10.0, 10.0]}, index=dates)
    volumes = pd.DataFrame({"A": [1_000.0, 1_000.0, 1_000.0]}, index=dates)
    weights = pd.Series({"A": 0.50})

    result = calculate_liquidity_diagnostics(
        prices,
        volumes,
        weights=weights,
        portfolio_value=1_000_000.0,
    )

    assert result.loc[0, "liquidity_warning"] == "High liquidity risk"


def test_low_liquidity_data_quality_warning_for_zero_adtv() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    prices = pd.DataFrame({"A": [10.0, 10.0, 10.0]}, index=dates)
    volumes = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=dates)
    weights = pd.Series({"A": 0.50})

    result = calculate_liquidity_diagnostics(
        prices,
        volumes,
        weights=weights,
        portfolio_value=1_000_000.0,
    )

    assert result.loc[0, "liquidity_warning"] == "Low liquidity data quality"


def test_liquidity_summary_counts_warning_types() -> None:
    diagnostics = pd.DataFrame(
        {
            "participation_rate": [0.01, 0.06, 0.20],
            "average_daily_traded_value": [1_000_000.0, 500_000.0, 100_000.0],
            "liquidity_warning": ["OK", "Moderate liquidity risk", "High liquidity risk"],
        }
    )

    summary = summarize_liquidity_diagnostics(diagnostics)

    assert summary["num_high_risk_assets"] == 1
    assert summary["num_moderate_risk_assets"] == 1
    assert np.isclose(summary["max_participation_rate"], 0.20)
