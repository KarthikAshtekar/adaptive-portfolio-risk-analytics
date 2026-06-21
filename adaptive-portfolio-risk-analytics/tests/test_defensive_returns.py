"""Tests for the centralized Phase 3E defensive-return contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.adaptive.defensive as defensive_module
from src.adaptive import get_defensive_returns


def test_synthetic_uses_compounded_daily_equivalent() -> None:
    index = pd.date_range("2024-01-02", periods=10, freq="B")

    result = get_defensive_returns(index, source="synthetic", annual_rate=0.04)

    expected = (1.04 ** (1.0 / 252.0)) - 1.0
    assert result.returns.eq(expected).all()
    assert result.metadata["defensive_source_used"] == "synthetic"
    assert result.metadata["defensive_fallback_used"] is False


def test_cash_zero_returns_are_zero() -> None:
    index = pd.date_range("2024-01-02", periods=10, freq="B")

    result = get_defensive_returns(index, source="cash_zero")

    assert result.returns.eq(0.0).all()
    assert result.source_used == "cash_zero"


def test_provided_series_is_aligned_and_missing_dates_are_safe() -> None:
    index = pd.date_range("2024-01-02", periods=5, freq="B")
    provided = pd.Series(
        [0.001, 0.002],
        index=[index[1], index[3]],
    )

    result = get_defensive_returns(
        index,
        source="provided_series",
        returns=provided,
    )

    assert result.returns.index.equals(index)
    assert result.returns.loc[index[1]] == pytest.approx(0.001)
    assert result.returns.loc[index[3]] == pytest.approx(0.002)
    assert result.returns.loc[index[[0, 2, 4]]].eq(0.0).all()


def test_missing_ticker_falls_back_with_metadata(monkeypatch) -> None:
    index = pd.date_range("2024-01-02", periods=10, freq="B")

    def fail_download(*args, **kwargs):
        _ = (args, kwargs)
        raise ValueError("ticker unavailable")

    monkeypatch.setattr(
        defensive_module,
        "_download_ticker_prices",
        fail_download,
    )
    result = get_defensive_returns(
        index,
        source="ticker",
        defensive_ticker="MISSING.NS",
        fallback="synthetic",
    )

    assert result.source_requested == "ticker"
    assert result.source_used == "synthetic"
    assert result.fallback_used is True
    assert result.ticker == "MISSING.NS"
    assert "ticker unavailable" in result.notes


def test_series_name_and_values_are_finite() -> None:
    index = pd.date_range("2024-01-02", periods=10, freq="B")
    result = get_defensive_returns(index)

    assert result.returns.name == "defensive_returns"
    assert np.isfinite(result.returns).all()
