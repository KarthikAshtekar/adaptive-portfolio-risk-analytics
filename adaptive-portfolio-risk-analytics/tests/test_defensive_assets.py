"""Tests for defensive sleeve data loading."""

from __future__ import annotations

import types

import numpy as np
import pandas as pd

from src.data_pipeline import get_defensive_asset_returns


def _fake_download_defensive(*args, **kwargs):
    _ = (args, kwargs)
    dates = pd.date_range("2022-01-03", periods=5, freq="B")
    columns = pd.MultiIndex.from_product([["Adj Close", "Volume"], ["LIQUIDBEES.NS"]])
    data = np.array(
        [
            [np.nan, 1000],
            [100.0, 1010],
            [101.0, 1020],
            [np.nan, 1030],
            [102.0, 1040],
        ]
    )
    return pd.DataFrame(data, index=dates, columns=columns)


def test_synthetic_defensive_returns_are_generated_correctly() -> None:
    defensive_returns, metadata = get_defensive_asset_returns(
        start_date="2022-01-03",
        end_date="2022-01-10",
        preferred_ticker=None,
        fallback_tickers=[],
        synthetic_annual_rate=0.04,
    )

    assert isinstance(defensive_returns, pd.Series)
    assert metadata["selected_mode"] == "synthetic"
    assert np.isclose(
        defensive_returns.iloc[0],
        (1.04 ** (1.0 / 252.0)) - 1.0,
    )
    assert defensive_returns.notna().all()


def test_defensive_returns_are_series_without_nans(monkeypatch) -> None:
    fake_module = types.SimpleNamespace(download=_fake_download_defensive)
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_module)

    defensive_returns, metadata = get_defensive_asset_returns(
        start_date="2022-01-03",
        end_date="2022-01-10",
        preferred_ticker="LIQUIDBEES.NS",
    )

    assert isinstance(defensive_returns, pd.Series)
    assert defensive_returns.notna().all()
    assert metadata["selected_mode"] == "ticker"
    assert metadata["selected_ticker"] == "LIQUIDBEES.NS"


def test_defensive_ticker_missing_values_are_filled(monkeypatch) -> None:
    fake_module = types.SimpleNamespace(download=_fake_download_defensive)
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_module)

    defensive_returns, metadata = get_defensive_asset_returns(
        start_date="2022-01-03",
        end_date="2022-01-10",
        preferred_ticker="LIQUIDBEES.NS",
    )

    assert metadata["missing_before"] > 0
    assert metadata["missing_after"] == 0
    assert defensive_returns.index.is_monotonic_increasing


def test_defensive_ticker_is_not_dropped_due_to_missing_values(monkeypatch) -> None:
    fake_module = types.SimpleNamespace(download=_fake_download_defensive)
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_module)

    defensive_returns, metadata = get_defensive_asset_returns(
        start_date="2022-01-03",
        end_date="2022-01-10",
        preferred_ticker="LIQUIDBEES.NS",
    )

    assert not defensive_returns.empty
    assert metadata["selected_ticker"] == "LIQUIDBEES.NS"
    assert metadata["fallback_used"] is False
