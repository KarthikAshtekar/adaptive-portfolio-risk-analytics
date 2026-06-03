"""Unit tests for Phase 1 data ingestion and preprocessing."""

import types

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.ingest import (
    MarketDataBundle,
    YahooFinanceProvider,
    YFinanceIngester,
    build_data_inspection_table,
)
from src.data_pipeline.preprocess import DataPreprocessor


def _fake_download_multi(*args, **kwargs):
    _ = (args, kwargs)
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    columns = pd.MultiIndex.from_product(
        [["Adj Close", "Close", "Volume"], ["SPY", "QQQ"]]
    )
    data = np.array(
        [
            [100.0, 200.0, 99.0, 199.0, 1000, 2000],
            [101.0, 201.0, 100.0, 200.0, 1010, 2010],
            [102.0, 202.0, 101.0, 201.0, 1020, 2020],
            [103.0, 203.0, 102.0, 202.0, 1030, 2030],
            [104.0, 204.0, 103.0, 203.0, 1040, 2040],
        ]
    )
    return pd.DataFrame(data, index=dates, columns=columns)


def _fake_download_single(*args, **kwargs):
    _ = (args, kwargs)
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    return pd.DataFrame(
        {
            "Close": [10.0, 10.5, 10.75],
            "Volume": [100, 120, 130],
        },
        index=dates,
    )


def test_yahoo_finance_provider_returns_prices_and_volume(monkeypatch) -> None:
    fake_module = types.SimpleNamespace(download=_fake_download_multi)
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_module)

    provider = YahooFinanceProvider()
    market_data = provider.get_market_data(["SPY", "QQQ"], "2020-01-01", "2020-02-01")

    assert market_data.price_field == "Adj Close"
    assert list(market_data.prices_df.columns) == ["QQQ", "SPY"]
    assert list(market_data.volume_df.columns) == ["QQQ", "SPY"]
    assert market_data.prices_df.shape == (5, 2)
    assert market_data.volume_df.shape == (5, 2)


def test_legacy_fetch_still_returns_prices_only(monkeypatch) -> None:
    fake_module = types.SimpleNamespace(download=_fake_download_multi)
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_module)

    ingester = YFinanceIngester()
    prices = ingester.fetch(["SPY", "QQQ"], "2020-01-01", "2020-02-01")

    assert isinstance(prices, pd.DataFrame)
    assert list(prices.columns) == ["QQQ", "SPY"]
    assert prices.shape == (5, 2)


def test_provider_falls_back_to_close_when_adjusted_close_missing(monkeypatch) -> None:
    fake_module = types.SimpleNamespace(download=_fake_download_single)
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_module)

    provider = YahooFinanceProvider()
    market_data = provider.get_market_data(["goldbees.ns"], "2020-01-01", "2020-02-01")

    assert market_data.price_field == "Close"
    assert list(market_data.prices_df.columns) == ["GOLDBEES.NS"]
    assert list(market_data.volume_df.columns) == ["GOLDBEES.NS"]


def test_build_data_inspection_table_captures_missing_values() -> None:
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    market_data = MarketDataBundle(
        prices_df=pd.DataFrame(
            {"AAA": [100.0, np.nan, 102.0], "BBB": [50.0, 51.0, 52.0]},
            index=dates,
        ),
        volume_df=pd.DataFrame(
            {"AAA": [1000, 1100, np.nan], "BBB": [500, 550, 600]},
            index=dates,
        ),
        raw_data=pd.DataFrame(),
        price_field="Adj Close",
    )

    inspection = build_data_inspection_table(market_data)

    assert inspection.loc["AAA", "start_date"] == "2020-01-01"
    assert inspection.loc["AAA", "end_date"] == "2020-01-03"
    assert inspection.loc["AAA", "missing_prices"] == 1
    assert inspection.loc["AAA", "missing_volume"] == 1
    assert inspection.loc["BBB", "missing_prices"] == 0
    assert bool(inspection.loc["BBB", "dates_monotonic_increasing"]) is True


def test_preprocessor_returns_generation() -> None:
    prices = pd.DataFrame(
        {
            "A": [100, 101, 102, 103],
            "B": [200, 202, 204, 206],
        },
        index=pd.date_range("2020-01-01", periods=4, freq="B"),
    )

    simple = DataPreprocessor.calculate_returns(prices, method="simple")
    log = DataPreprocessor.calculate_returns(prices, method="log")

    assert simple.shape[0] == 3
    assert log.shape[0] == 3
    assert np.all(np.isfinite(simple.values))
    assert np.all(np.isfinite(log.values))


def test_preprocessor_missing_values() -> None:
    df = pd.DataFrame(
        {"A": [1.0, np.nan, 3.0], "B": [2.0, 2.5, np.nan]},
        index=pd.date_range("2020-01-01", periods=3, freq="B"),
    )

    filled = DataPreprocessor.handle_missing_values(df, method="forward_fill")
    assert not filled.isna().any().any()


def test_preprocessor_invalid_method() -> None:
    prices = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(ValueError):
        DataPreprocessor.calculate_returns(prices, method="invalid")
