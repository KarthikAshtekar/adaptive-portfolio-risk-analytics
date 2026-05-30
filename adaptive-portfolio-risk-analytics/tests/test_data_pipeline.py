"""Unit tests for Phase 1 data ingestion and preprocessing."""

import types

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.ingest import YFinanceIngester
from src.data_pipeline.preprocess import DataPreprocessor


def _fake_download(*args, **kwargs):
    _ = (args, kwargs)
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    cols = pd.MultiIndex.from_product([["Close", "Open"], ["SPY", "QQQ"]])
    data = np.array(
        [
            [100, 200, 99, 199],
            [101, 201, 100, 200],
            [102, 202, 101, 201],
            [103, 203, 102, 202],
            [104, 204, 103, 203],
        ]
    )
    return pd.DataFrame(data, index=dates, columns=cols)


def test_yfinance_ingester_extracts_close(monkeypatch) -> None:
    fake_module = types.SimpleNamespace(download=_fake_download)
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_module)

    ingester = YFinanceIngester()
    prices = ingester.fetch(["SPY", "QQQ"], "2020-01-01", "2020-02-01")

    assert list(prices.columns) == ["QQQ", "SPY"]
    assert prices.shape == (5, 2)


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
