"""Unit tests for Phase 1 data ingestion and preprocessing."""

import logging
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
from src.data_pipeline.preprocess import DataPreprocessor, DataQualityProcessor


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


def test_preprocessor_forward_fill_applies_to_internal_gaps() -> None:
    dates = pd.date_range("2020-01-01", periods=20, freq="B")
    df = pd.DataFrame(
        {
            "A": [float(i) if i != 10 else np.nan for i in range(20)],
            "B": [float(i) for i in range(20)],
        },
        index=dates,
    )


def _anomalous_price_fixture() -> pd.DataFrame:
    dates = pd.date_range("2019-12-16", periods=7, freq="B")
    return pd.DataFrame(
        {
            "GOLDBEES.NS": [100.0, 101.0, 102.0, 0.01, 0.02, 105.0, 106.0],
            "SPY": [300.0, 301.0, 302.0, 303.0, 304.0, 305.0, 306.0],
        },
        index=dates,
    )


def _outlier_returns_fixture() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    asset_a = [0.01, 0.012, 0.015, 0.013, 0.011, 0.014, 0.012, 0.013, 0.011, 0.014,
               0.012, 0.013, 0.011, 0.014, 0.012, 0.013, 0.011, 0.014, 0.012, 50.0]
    return pd.DataFrame(
        {
            "AssetA": asset_a,
            "AssetB": [
                0.00,
                -0.005,
                0.004,
                0.003,
                -0.002,
                0.001,
                0.000,
                -0.003,
                0.002,
                0.001,
                -0.001,
                0.003,
                0.000,
                -0.002,
                0.001,
                0.002,
                -0.001,
                0.000,
                0.001,
                -0.002,
            ],
        },
        index=dates,
    )

    cleaned, summary = DataPreprocessor.handle_missing_values(df)

    assert cleaned.loc[dates[10], "A"] == cleaned.loc[dates[9], "A"]
    assert summary.assets_dropped == 0
    assert summary.missing_after == 0
    assert not cleaned.isna().any().any()


def test_preprocessor_back_fill_applies_to_leading_gaps() -> None:
    dates = pd.date_range("2020-01-01", periods=20, freq="B")
    df = pd.DataFrame(
        {
            "A": [np.nan] + [float(i) for i in range(1, 20)],
            "B": [float(i) for i in range(20)],
        },
        index=dates,
    )

    cleaned, summary = DataPreprocessor.handle_missing_values(df)

    assert cleaned.loc[dates[0], "A"] == cleaned.loc[dates[1], "A"]
    assert summary.missing_after == 0
    assert not cleaned.isna().any().any()


def test_preprocessor_drops_assets_above_missing_threshold() -> None:
    dates = pd.date_range("2020-01-01", periods=20, freq="B")
    df = pd.DataFrame(
        {
            "A": [np.nan, np.nan] + [float(i) for i in range(2, 20)],
            "B": [float(i) for i in range(20)],
        },
        index=dates,
    )

    cleaned, summary = DataPreprocessor.handle_missing_values(df)

    assert list(cleaned.columns) == ["B"]
    assert summary.assets_dropped == 1
    assert summary.dropped_asset_names == ("A",)


def test_preprocessor_logs_dropped_assets(caplog: pytest.LogCaptureFixture) -> None:
    dates = pd.date_range("2020-01-01", periods=20, freq="B")
    df = pd.DataFrame(
        {
            "A": [np.nan, np.nan] + [float(i) for i in range(2, 20)],
            "B": [float(i) for i in range(20)],
        },
        index=dates,
    )

    with caplog.at_level(logging.WARNING):
        DataPreprocessor.handle_missing_values(df)

    assert "Dropped assets with more than 5% missing observations" in caplog.text
    assert "A (10.00%)" in caplog.text


def test_preprocessor_no_remaining_nans_after_cleaning() -> None:
    dates = pd.date_range("2020-01-01", periods=20, freq="B")
    df = pd.DataFrame(
        {
            "A": [np.nan] + [float(i) for i in range(1, 20)],
            "B": [float(i) if i != 8 else np.nan for i in range(20)],
        },
        index=dates,
    )

    cleaned, summary = DataPreprocessor.handle_missing_values(df)

    assert summary.missing_after == 0
    assert int(cleaned.isna().sum().sum()) == 0


def test_preprocessor_invalid_method() -> None:
    prices = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(ValueError):
        DataPreprocessor.calculate_returns(prices, method="invalid")


def test_price_anomaly_detection_flags_extreme_log_returns() -> None:
    prices = _anomalous_price_fixture()
    processor = DataQualityProcessor()

    report = processor.detect_price_anomalies(prices)

    assert list(report.columns) == ["date", "asset", "price", "log_return"]
    assert len(report) == 3
    assert set(report["asset"]) == {"GOLDBEES.NS"}
    assert report["date"].tolist() == list(prices.index[3:6])
    assert (report["log_return"].abs() > 0.50).all()


def test_price_repair_iteratively_eliminates_threshold_violations() -> None:
    prices = _anomalous_price_fixture()
    processor = DataQualityProcessor()

    clean_prices, repair_report = processor.repair_price_anomalies(prices)
    cleaned_anomalies = processor.detect_price_anomalies(clean_prices)
    repaired_log_returns = DataPreprocessor.calculate_returns(clean_prices, method="log")

    assert clean_prices.shape == prices.shape
    assert repair_report.shape[0] == 3
    assert not clean_prices.isna().any().any()
    assert cleaned_anomalies.empty
    assert repaired_log_returns.abs().max().max() <= 0.50


def test_return_outlier_detection_mad_flags_extreme_observation() -> None:
    returns = _outlier_returns_fixture()
    processor = DataQualityProcessor()

    report = processor.detect_return_outliers(returns, method="mad")

    assert list(report.columns) == ["date", "asset", "return", "score", "method"]
    assert not report.empty
    assert report["method"].eq("mad").all()
    assert report["asset"].tolist() == ["AssetA"]
    assert report["date"].tolist() == [returns.index[-1]]


def test_return_outlier_detection_zscore_flags_extreme_observation() -> None:
    returns = _outlier_returns_fixture()
    processor = DataQualityProcessor()

    report = processor.detect_return_outliers(returns, method="zscore")

    assert list(report.columns) == ["date", "asset", "return", "score", "method"]
    assert not report.empty
    assert report["method"].eq("zscore").all()
    assert report["asset"].tolist() == ["AssetA"]
    assert report["date"].tolist() == [returns.index[-1]]


def test_winsorization_clips_returns_and_reports_changes() -> None:
    returns = pd.DataFrame(
        {
            "AssetA": [-0.35, 0.10, 0.25],
            "AssetB": [0.05, -0.30, 0.15],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="B"),
    )
    processor = DataQualityProcessor()

    stabilized_returns, stabilization_report = processor.stabilize_returns(returns)

    assert stabilized_returns.shape == returns.shape
    assert stabilization_report.loc[0, "method"] == "winsorize"
    assert stabilization_report.loc[0, "num_clipped"] == 3
    assert stabilization_report.loc[0, "affected_assets"] == ("AssetA", "AssetB")
    assert stabilized_returns.min().min() >= -0.20
    assert stabilized_returns.max().max() <= 0.20


def test_pipeline_integration_generates_quality_reports_and_stabilized_returns() -> None:
    prices = _anomalous_price_fixture()

    outputs = DataPreprocessor.build_returns_risk_outputs(
        prices,
        periods_per_year=252,
        rolling_windows=(2, 3),
    )

    assert outputs.simple_returns_df.shape == outputs.log_returns_df.shape == outputs.returns_df.shape
    assert outputs.anomaly_report_df.shape[1] == 4
    assert outputs.repair_report_df.shape[1] == 6
    assert outputs.outlier_report_df.shape[1] == 5
    assert outputs.stabilization_report_df.shape[1] == 5
    assert not outputs.quality_report_df.empty
    assert outputs.quality_report_df.loc[0, "price_anomalies_detected"] == len(
        outputs.anomaly_report_df
    )
    assert outputs.quality_report_df.loc[0, "price_repairs_applied"] == len(outputs.repair_report_df)
    assert outputs.quality_report_df.loc[0, "return_outliers_detected"] == len(
        outputs.outlier_report_df
    )
    assert outputs.returns_df.isna().sum().sum() == 0
    assert outputs.returns_df.abs().max().max() <= 0.20 + 1e-12
    assert outputs.volatility_summary_df.shape[0] == outputs.returns_df.shape[1]
