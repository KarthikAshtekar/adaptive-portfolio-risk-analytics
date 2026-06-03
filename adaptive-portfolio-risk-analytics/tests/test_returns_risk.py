"""Unit tests for Stage 2 returns and risk analytics."""

import numpy as np
import pandas as pd

from src.data_pipeline.preprocess import DataPreprocessor


def _price_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AssetA": [100.0, 110.0, 121.0, 133.1],
            "AssetB": [200.0, 190.0, 199.5, 209.475],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="B"),
    )


def test_simple_returns_are_calculated_correctly() -> None:
    prices = _price_fixture()

    simple_returns = DataPreprocessor.calculate_returns(prices, method="simple")

    expected = pd.DataFrame(
        {
            "AssetA": [0.10, 0.10, 0.10],
            "AssetB": [-0.05, 0.05, 0.05],
        },
        index=prices.index[1:],
    )

    pd.testing.assert_frame_equal(simple_returns, expected)


def test_log_returns_are_calculated_correctly() -> None:
    prices = _price_fixture()

    log_returns = DataPreprocessor.calculate_returns(prices, method="log")

    expected = pd.DataFrame(
        {
            "AssetA": [np.log(1.10), np.log(1.10), np.log(1.10)],
            "AssetB": [np.log(0.95), np.log(1.05), np.log(1.05)],
        },
        index=prices.index[1:],
    )

    pd.testing.assert_frame_equal(log_returns, expected)


def test_first_observation_is_dropped_without_lookahead_bias() -> None:
    prices = _price_fixture()

    simple_returns = DataPreprocessor.calculate_returns(prices, method="simple")

    assert simple_returns.index[0] == prices.index[1]
    assert simple_returns.loc[prices.index[1], "AssetA"] == (110.0 / 100.0) - 1.0
    assert simple_returns.loc[prices.index[2], "AssetA"] == (121.0 / 110.0) - 1.0


def test_volatility_summary_uses_sample_standard_deviation() -> None:
    returns = pd.DataFrame(
        {
            "AssetA": [0.01, 0.02, 0.03, 0.04],
            "AssetB": [0.02, 0.01, 0.00, -0.01],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="B"),
    )

    summary = DataPreprocessor.calculate_volatility(returns, periods_per_year=252)

    expected_daily = returns.std(ddof=1)
    expected_annualized = expected_daily * np.sqrt(252)

    pd.testing.assert_series_equal(summary["daily_volatility"], expected_daily, check_names=False)
    pd.testing.assert_series_equal(
        summary["annualized_volatility"],
        expected_annualized,
        check_names=False,
    )
    assert (summary["annualization_factor"] == 252).all()


def test_annualization_logic_scales_daily_volatility() -> None:
    returns = pd.DataFrame(
        {"AssetA": [0.01, 0.02, 0.03]},
        index=pd.date_range("2024-01-01", periods=3, freq="B"),
    )

    summary = DataPreprocessor.calculate_volatility(returns, periods_per_year=12)

    expected_daily = returns["AssetA"].std(ddof=1)
    expected_annualized = expected_daily * np.sqrt(12)

    assert np.isclose(summary.loc["AssetA", "daily_volatility"], expected_daily)
    assert np.isclose(summary.loc["AssetA", "annualized_volatility"], expected_annualized)


def test_rolling_volatility_is_calculated_for_each_window() -> None:
    returns = pd.DataFrame(
        {"AssetA": [0.01, 0.02, 0.03, 0.04]},
        index=pd.date_range("2024-01-01", periods=4, freq="B"),
    )

    rolling = DataPreprocessor.calculate_rolling_volatility(
        returns,
        windows=(2, 3),
        periods_per_year=252,
    )

    expected_2d_last = returns["AssetA"].iloc[2:4].std(ddof=1) * np.sqrt(252)
    expected_3d_last = returns["AssetA"].iloc[1:4].std(ddof=1) * np.sqrt(252)

    assert np.isnan(rolling.loc[returns.index[0], ("2d", "AssetA")])
    assert np.isclose(rolling.loc[returns.index[3], ("2d", "AssetA")], expected_2d_last)
    assert np.isclose(rolling.loc[returns.index[3], ("3d", "AssetA")], expected_3d_last)


def test_stage2_outputs_default_to_log_returns_for_downstream_use() -> None:
    prices = _price_fixture()

    outputs = DataPreprocessor.build_returns_risk_outputs(
        prices,
        periods_per_year=252,
        rolling_windows=(2, 3),
    )

    pd.testing.assert_frame_equal(outputs.returns_df, outputs.log_returns_df)
    assert list(outputs.volatility_summary_df.columns) == [
        "daily_volatility",
        "annualized_volatility",
        "annualization_factor",
    ]
    assert set(outputs.rolling_volatility_df.columns.get_level_values("window")) == {"2d", "3d"}
