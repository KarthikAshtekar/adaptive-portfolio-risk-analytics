"""Tests for Phase 3B market regime feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.regime import calculate_regime_features

EXPECTED_COLUMNS = {
    "rolling_volatility",
    "volatility_percentile",
    "rolling_drawdown",
    "trend_126d",
    "momentum_63d",
    "average_correlation",
    "correlation_percentile",
    "benchmark_return_21d",
    "benchmark_volatility_63d",
}


def _returns(periods: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(20260618)
    index = pd.date_range("2022-01-03", periods=periods, freq="B")
    common = rng.normal(0.0003, 0.006, size=periods)
    return pd.DataFrame(
        {
            "A": common + rng.normal(0.0, 0.004, size=periods),
            "B": common + rng.normal(0.0, 0.005, size=periods),
            "C": common + rng.normal(0.0, 0.006, size=periods),
        },
        index=index,
    )


def test_regime_features_returns_expected_columns() -> None:
    returns = _returns()
    benchmark = returns.mean(axis=1)

    features = calculate_regime_features(returns, benchmark_returns=benchmark)

    assert EXPECTED_COLUMNS.issubset(features.columns)
    assert features.index.equals(returns.index)
    assert features["rolling_volatility"].dropna().ge(0.0).all()


def test_regime_features_handles_missing_optional_inputs() -> None:
    features = calculate_regime_features(_returns())

    assert not features.empty
    assert "gold_equity_spread_63d" not in features
    assert "silver_equity_spread_63d" not in features
    assert features["benchmark_return_21d"].notna().any()


def test_optional_gold_and_silver_spreads_are_added() -> None:
    returns = _returns()
    gold = returns["A"] * 0.5
    silver = returns["B"] * 0.75

    features = calculate_regime_features(
        returns,
        gold_returns=gold,
        silver_returns=silver,
    )

    assert "gold_equity_spread_63d" in features
    assert "silver_equity_spread_63d" in features
    assert features["gold_equity_spread_63d"].notna().any()


def test_future_returns_do_not_change_past_features() -> None:
    returns = _returns()
    altered = returns.copy()
    altered.iloc[-20:] = altered.iloc[-20:] * 10.0

    original_features = calculate_regime_features(returns)
    altered_features = calculate_regime_features(altered)

    pd.testing.assert_frame_equal(
        original_features.iloc[:-20],
        altered_features.iloc[:-20],
    )


def test_small_sample_is_returned_safely_with_warmup_nans() -> None:
    returns = _returns(periods=10)

    features = calculate_regime_features(returns)

    assert len(features) == 10
    assert features["rolling_volatility"].isna().all()
    assert features["trend_126d"].isna().all()


def test_average_correlation_works_for_multi_asset_returns() -> None:
    returns = _returns()

    features = calculate_regime_features(returns, lookback_corr=21)

    average_correlation = features["average_correlation"].dropna()
    assert not average_correlation.empty
    assert average_correlation.between(-1.0, 1.0).all()
