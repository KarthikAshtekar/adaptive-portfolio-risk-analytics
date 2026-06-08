"""Tests for adaptive volatility-targeting overlays."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtesting import (
    VolatilityTargetingConfig,
    apply_volatility_targeting,
    classify_volatility_regime,
    compute_exposure_series,
    compute_realized_volatility,
)


def _risky_returns() -> pd.Series:
    dates = pd.date_range("2021-01-01", periods=220, freq="B")
    low_vol = np.tile([0.0008, 0.0010, 0.0012, 0.0009, 0.0011], 22)
    high_vol = np.tile([0.04, -0.035, 0.03, -0.025, 0.02], 22)
    values = np.concatenate([low_vol, high_vol[:110]])
    return pd.Series(values, index=dates, name="risky")


def _defensive_returns(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(0.04 / 252.0, index=index, name="defensive")


def test_realized_volatility_is_positive_after_warmup() -> None:
    risky_returns = _risky_returns()
    realized_vol = compute_realized_volatility(risky_returns, window=21)

    assert (realized_vol.dropna() > 0.0).all()


def test_regime_classification_returns_valid_labels() -> None:
    risky_returns = _risky_returns()
    realized_vol = compute_realized_volatility(risky_returns, window=21)
    config = VolatilityTargetingConfig(realized_vol_window=21, regime_lookback_window=63)
    regimes = classify_volatility_regime(realized_vol, lookback_window=63, thresholds=config)

    assert set(regimes.unique()) <= {"calm", "normal", "stress", "crisis"}


def test_exposure_is_bounded_between_floor_and_cap() -> None:
    risky_returns = _risky_returns()
    config = VolatilityTargetingConfig(
        realized_vol_window=21,
        regime_lookback_window=63,
        exposure_floor=0.25,
        exposure_cap=1.0,
    )
    results = apply_volatility_targeting(risky_returns, _defensive_returns(risky_returns.index), config)
    exposure = results["exposure_series"]

    assert (exposure >= config.exposure_floor - 1e-12).all()
    assert (exposure <= config.exposure_cap + 1e-12).all()


def test_no_trade_band_reduces_exposure_changes() -> None:
    risky_returns = _risky_returns()
    realized_vol = compute_realized_volatility(risky_returns, window=21)
    config_raw = VolatilityTargetingConfig(realized_vol_window=21, no_trade_band=0.0)
    config_smooth = VolatilityTargetingConfig(realized_vol_window=21, no_trade_band=0.20)
    regimes = classify_volatility_regime(realized_vol, lookback_window=63, thresholds=config_raw)
    target_vol = pd.Series(config_raw.base_target_vol, index=risky_returns.index)

    exposure_raw = compute_exposure_series(realized_vol, target_vol, config_raw)
    exposure_smooth = compute_exposure_series(realized_vol, target_vol, config_smooth)

    changes_raw = exposure_raw.diff().abs().fillna(0.0)
    changes_smooth = exposure_smooth.diff().abs().fillna(0.0)

    assert changes_smooth.gt(0.0).sum() <= changes_raw.gt(0.0).sum()
    assert set(regimes.unique()) <= {"calm", "normal", "stress", "crisis"}


def test_apply_volatility_targeting_returns_expected_keys() -> None:
    risky_returns = _risky_returns()
    results = apply_volatility_targeting(
        risky_returns,
        _defensive_returns(risky_returns.index),
        VolatilityTargetingConfig(realized_vol_window=21, regime_lookback_window=63),
    )

    for key in (
        "targeted_returns",
        "exposure_series",
        "realized_volatility",
        "regime_series",
        "target_volatility",
        "defensive_returns",
        "risky_returns",
        "diagnostics_df",
        "summary",
    ):
        assert key in results


def test_targeted_returns_have_same_index_as_risky_returns() -> None:
    risky_returns = _risky_returns()
    results = apply_volatility_targeting(
        risky_returns,
        _defensive_returns(risky_returns.index),
        VolatilityTargetingConfig(realized_vol_window=21, regime_lookback_window=63),
    )

    assert results["targeted_returns"].index.equals(risky_returns.index)


def test_no_look_ahead_behavior_for_exposure_values() -> None:
    risky_returns = _risky_returns()
    altered_returns = risky_returns.copy()
    altered_returns.iloc[150:] = altered_returns.iloc[150:] * -2.0
    config = VolatilityTargetingConfig(realized_vol_window=21, regime_lookback_window=63)

    baseline = apply_volatility_targeting(
        risky_returns,
        _defensive_returns(risky_returns.index),
        config,
    )["exposure_series"]
    altered = apply_volatility_targeting(
        altered_returns,
        _defensive_returns(altered_returns.index),
        config,
    )["exposure_series"]

    assert baseline.iloc[:150].equals(altered.iloc[:150])


def test_defensive_returns_are_aligned_with_risky_returns() -> None:
    risky_returns = _risky_returns()
    defensive_index = risky_returns.index[5:]
    defensive_returns = _defensive_returns(defensive_index)
    results = apply_volatility_targeting(
        risky_returns,
        defensive_returns,
        VolatilityTargetingConfig(realized_vol_window=21, regime_lookback_window=63),
    )

    assert results["defensive_returns"].index.equals(risky_returns.index)
    assert results["defensive_returns"].notna().all()


def test_warmup_exposure_defaults_to_one() -> None:
    risky_returns = _risky_returns()
    config = VolatilityTargetingConfig(realized_vol_window=21, regime_lookback_window=63)
    exposure = apply_volatility_targeting(
        risky_returns,
        _defensive_returns(risky_returns.index),
        config,
    )["exposure_series"]

    assert exposure.iloc[: config.realized_vol_window].eq(1.0).all()


def test_adaptive_targeting_reduces_exposure_during_high_volatility() -> None:
    risky_returns = _risky_returns()
    config = VolatilityTargetingConfig(realized_vol_window=21, regime_lookback_window=63)
    exposure = apply_volatility_targeting(
        risky_returns,
        _defensive_returns(risky_returns.index),
        config,
    )["exposure_series"]

    low_vol_avg = float(exposure.iloc[40:100].mean())
    high_vol_avg = float(exposure.iloc[150:210].mean())
    assert high_vol_avg < low_vol_avg
