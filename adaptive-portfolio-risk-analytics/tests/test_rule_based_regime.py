"""Tests for explainable rule-based market regimes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.regime import (
    calculate_regime_state_table,
    classify_rule_based_regime,
    lag_regime_labels,
)


def _feature_row(**overrides) -> pd.DataFrame:
    row = {
        "rolling_volatility": 0.15,
        "volatility_percentile": 0.50,
        "rolling_drawdown": -0.02,
        "trend_126d": 0.05,
        "momentum_63d": 0.03,
        "average_correlation": 0.40,
        "correlation_percentile": 0.50,
        "benchmark_return_21d": 0.01,
    }
    row.update(overrides)
    return pd.DataFrame([row], index=pd.date_range("2024-01-01", periods=1))


def test_high_volatility_percentile_maps_to_stress_or_crisis() -> None:
    stress = classify_rule_based_regime(_feature_row(volatility_percentile=0.90))
    crisis = classify_rule_based_regime(_feature_row(volatility_percentile=0.99))

    assert stress.iloc[0] == "Stress"
    assert crisis.iloc[0] == "Crisis"


def test_deep_drawdown_maps_to_crisis() -> None:
    regimes = classify_rule_based_regime(_feature_row(rolling_drawdown=-0.20))

    assert regimes.iloc[0] == "Crisis"


def test_positive_trend_and_low_volatility_map_to_calm() -> None:
    regimes = classify_rule_based_regime(
        _feature_row(
            volatility_percentile=0.30,
            rolling_drawdown=-0.01,
            trend_126d=0.10,
        )
    )

    assert regimes.iloc[0] == "Calm"


def test_missing_required_features_map_to_unknown() -> None:
    features = _feature_row()
    features.loc[:, "trend_126d"] = np.nan

    regimes = classify_rule_based_regime(features)

    assert regimes.iloc[0] == "Unknown"


def test_thresholds_are_configurable() -> None:
    features = _feature_row(volatility_percentile=0.70)

    default_regime = classify_rule_based_regime(features)
    configured_regime = classify_rule_based_regime(
        features,
        stress_vol_pct=0.60,
        crisis_vol_pct=0.90,
    )

    assert default_regime.iloc[0] == "Normal"
    assert configured_regime.iloc[0] == "Stress"


def test_lagged_regime_labels_shift_correctly() -> None:
    index = pd.date_range("2024-01-01", periods=4)
    regimes = pd.Series(
        ["Calm", "Normal", "Stress", "Crisis"],
        index=index,
        name="regime",
    )

    lagged = lag_regime_labels(regimes, lag=1)

    assert lagged.tolist() == ["Unknown", "Calm", "Normal", "Stress"]


def test_regime_state_table_contains_explanatory_columns() -> None:
    features = _feature_row()
    regimes = classify_rule_based_regime(features)

    table = calculate_regime_state_table(features, regimes)

    assert table.loc[0, "date"] == features.index[0]
    assert table.loc[0, "regime"] == "Normal"
    assert "average_correlation" in table.columns
