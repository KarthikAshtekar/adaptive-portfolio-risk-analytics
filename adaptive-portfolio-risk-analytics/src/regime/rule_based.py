"""Explainable rule-based market regime classification."""

from __future__ import annotations

import numpy as np
import pandas as pd

STATE_TABLE_COLUMNS = [
    "rolling_volatility",
    "volatility_percentile",
    "rolling_drawdown",
    "trend_126d",
    "momentum_63d",
    "average_correlation",
    "correlation_percentile",
]


def classify_rule_based_regime(
    features,
    calm_vol_pct: float = 0.40,
    stress_vol_pct: float = 0.80,
    crisis_vol_pct: float = 0.95,
    crisis_drawdown: float = -0.15,
    stress_drawdown: float = -0.08,
    negative_trend_threshold: float = -0.05,
) -> pd.Series:
    """Classify each date as Calm, Normal, Stress, Crisis, or Unknown.

    Crisis has priority over Stress, which has priority over Calm. Dates without
    enough volatility, drawdown, trend, and 21-day return history remain Unknown.
    Correlation is used when available but is not required for single-asset input.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if features.empty:
        return pd.Series(dtype="object", index=features.index, name="regime")
    if not 0.0 <= calm_vol_pct <= stress_vol_pct <= crisis_vol_pct <= 1.0:
        raise ValueError("volatility percentile thresholds must be ordered within [0, 1]")
    if crisis_drawdown >= stress_drawdown:
        raise ValueError("crisis_drawdown must be more negative than stress_drawdown")

    values = features.reindex(
        columns=[
            "volatility_percentile",
            "rolling_drawdown",
            "trend_126d",
            "benchmark_return_21d",
            "correlation_percentile",
        ]
    )
    required = values[
        [
            "volatility_percentile",
            "rolling_drawdown",
            "trend_126d",
            "benchmark_return_21d",
        ]
    ]
    known = required.notna().all(axis=1)

    crisis = (
        (values["volatility_percentile"] > crisis_vol_pct)
        | (values["rolling_drawdown"] <= crisis_drawdown)
        | (values["benchmark_return_21d"] <= -0.10)
    )
    stress = (
        (values["volatility_percentile"] > stress_vol_pct)
        | (values["rolling_drawdown"] <= stress_drawdown)
        | (values["trend_126d"] <= negative_trend_threshold)
        | (values["correlation_percentile"] > 0.80)
    )
    calm = (
        (values["volatility_percentile"] <= calm_vol_pct)
        & (values["rolling_drawdown"] > -0.05)
        & (values["trend_126d"] >= 0.0)
    )

    labels = np.select(
        [known & crisis, known & stress, known & calm, known],
        ["Crisis", "Stress", "Calm", "Normal"],
        default="Unknown",
    )
    return pd.Series(labels, index=features.index, dtype="object", name="regime")


def calculate_regime_state_table(
    features: pd.DataFrame,
    regimes: pd.Series,
) -> pd.DataFrame:
    """Combine regime labels with their principal explanatory feature values."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if not isinstance(regimes, pd.Series):
        raise TypeError("regimes must be a pandas Series")

    table = features.reindex(columns=STATE_TABLE_COLUMNS).copy()
    table.insert(0, "regime", regimes.reindex(table.index))
    table.insert(0, "date", table.index)
    return table.reset_index(drop=True)


def lag_regime_labels(regimes: pd.Series, lag: int = 1) -> pd.Series:
    """Shift observed labels so a state detected at t is used at t + ``lag``."""
    if not isinstance(regimes, pd.Series):
        raise TypeError("regimes must be a pandas Series")
    if int(lag) < 0:
        raise ValueError("lag must be non-negative")

    lagged = regimes.shift(int(lag)).fillna("Unknown").astype("object")
    lagged.name = "decision_regime"
    return lagged
