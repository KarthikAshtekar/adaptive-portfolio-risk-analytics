"""Adaptive volatility-targeting overlay utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolatilityTargetingConfig:
    realized_vol_window: int = 63
    regime_lookback_window: int = 252
    base_target_vol: float = 0.10
    calm_target_vol: float = 0.12
    normal_target_vol: float = 0.10
    stress_target_vol: float = 0.06
    crisis_target_vol: float = 0.03
    calm_percentile: float = 0.40
    stress_percentile: float = 0.80
    crisis_percentile: float = 0.95
    exposure_floor: float = 0.25
    exposure_cap: float = 1.0
    no_trade_band: float = 0.05
    periods_per_year: int = 252


def compute_realized_volatility(
    returns,
    window: int = 63,
    periods_per_year: int = 252,
) -> pd.Series:
    """Compute annualized rolling realized volatility."""
    returns_s = _ensure_series(returns, "returns")
    if window <= 1:
        raise ValueError("window must be greater than 1")
    realized_vol = returns_s.rolling(window=window).std(ddof=1) * np.sqrt(periods_per_year)
    realized_vol.name = "realized_volatility"
    return realized_vol


def classify_volatility_regime(
    realized_vol,
    lookback_window: int = 252,
    thresholds: dict[str, float] | VolatilityTargetingConfig | None = None,
) -> pd.Series:
    """Classify volatility regimes using only information available up to t-1."""
    realized_vol_s = _ensure_series(realized_vol, "realized_vol")
    if lookback_window <= 1:
        raise ValueError("lookback_window must be greater than 1")

    if isinstance(thresholds, VolatilityTargetingConfig):
        calm_percentile = thresholds.calm_percentile
        stress_percentile = thresholds.stress_percentile
        crisis_percentile = thresholds.crisis_percentile
    else:
        threshold_map = dict(thresholds or {})
        calm_percentile = float(threshold_map.get("calm_percentile", 0.40))
        stress_percentile = float(threshold_map.get("stress_percentile", 0.80))
        crisis_percentile = float(threshold_map.get("crisis_percentile", 0.95))

    lagged_vol = realized_vol_s.shift(1)
    regimes: list[str] = []

    for idx in range(len(lagged_vol)):
        current_vol = lagged_vol.iloc[idx]
        if not np.isfinite(current_vol):
            regimes.append("normal")
            continue

        history = lagged_vol.iloc[max(0, idx - lookback_window + 1) : idx + 1].dropna()
        if history.empty:
            regimes.append("normal")
            continue

        percentile = float((history <= current_vol).mean())
        if percentile <= calm_percentile:
            regimes.append("calm")
        elif percentile <= stress_percentile:
            regimes.append("normal")
        elif percentile <= crisis_percentile:
            regimes.append("stress")
        else:
            regimes.append("crisis")

    return pd.Series(regimes, index=realized_vol_s.index, name="regime")


def compute_adaptive_target_volatility(
    regime_series,
    config: VolatilityTargetingConfig,
) -> pd.Series:
    """Map volatility regimes to annualized target-volatility levels."""
    regimes = _ensure_series(regime_series, "regime_series").astype(str)
    mapping = {
        "calm": float(config.calm_target_vol),
        "normal": float(config.normal_target_vol),
        "stress": float(config.stress_target_vol),
        "crisis": float(config.crisis_target_vol),
    }
    target_vol = regimes.map(mapping).fillna(float(config.base_target_vol)).astype(float)
    target_vol.name = "target_volatility"
    return target_vol


def compute_exposure_series(
    realized_vol,
    target_vol,
    config: VolatilityTargetingConfig,
) -> pd.Series:
    """Compute exposure using lagged realized volatility and a no-trade band."""
    realized_vol_s = _ensure_series(realized_vol, "realized_vol")
    target_vol_s = _ensure_series(target_vol, "target_vol").reindex(realized_vol_s.index)
    lagged_realized_vol = realized_vol_s.shift(1)

    exposures: list[float] = []
    previous_exposure = 1.0

    for current_date in realized_vol_s.index:
        realized_value = lagged_realized_vol.loc[current_date]
        target_value = target_vol_s.loc[current_date]

        if (
            not np.isfinite(realized_value)
            or realized_value <= 0.0
            or not np.isfinite(target_value)
        ):
            new_exposure = 1.0
        else:
            raw_exposure = float(target_value) / float(realized_value)
            new_exposure = float(np.clip(raw_exposure, config.exposure_floor, config.exposure_cap))

        if abs(new_exposure - previous_exposure) < config.no_trade_band:
            smoothed_exposure = previous_exposure
        else:
            smoothed_exposure = new_exposure

        exposures.append(float(smoothed_exposure))
        previous_exposure = float(smoothed_exposure)

    return pd.Series(exposures, index=realized_vol_s.index, name="exposure")


def apply_volatility_targeting(
    risky_returns,
    defensive_returns,
    config: VolatilityTargetingConfig | None = None,
) -> dict[str, object]:
    """Apply adaptive volatility targeting to a base risky portfolio return stream."""
    config = config or VolatilityTargetingConfig()
    risky_returns_s = _ensure_series(risky_returns, "risky_returns").astype(float)
    defensive_returns_s = _ensure_series(defensive_returns, "defensive_returns").astype(float)

    if risky_returns_s.empty:
        raise ValueError("risky_returns must not be empty")

    defensive_aligned = defensive_returns_s.reindex(risky_returns_s.index).ffill().bfill()
    if defensive_aligned.isna().any():
        raise ValueError("defensive_returns could not be aligned to risky_returns without NaNs")
    defensive_aligned.name = defensive_returns_s.name or "defensive_returns"

    realized_vol = compute_realized_volatility(
        risky_returns_s,
        window=config.realized_vol_window,
        periods_per_year=config.periods_per_year,
    )
    regime_series = classify_volatility_regime(
        realized_vol,
        lookback_window=config.regime_lookback_window,
        thresholds=config,
    )
    target_volatility = compute_adaptive_target_volatility(regime_series, config)
    exposure_series = compute_exposure_series(realized_vol, target_volatility, config)

    targeted_returns = (
        exposure_series * risky_returns_s + (1.0 - exposure_series) * defensive_aligned
    )
    targeted_returns.name = "targeted_return"

    base_growth = (1.0 + risky_returns_s).cumprod()
    targeted_growth = (1.0 + targeted_returns).cumprod()

    diagnostics_df = pd.DataFrame(
        {
            "risky_return": risky_returns_s,
            "defensive_return": defensive_aligned,
            "realized_volatility": realized_vol,
            "regime": regime_series,
            "target_volatility": target_volatility,
            "exposure": exposure_series,
            "defensive_allocation": 1.0 - exposure_series,
            "targeted_return": targeted_returns,
            "base_growth": base_growth,
            "targeted_growth": targeted_growth,
        }
    )

    summary = {
        "average_exposure": float(exposure_series.mean()),
        "min_exposure": float(exposure_series.min()),
        "max_exposure": float(exposure_series.max()),
        "percent_time_calm": float((regime_series == "calm").mean()),
        "percent_time_normal": float((regime_series == "normal").mean()),
        "percent_time_stress": float((regime_series == "stress").mean()),
        "percent_time_crisis": float((regime_series == "crisis").mean()),
        "final_base_growth": float(base_growth.iloc[-1]),
        "final_targeted_growth": float(targeted_growth.iloc[-1]),
    }

    return {
        "targeted_returns": targeted_returns,
        "exposure_series": exposure_series,
        "realized_volatility": realized_vol,
        "regime_series": regime_series,
        "target_volatility": target_volatility,
        "defensive_returns": defensive_aligned,
        "risky_returns": risky_returns_s,
        "diagnostics_df": diagnostics_df,
        "summary": summary,
    }


def _ensure_series(values, name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        if not isinstance(values.index, pd.DatetimeIndex):
            raise ValueError(f"{name} index must be a DatetimeIndex")
        return values.sort_index()
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError(f"{name} must be a Series or single-column DataFrame")
        series = values.iloc[:, 0].copy()
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError(f"{name} index must be a DatetimeIndex")
        return series.sort_index()
    raise TypeError(f"{name} must be a pandas Series or single-column DataFrame")
