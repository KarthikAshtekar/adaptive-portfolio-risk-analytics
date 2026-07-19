"""Basic liquidity diagnostics using price, volume, ADTV, and participation rate."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_liquidity_diagnostics(
    prices,
    volumes,
    weights=None,
    portfolio_value: float | None = None,
    target_weights=None,
    current_weights=None,
    lookback_days: int = 60,
) -> pd.DataFrame:
    """Estimate asset-level liquidity risk from ADTV and participation rate."""
    prices_df = _clean_frame(prices)
    volumes_df = _clean_frame(volumes)
    if prices_df.empty or volumes_df.empty or lookback_days <= 0:
        return _empty_liquidity_frame()

    common_assets = [asset for asset in prices_df.columns if asset in volumes_df.columns]
    if not common_assets:
        return _empty_liquidity_frame()

    latest_prices = prices_df[common_assets].ffill().iloc[-1]
    average_daily_volume = volumes_df[common_assets].tail(lookback_days).mean()
    target = _prepare_weights(
        target_weights if target_weights is not None else weights, common_assets
    )
    current = _prepare_weights(current_weights, common_assets)

    if current_weights is None and target_weights is None and weights is not None:
        current = pd.Series(0.0, index=common_assets, dtype=float)

    portfolio_value_f = _safe_portfolio_value(portfolio_value)
    estimated_trade_value = (target - current).abs() * portfolio_value_f
    average_daily_traded_value = average_daily_volume * latest_prices
    participation_rate = estimated_trade_value / average_daily_traded_value.replace(0.0, np.nan)

    diagnostics = pd.DataFrame(
        {
            "asset": common_assets,
            "latest_price": latest_prices.reindex(common_assets).astype(float).values,
            "average_daily_volume": average_daily_volume.reindex(common_assets)
            .astype(float)
            .values,
            "average_daily_traded_value": average_daily_traded_value.reindex(common_assets)
            .astype(float)
            .values,
            "current_weight": current.reindex(common_assets).astype(float).values,
            "target_weight": target.reindex(common_assets).astype(float).values,
            "estimated_trade_value": estimated_trade_value.reindex(common_assets)
            .astype(float)
            .values,
            "participation_rate": participation_rate.reindex(common_assets).astype(float).values,
        }
    )
    diagnostics["liquidity_warning"] = diagnostics.apply(_liquidity_warning, axis=1)
    return diagnostics


def summarize_liquidity_diagnostics(liquidity_df: pd.DataFrame) -> dict[str, float]:
    """Summarize average and maximum liquidity diagnostics for dashboard cards."""
    if liquidity_df.empty:
        return {
            "average_participation_rate": float("nan"),
            "max_participation_rate": float("nan"),
            "min_adtv": float("nan"),
            "num_high_risk_assets": 0,
            "num_moderate_risk_assets": 0,
        }
    participation = pd.to_numeric(liquidity_df["participation_rate"], errors="coerce")
    adtv = pd.to_numeric(liquidity_df["average_daily_traded_value"], errors="coerce")
    warnings = liquidity_df["liquidity_warning"].astype(str)
    return {
        "average_participation_rate": float(participation.mean()),
        "max_participation_rate": float(participation.max()),
        "min_adtv": float(adtv.min()),
        "num_high_risk_assets": int((warnings == "High liquidity risk").sum()),
        "num_moderate_risk_assets": int((warnings == "Moderate liquidity risk").sum()),
    }


def _clean_frame(values) -> pd.DataFrame:
    if values is None:
        return pd.DataFrame()
    if isinstance(values, pd.Series):
        frame = values.to_frame()
    elif isinstance(values, pd.DataFrame):
        frame = values.copy()
    else:
        frame = pd.DataFrame(values)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame.sort_index()


def _prepare_weights(weights, assets: list[str]) -> pd.Series:
    if weights is None:
        return pd.Series(0.0, index=assets, dtype=float)
    if isinstance(weights, pd.Series):
        series = weights.copy()
    else:
        series = pd.Series(weights, index=assets if len(weights) == len(assets) else None)
    series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return series.reindex(assets).fillna(0.0).astype(float)


def _safe_portfolio_value(portfolio_value: float | None) -> float:
    if portfolio_value is None:
        return float("nan")
    value = float(portfolio_value)
    return value if np.isfinite(value) and value >= 0.0 else float("nan")


def _liquidity_warning(row: pd.Series) -> str:
    adtv = row.get("average_daily_traded_value")
    participation_rate = row.get("participation_rate")
    if not np.isfinite(adtv) or adtv <= 0.0 or not np.isfinite(participation_rate):
        return "Low liquidity data quality"
    if participation_rate > 0.10:
        return "High liquidity risk"
    if participation_rate > 0.05:
        return "Moderate liquidity risk"
    return "OK"


def _empty_liquidity_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "asset",
            "latest_price",
            "average_daily_volume",
            "average_daily_traded_value",
            "current_weight",
            "target_weight",
            "estimated_trade_value",
            "participation_rate",
            "liquidity_warning",
        ]
    )
