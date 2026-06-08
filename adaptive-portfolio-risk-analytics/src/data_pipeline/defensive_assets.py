"""Defensive sleeve data utilities for volatility-targeting overlays."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .ingest import YahooFinanceProvider


def get_defensive_asset_returns(
    start_date,
    end_date,
    preferred_ticker: str | None = "LIQUIDBEES.NS",
    fallback_tickers: Iterable[str] | None = None,
    synthetic_annual_rate: float = 0.04,
) -> tuple[pd.Series, dict[str, object]]:
    """Fetch defensive asset returns or fall back to a synthetic risk-free series."""
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if start_ts >= end_ts:
        raise ValueError("start_date must be earlier than end_date")

    fallback_candidates = (
        list(fallback_tickers)
        if fallback_tickers is not None
        else ["LIQUIDETF.NS"]
    )
    candidates = []
    if preferred_ticker and str(preferred_ticker).strip().lower() != "synthetic risk-free":
        candidates.append(str(preferred_ticker).strip().upper())
    candidates.extend(
        ticker.strip().upper()
        for ticker in fallback_candidates
        if ticker and ticker.strip()
    )

    provider = YahooFinanceProvider()
    errors: list[str] = []

    for idx, ticker in enumerate(dict.fromkeys(candidates)):
        try:
            market_data = provider.get_market_data(
                symbols=[ticker],
                start_date=start_ts.date().isoformat(),
                end_date=end_ts.date().isoformat(),
            )
            prices = _extract_defensive_prices(market_data.raw_data, ticker, market_data.price_field)
            missing_before = int(prices.isna().sum())
            cleaned_prices = prices.ffill().bfill()
            missing_after = int(cleaned_prices.isna().sum())
            if cleaned_prices.dropna().shape[0] < 2:
                raise ValueError("defensive asset prices must contain at least two valid observations")

            defensive_returns = cleaned_prices.pct_change().dropna()
            defensive_returns.name = ticker
            return defensive_returns, {
                "selected_mode": "ticker",
                "selected_ticker": ticker,
                "synthetic_annual_rate": float(synthetic_annual_rate),
                "missing_before": missing_before,
                "missing_after": missing_after,
                "fallback_used": idx > 0,
                "errors": errors,
            }
        except Exception as exc:  # pragma: no cover - exercised via fallback tests
            errors.append(f"{ticker}: {exc}")

    synthetic_index = pd.date_range(start=start_ts, end=end_ts, freq="B")
    synthetic_daily_rate = float(synthetic_annual_rate) / 252.0
    defensive_returns = pd.Series(
        synthetic_daily_rate,
        index=synthetic_index,
        dtype=float,
        name="Synthetic Risk-Free",
    )
    return defensive_returns, {
        "selected_mode": "synthetic",
        "selected_ticker": None,
        "synthetic_annual_rate": float(synthetic_annual_rate),
        "missing_before": 0,
        "missing_after": 0,
        "fallback_used": bool(candidates),
        "errors": errors,
    }


def _extract_defensive_prices(raw_data: pd.DataFrame, ticker: str, price_field: str) -> pd.Series:
    if isinstance(raw_data.columns, pd.MultiIndex):
        price_frame = raw_data.get(price_field)
        if price_frame is None or ticker not in price_frame.columns:
            raise ValueError(f"{ticker} not available in defensive asset raw data")
        prices = price_frame[ticker].copy()
    else:
        if price_field not in raw_data.columns:
            raise ValueError(f"{price_field} not available in defensive asset raw data")
        prices = raw_data[price_field].copy()

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()
    prices.name = ticker
    return prices.astype(float)
