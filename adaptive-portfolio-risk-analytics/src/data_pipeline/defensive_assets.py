"""Defensive sleeve data utilities for volatility-targeting overlays."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def get_defensive_asset_returns(
    start_date,
    end_date,
    preferred_ticker: str | None = "LIQUIDBEES.NS",
    fallback_tickers: Iterable[str] | None = None,
    synthetic_annual_rate: float = 0.04,
) -> tuple[pd.Series, dict[str, object]]:
    """Compatibility wrapper around the central defensive-return utility."""
    from src.adaptive.defensive import get_defensive_returns
    from .ingest import YahooFinanceProvider

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if start_ts >= end_ts:
        raise ValueError("start_date must be earlier than end_date")

    fallback_candidates = (
        list(fallback_tickers) if fallback_tickers is not None else ["LIQUIDETF.NS"]
    )
    candidates = []
    if preferred_ticker and str(preferred_ticker).strip().lower() != "synthetic risk-free":
        candidates.append(str(preferred_ticker).strip().upper())
    candidates.extend(
        ticker.strip().upper() for ticker in fallback_candidates if ticker and ticker.strip()
    )

    target_index = pd.date_range(start=start_ts, end=end_ts, freq="B")
    provider = YahooFinanceProvider()
    errors: list[str] = []

    for idx, ticker in enumerate(dict.fromkeys(candidates)):
        try:
            market_data = provider.get_market_data(
                symbols=[ticker],
                start_date=start_ts.date().isoformat(),
                end_date=end_ts.date().isoformat(),
            )
            prices = _extract_defensive_prices(
                market_data.raw_data,
                ticker,
                market_data.price_field,
            )
            missing_before = int(prices.isna().sum())
            result = get_defensive_returns(
                index=target_index,
                source="ticker",
                annual_rate=synthetic_annual_rate,
                defensive_ticker=ticker,
                prices=prices,
                fallback="synthetic",
            )
        except Exception as exc:  # external availability is recoverable
            errors.append(f"{ticker}: {exc}")
            continue
        if result.source_used == "ticker":
            metadata = {
                "selected_mode": "ticker",
                "selected_ticker": ticker,
                "synthetic_annual_rate": float(synthetic_annual_rate),
                "missing_before": missing_before,
                "missing_after": 0,
                "fallback_used": idx > 0,
                "errors": errors,
                **result.metadata,
            }
            defensive_returns = result.returns.rename(ticker)
            return defensive_returns, metadata
        errors.append(f"{ticker}: {result.notes}")

    result = get_defensive_returns(
        index=target_index,
        source="synthetic",
        annual_rate=synthetic_annual_rate,
    )
    return result.returns.rename("Synthetic Risk-Free"), {
        "selected_mode": "synthetic",
        "selected_ticker": None,
        "synthetic_annual_rate": float(synthetic_annual_rate),
        "missing_before": 0,
        "missing_after": 0,
        "fallback_used": bool(candidates),
        "errors": errors,
        **{
            **result.metadata,
            "defensive_source_requested": "ticker" if candidates else "synthetic",
            "defensive_fallback_used": bool(candidates),
        },
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
