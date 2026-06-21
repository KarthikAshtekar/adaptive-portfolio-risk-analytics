"""Central defensive-return resolution for adaptive and overlay workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

SUPPORTED_DEFENSIVE_SOURCES = {
    "synthetic",
    "cash_zero",
    "ticker",
    "provided_series",
}


@dataclass(frozen=True)
class DefensiveReturnResult:
    """Aligned defensive returns with auditable source metadata."""

    returns: pd.Series
    source_requested: str
    source_used: str
    annual_rate: float
    ticker: str | None
    fallback_used: bool
    notes: str

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "defensive_source_requested": self.source_requested,
            "defensive_source_used": self.source_used,
            "defensive_annual_rate": self.annual_rate,
            "defensive_ticker": self.ticker,
            "defensive_fallback_used": self.fallback_used,
            "defensive_notes": self.notes,
        }


def get_defensive_returns(
    index,
    source: str = "synthetic",
    annual_rate: float = 0.04,
    defensive_ticker: str | None = None,
    prices=None,
    returns=None,
    fallback: str = "synthetic",
    periods_per_year: int = 252,
) -> DefensiveReturnResult:
    """Return defensive returns aligned to ``index`` with source provenance.

    Ticker mode uses supplied returns or prices first, then attempts a Yahoo
    Finance download. Any switch to the configured fallback is explicit in the
    returned metadata.
    """
    target_index = _validate_index(index)
    requested = _normalize_source(source)
    fallback_source = _normalize_source(fallback)
    annual_rate = float(annual_rate)
    periods_per_year = int(periods_per_year)
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    if annual_rate <= -1.0 or not np.isfinite(annual_rate):
        raise ValueError("annual_rate must be finite and greater than -1")

    if requested == "synthetic":
        daily_return = (1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0
        series = pd.Series(
            daily_return,
            index=target_index,
            name="defensive_returns",
            dtype=float,
        )
        return DefensiveReturnResult(
            returns=series,
            source_requested=requested,
            source_used="synthetic",
            annual_rate=annual_rate,
            ticker=None,
            fallback_used=False,
            notes=(
                f"Compounded daily equivalent of {annual_rate:.4%} annualized "
                f"over {periods_per_year} periods."
            ),
        )

    if requested == "cash_zero":
        return DefensiveReturnResult(
            returns=pd.Series(
                0.0,
                index=target_index,
                name="defensive_returns",
                dtype=float,
            ),
            source_requested=requested,
            source_used="cash_zero",
            annual_rate=annual_rate,
            ticker=None,
            fallback_used=False,
            notes="Zero-return cash sleeve.",
        )

    if requested == "provided_series":
        try:
            series, alignment_note = _coerce_return_series(
                returns,
                target_index,
                defensive_ticker,
            )
        except (TypeError, ValueError) as exc:
            return _fallback_result(
                target_index,
                requested=requested,
                fallback=fallback_source,
                annual_rate=annual_rate,
                defensive_ticker=defensive_ticker,
                periods_per_year=periods_per_year,
                reason=str(exc),
            )
        return DefensiveReturnResult(
            returns=series,
            source_requested=requested,
            source_used="provided_series",
            annual_rate=annual_rate,
            ticker=_normalize_ticker(defensive_ticker),
            fallback_used=False,
            notes=alignment_note,
        )

    ticker = _normalize_ticker(defensive_ticker)
    if not ticker:
        return _fallback_result(
            target_index,
            requested=requested,
            fallback=fallback_source,
            annual_rate=annual_rate,
            defensive_ticker=None,
            periods_per_year=periods_per_year,
            reason="Ticker source requested without defensive_ticker.",
        )

    errors: list[str] = []
    if returns is not None:
        try:
            series, note = _coerce_return_series(returns, target_index, ticker)
            return DefensiveReturnResult(
                returns=series,
                source_requested=requested,
                source_used="ticker",
                annual_rate=annual_rate,
                ticker=ticker,
                fallback_used=False,
                notes=f"Ticker returns supplied directly. {note}",
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"supplied returns: {exc}")

    if prices is not None:
        try:
            series, note = _returns_from_prices(prices, target_index, ticker)
            return DefensiveReturnResult(
                returns=series,
                source_requested=requested,
                source_used="ticker",
                annual_rate=annual_rate,
                ticker=ticker,
                fallback_used=False,
                notes=f"Ticker prices supplied directly. {note}",
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"supplied prices: {exc}")

    try:
        downloaded_prices = _download_ticker_prices(ticker, target_index)
        series, note = _returns_from_prices(
            downloaded_prices,
            target_index,
            ticker,
        )
        return DefensiveReturnResult(
            returns=series,
            source_requested=requested,
            source_used="ticker",
            annual_rate=annual_rate,
            ticker=ticker,
            fallback_used=False,
            notes=f"Ticker prices downloaded from Yahoo Finance. {note}",
        )
    except Exception as exc:  # external availability is intentionally recoverable
        errors.append(f"download: {exc}")

    return _fallback_result(
        target_index,
        requested=requested,
        fallback=fallback_source,
        annual_rate=annual_rate,
        defensive_ticker=ticker,
        periods_per_year=periods_per_year,
        reason="; ".join(errors),
    )


def defensive_source_from_label(label: str | None) -> tuple[str, str | None]:
    """Map dashboard/experiment labels to the central source contract."""
    normalized = str(label or "synthetic").strip()
    lowered = normalized.lower().replace("-", "_")
    if lowered in {
        "synthetic",
        "synthetic risk_free",
        "synthetic risk free",
        "synthetic_4pct",
        "synthetic 4% annualized",
    }:
        return "synthetic", None
    if lowered in {"cash_zero", "cash / zero return", "cash", "zero"}:
        return "cash_zero", None
    if lowered in {"provided_series", "provided series if available"}:
        return "provided_series", None
    if normalized.upper() in {"LIQUIDBEES.NS", "LIQUIDETF.NS"}:
        return "ticker", normalized.upper()
    if normalized.upper().endswith((".NS", ".BO")):
        return "ticker", normalized.upper()
    return _normalize_source(normalized), None


def format_defensive_source(metadata: Mapping[str, object] | None) -> str:
    """Return a concise user-facing defensive sleeve label."""
    values = dict(metadata or {})
    source_used = str(values.get("defensive_source_used", "synthetic"))
    if source_used == "ticker":
        return str(values.get("defensive_ticker") or "Ticker defensive sleeve")
    if source_used == "cash_zero":
        return "Cash / zero return"
    if source_used == "provided_series":
        return "Provided defensive return series"
    rate = float(values.get("defensive_annual_rate", 0.04))
    return f"Synthetic {rate:.0%} annualized"


def _fallback_result(
    index: pd.DatetimeIndex,
    *,
    requested: str,
    fallback: str,
    annual_rate: float,
    defensive_ticker: str | None,
    periods_per_year: int,
    reason: str,
) -> DefensiveReturnResult:
    if fallback not in {"synthetic", "cash_zero"}:
        raise ValueError(
            "fallback must be 'synthetic' or 'cash_zero' when the requested "
            "defensive source is unavailable"
        )
    fallback_result = get_defensive_returns(
        index=index,
        source=fallback,
        annual_rate=annual_rate,
        periods_per_year=periods_per_year,
    )
    return DefensiveReturnResult(
        returns=fallback_result.returns,
        source_requested=requested,
        source_used=fallback_result.source_used,
        annual_rate=annual_rate,
        ticker=_normalize_ticker(defensive_ticker),
        fallback_used=True,
        notes=(
            f"Requested source unavailable; used {fallback_result.source_used} fallback. "
            f"Reason: {reason}"
        ),
    )


def _coerce_return_series(
    values,
    index: pd.DatetimeIndex,
    ticker: str | None,
) -> tuple[pd.Series, str]:
    series = _extract_series(values, ticker, value_name="returns")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("defensive return series index must be a DatetimeIndex")
    numeric = pd.to_numeric(series, errors="coerce").sort_index()
    numeric = numeric[~numeric.index.duplicated(keep="last")]
    aligned = numeric.reindex(index)
    missing = int(aligned.isna().sum())
    if aligned.notna().sum() == 0:
        raise ValueError("defensive return series has no observations on the target index")
    aligned = aligned.fillna(0.0).rename("defensive_returns").astype(float)
    note = (
        "Aligned provided returns to the target index; "
        f"{missing} missing observations were filled with zero."
    )
    return aligned, note


def _returns_from_prices(
    values,
    index: pd.DatetimeIndex,
    ticker: str | None,
) -> tuple[pd.Series, str]:
    series = _extract_series(values, ticker, value_name="prices")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("defensive price series index must be a DatetimeIndex")
    numeric = pd.to_numeric(series, errors="coerce").sort_index()
    numeric = numeric[~numeric.index.duplicated(keep="last")]
    expanded_index = numeric.index.union(index).sort_values()
    aligned_prices = numeric.reindex(expanded_index).ffill().reindex(index)
    missing = int(aligned_prices.isna().sum())
    if aligned_prices.notna().sum() < 2:
        raise ValueError("defensive price series has fewer than two aligned observations")
    aligned_prices = aligned_prices.bfill()
    returns = (
        aligned_prices.pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .rename("defensive_returns")
        .astype(float)
    )
    return returns, (
        "Converted prices to simple returns; "
        f"{missing} leading or unaligned price observations were safely filled."
    )


def _extract_series(values, ticker: str | None, *, value_name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.copy()
    if isinstance(values, Mapping):
        normalized_ticker = _normalize_ticker(ticker)
        if normalized_ticker in values:
            return _extract_series(
                values[normalized_ticker],
                normalized_ticker,
                value_name=value_name,
            )
        if ticker in values:
            return _extract_series(values[ticker], ticker, value_name=value_name)
        if len(values) == 1:
            return _extract_series(
                next(iter(values.values())),
                ticker,
                value_name=value_name,
            )
        raise ValueError(f"{value_name} mapping does not contain ticker '{ticker}'")
    if isinstance(values, pd.DataFrame):
        if values.empty:
            raise ValueError(f"{value_name} DataFrame is empty")
        normalized_ticker = _normalize_ticker(ticker)
        if normalized_ticker and normalized_ticker in values.columns:
            return values[normalized_ticker].copy()
        if ticker in values.columns:
            return values[ticker].copy()
        if values.shape[1] == 1:
            return values.iloc[:, 0].copy()
        raise ValueError(
            f"{value_name} DataFrame must contain ticker '{ticker}' or one column"
        )
    if values is None:
        raise ValueError(f"{value_name} were not supplied")
    raise TypeError(
        f"{value_name} must be a pandas Series, DataFrame, mapping, or None"
    )


def _download_ticker_prices(
    ticker: str,
    index: pd.DatetimeIndex,
) -> pd.Series:
    from src.data_pipeline.ingest import YahooFinanceProvider

    start = index.min().date().isoformat()
    end = (index.max().date() + timedelta(days=1)).isoformat()
    market_data = YahooFinanceProvider().get_market_data(
        symbols=[ticker],
        start_date=start,
        end_date=end,
    )
    if ticker not in market_data.prices_df.columns:
        raise ValueError(f"download did not contain {ticker}")
    return market_data.prices_df[ticker]


def _normalize_source(source: str) -> str:
    normalized = str(source).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "cash": "cash_zero",
        "zero": "cash_zero",
        "provided": "provided_series",
        "series": "provided_series",
        "synthetic_risk_free": "synthetic",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_DEFENSIVE_SOURCES:
        supported = ", ".join(sorted(SUPPORTED_DEFENSIVE_SOURCES))
        raise ValueError(
            f"unsupported defensive source '{source}'. Supported: {supported}"
        )
    return normalized


def _normalize_ticker(ticker: str | None) -> str | None:
    normalized = str(ticker or "").strip().upper()
    return normalized or None


def _validate_index(index) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        try:
            index = pd.DatetimeIndex(index)
        except Exception as exc:
            raise TypeError("index must be convertible to a DatetimeIndex") from exc
    if index.empty:
        raise ValueError("index must not be empty")
    normalized = pd.DatetimeIndex(index).tz_localize(None)
    if normalized.has_duplicates:
        raise ValueError("index must not contain duplicate dates")
    if not normalized.is_monotonic_increasing:
        normalized = normalized.sort_values()
    return normalized
