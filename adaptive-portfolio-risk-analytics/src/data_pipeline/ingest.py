"""Runtime market data ingestion for Phase 1."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import pandas as pd

SAMPLE_UNIVERSE: tuple[str, ...] = (
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "KOTAKBANK.NS",
    "AXISBANK.NS",
    "TCS.NS",
    "INFY.NS",
    "WIPRO.NS",
    "HCLTECH.NS",
    "TECHM.NS",
    "RELIANCE.NS",
    "ONGC.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "NESTLEIND.NS",
    "TATACONSUM.NS",
    "SUNPHARMA.NS",
    "DRREDDY.NS",
    "CIPLA.NS",
    "MARUTI.NS",
    "M&M.NS",
    "LT.NS",
    "ULTRACEMCO.NS",
    "ASIANPAINT.NS",
    "BHARTIARTL.NS",
    "GOLDBEES.NS",
)


@dataclass(frozen=True)
class MarketDataBundle:
    """In-memory market data container used throughout Stage 1."""

    prices_df: pd.DataFrame
    volume_df: pd.DataFrame
    raw_data: pd.DataFrame
    price_field: str


class DataProvider(ABC):
    """Abstract interface for runtime market data providers."""

    @abstractmethod
    def get_market_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> MarketDataBundle:
        """Download market data for the requested assets and date range."""

    def fetch(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Return prices only for compatibility with existing callers."""

        return self.get_market_data(symbols, start_date, end_date).prices_df


class YahooFinanceProvider(DataProvider):
    """Download adjusted close prices and volume from Yahoo Finance."""

    def get_market_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> MarketDataBundle:
        tickers = _normalize_symbols(symbols)
        _validate_dates(start_date, end_date)

        import yfinance as yf

        raw = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
            actions=False,
            group_by="column",
        )
        if raw.empty:
            raise ValueError("no market data returned by yfinance")

        prices_df, price_field = _extract_prices(raw, tickers)
        volume_df = _extract_field(raw, tickers, "Volume")

        return MarketDataBundle(
            prices_df=_clean_frame(prices_df, tickers),
            volume_df=_clean_frame(volume_df, tickers),
            raw_data=raw.sort_index(),
            price_field=price_field,
        )


class YFinanceIngester(YahooFinanceProvider):
    """Backward-compatible alias for the legacy ingester name."""


class AlphaVantageProvider(DataProvider):
    """Phase 2 extension point for Alpha Vantage ingestion."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_market_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> MarketDataBundle:
        _ = (symbols, start_date, end_date)
        raise NotImplementedError("Alpha Vantage ingestion is reserved for Phase 2.")


class AlphaVantageIngester(AlphaVantageProvider):
    """Backward-compatible alias for the legacy ingester name."""


class DataIngester(DataProvider):
    """Backward-compatible base class alias for the legacy interface."""

    @abstractmethod
    def get_market_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> MarketDataBundle:
        """Download market data for the requested assets and date range."""


def build_data_inspection_table(data: MarketDataBundle) -> pd.DataFrame:
    """Summarize Stage 1 data-quality checks for prices and volume."""

    if data.prices_df.empty or data.volume_df.empty:
        raise ValueError("prices_df and volume_df must not be empty")

    assets = list(data.prices_df.columns)
    report = pd.DataFrame(index=assets)
    report.index.name = "symbol"
    report["price_field"] = data.price_field
    report["start_date"] = [_first_valid_date(data.prices_df[symbol]) for symbol in assets]
    report["end_date"] = [_last_valid_date(data.prices_df[symbol]) for symbol in assets]
    report["price_observations"] = data.prices_df.notna().sum().astype(int)
    report["volume_observations"] = data.volume_df.notna().sum().astype(int)
    report["missing_prices"] = data.prices_df.isna().sum().astype(int)
    report["missing_volume"] = data.volume_df.isna().sum().astype(int)
    report["dates_monotonic_increasing"] = bool(data.prices_df.index.is_monotonic_increasing)
    return report


def _normalize_symbols(symbols: Iterable[str]) -> list[str]:
    tickers = sorted({symbol.strip().upper() for symbol in symbols if symbol.strip()})
    if not tickers:
        raise ValueError("symbols must contain at least one non-empty ticker")
    return tickers


def _extract_prices(raw: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, str]:
    for field in ("Adj Close", "Close"):
        try:
            return _extract_field(raw, tickers, field), field
        except ValueError:
            continue
    raise ValueError("yfinance output does not contain Adj Close or Close prices")


def _extract_field(raw: pd.DataFrame, tickers: list[str], field: str) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        extracted = raw.get(field)
        if extracted is None:
            raise ValueError(f"yfinance output does not contain {field}")
        return extracted.copy()

    if len(tickers) != 1:
        raise ValueError("non-multiindex yfinance output is only supported for one ticker")
    if field not in raw.columns:
        raise ValueError(f"yfinance output does not contain {field}")

    return pd.DataFrame(raw[field]).rename(columns={field: tickers[0]})


def _clean_frame(frame: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    cleaned = frame.sort_index().copy()
    cleaned.index = pd.to_datetime(cleaned.index).tz_localize(None)
    cleaned.index.name = "Date"
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]

    missing = [ticker for ticker in tickers if ticker not in cleaned.columns]
    if missing:
        raise ValueError(f"missing symbols in downloaded data: {missing}")

    return cleaned.reindex(columns=tickers).dropna(how="all")


def _first_valid_date(series: pd.Series) -> str | None:
    first_valid = series.first_valid_index()
    if first_valid is None:
        return None
    return pd.Timestamp(first_valid).date().isoformat()


def _last_valid_date(series: pd.Series) -> str | None:
    last_valid = series.last_valid_index()
    if last_valid is None:
        return None
    return pd.Timestamp(last_valid).date().isoformat()


def _validate_dates(start_date: str, end_date: str) -> None:
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    if start >= end:
        raise ValueError("start_date must be earlier than end_date")
