"""Data ingestion module for market price downloads."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

import pandas as pd


class DataIngester(ABC):
    """Abstract interface for market data ingestion."""

    @abstractmethod
    def fetch(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch adjusted close prices for symbols and date range."""


class YFinanceIngester(DataIngester):
    """Fetch adjusted close prices from Yahoo Finance."""

    def fetch(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        if not symbols:
            raise ValueError("symbols must not be empty")

        _validate_dates(start_date, end_date)

        import yfinance as yf

        tickers = sorted({s.strip().upper() for s in symbols if s.strip()})
        if not tickers:
            raise ValueError("symbols must contain at least one non-empty ticker")

        raw = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            group_by="column",
        )

        if raw.empty:
            raise ValueError("no market data returned by yfinance")

        if isinstance(raw.columns, pd.MultiIndex):
            close = raw.get("Close")
            if close is None:
                raise ValueError("yfinance output does not contain Close prices")
            prices = close.copy()
        else:
            col_name = tickers[0]
            prices = pd.DataFrame(raw["Close"]).rename(columns={"Close": col_name})

        prices = prices.sort_index()
        prices = prices[~prices.index.duplicated(keep="last")]

        missing = [s for s in tickers if s not in prices.columns]
        if missing:
            raise ValueError(f"missing symbols in downloaded data: {missing}")

        return prices[tickers].dropna(how="all")


class AlphaVantageIngester(DataIngester):
    """Phase 2 extension point for Alpha Vantage ingestion."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        _ = (symbols, start_date, end_date)
        raise NotImplementedError("Alpha Vantage ingestion is reserved for Phase 2.")


def _validate_dates(start_date: str, end_date: str) -> None:
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    if start >= end:
        raise ValueError("start_date must be earlier than end_date")
