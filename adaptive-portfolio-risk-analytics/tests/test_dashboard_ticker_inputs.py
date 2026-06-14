"""Tests for dashboard ticker input helpers."""

from __future__ import annotations

from src.dashboard.app import merge_portfolio_tickers, parse_ticker_entries, ticker_label


def test_parse_ticker_entries_normalizes_and_deduplicates_symbols() -> None:
    tickers = parse_ticker_entries(" aapl, RELIANCE.ns\nAAPL;btc-usd ")

    assert tickers == ["AAPL", "RELIANCE.NS", "BTC-USD"]


def test_merge_portfolio_tickers_appends_added_tickers_without_replacing_selection() -> None:
    selected_labels = [ticker_label("HDFCBANK.NS"), ticker_label("TCS.NS")]

    tickers = merge_portfolio_tickers(selected_labels, ["AAPL", "TCS.NS"])

    assert tickers == ["HDFCBANK.NS", "TCS.NS", "AAPL"]


def test_ticker_label_handles_custom_tickers() -> None:
    assert ticker_label("AAPL").endswith("Custom ticker")
