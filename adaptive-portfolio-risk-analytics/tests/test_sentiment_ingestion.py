"""Tests for Phase 4A local sentiment ingestion."""

from __future__ import annotations

from io import StringIO

import pandas as pd

from src.sentiment import load_local_sentiment_csv


def test_csv_ingestion_drops_invalid_timestamps_and_duplicates() -> None:
    csv = StringIO(
        """timestamp,source,title,text,ticker,url
2024-01-02 10:00:00,Sample,Market rally,Strong rally,,https://example.test/1
invalid,Sample,Invalid row,This row is invalid,,
2024-01-02 10:00:00,Sample,Market rally,Duplicate,,
2024-01-03 09:00:00,Sample,Rate hike concern,Rate hike concern,,
"""
    )

    records = load_local_sentiment_csv(csv)

    assert len(records) == 2
    assert records["timestamp"].notna().all()
    assert records["timestamp"].is_monotonic_increasing
    assert records.iloc[0]["title"] == "Market rally"


def test_missing_optional_columns_are_added_safely() -> None:
    records = load_local_sentiment_csv(
        StringIO(
            """timestamp,source,title,text
2024-01-02,Sample,Stable market,No strong signal
"""
        )
    )

    assert {"ticker", "url", "market"}.issubset(records.columns)
    assert pd.isna(records.iloc[0]["ticker"])

