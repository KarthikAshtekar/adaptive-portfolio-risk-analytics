"""Unified provider ingestion diagnostics and deduplication tests."""

from __future__ import annotations

import pandas as pd

from src.sentiment.api_ingestion import (
    INGESTION_OUTPUT_FILES,
    run_sentiment_provider_ingestion,
)
from src.sentiment.providers import LocalProvider


def test_api_ingestion_writes_outputs_and_deduplicates(tmp_path) -> None:
    records = pd.DataFrame(
        [
            {
                "record_id": "one",
                "timestamp": "2024-01-01T08:00:00Z",
                "publication_time": "2024-01-01T08:00:00Z",
                "retrieval_time": "2024-01-02T08:00:00Z",
                "source": "Fixture",
                "provider": "fixture",
                "document_type": "financial_news",
                "entity": "",
                "ticker": "",
                "sector": "",
                "country": "IN",
                "title": "Inflation risk",
                "text": "Inflation shock may remain elevated.",
                "url": "https://example.com/one",
                "language": "en",
                "raw_metadata": "{}",
            },
            {
                "record_id": "duplicate",
                "timestamp": "2024-01-01T08:00:00Z",
                "publication_time": "2024-01-01T08:00:00Z",
                "retrieval_time": "2024-01-02T08:00:00Z",
                "source": "Fixture",
                "provider": "fixture",
                "document_type": "financial_news",
                "entity": "",
                "ticker": "",
                "sector": "",
                "country": "IN",
                "title": "Inflation risk duplicate",
                "text": "Same URL and timestamp.",
                "url": "https://example.com/one",
                "language": "en",
                "raw_metadata": "{}",
            },
        ]
    )
    result = run_sentiment_provider_ingestion(
        [LocalProvider(records, provider_name="fixture")],
        "2024-01-01",
        "2024-01-31",
        tmp_path,
        use_cache=False,
    )

    assert all((tmp_path / filename).is_file() for filename in INGESTION_OUTPUT_FILES)
    assert len(result["normalized_sentiment_records"]) == 2
    assert len(result["deduped_sentiment_records"]) == 1
    assert result["duplicate_record_count"] == 1
    assert result["provider_diagnostics"].loc[0, "valid_record_count"] == 2
