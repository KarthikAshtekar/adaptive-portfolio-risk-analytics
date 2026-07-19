"""Placeholder exclusion tests for intake and collection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from scripts.collect_real_nlp_data import collect_real_nlp_data
from src.sentiment import validate_corpus_manifest


def test_placeholder_rows_are_excluded_from_real_intake(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "record_id": "DO_NOT_USE_PLACEHOLDER_NEWS",
                "publication_time": "2024-01-01T09:00:00Z",
                "source": "DO_NOT_USE_PLACEHOLDER",
                "provider": "local_manifest",
                "document_type": "unknown",
                "entity": "India",
                "ticker": "",
                "sector": "",
                "country": "IN",
                "title": "DO_NOT_USE_PLACEHOLDER",
                "text": "DO_NOT_USE_PLACEHOLDER",
                "url": "https://example.com/DO_NOT_USE_PLACEHOLDER",
                "language": "en",
                "retrieval_time": "2024-01-01T10:00:00Z",
                "notes": "DO_NOT_USE_PLACEHOLDER",
            }
        ]
    ).to_csv(tmp_path / "news.csv", index=False)

    result = validate_corpus_manifest("news", tmp_path / "news.csv")

    assert result["summary"]["placeholder_record_count"] == 1
    assert result["summary"]["valid_record_count"] == 0
    assert result["rows"]["validation_status"].eq("placeholder_excluded").all()


def test_collection_ignores_placeholder_earnings_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "earnings.csv"
    pd.DataFrame(
        [
            {
                "document_id": "DO_NOT_USE_PLACEHOLDER_EARNINGS",
                "company": "DO_NOT_USE_PLACEHOLDER",
                "ticker": "PLACEHOLDER.NS",
                "sector": "Banking",
                "quarter": "Q1",
                "publication_date": "2024-01-01",
                "title": "DO_NOT_USE_PLACEHOLDER",
                "local_path": "missing.txt",
                "source_url": "https://example.com/DO_NOT_USE_PLACEHOLDER",
                "retrieval_date": "2024-01-02",
                "language": "en",
                "notes": "DO_NOT_USE_PLACEHOLDER",
            }
        ]
    ).to_csv(manifest, index=False)
    config = {
        "rbi": {"enabled": False, "mode": "local_manifest"},
        "earnings": {
            "enabled": True,
            "mode": "local_manifest",
            "local_manifest_path": str(manifest),
        },
        "gdelt": {"enabled": False, "mode": "api"},
        "alpha_vantage": {
            "enabled": False,
            "mode": "api",
            "api_key_env": "ALPHAVANTAGE_API_KEY",
        },
        "scoring": {"method": "lexicon", "finbert_enabled": False},
        "validation": {
            "decision_lag_days": 1,
            "min_coverage_ratio": 0.20,
            "min_records": 50,
            "min_distinct_dates": 20,
            "max_reaction_warning_rate": 0.25,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = collect_real_nlp_data(
        config_path=config_path,
        start_date="2024-01-01",
        end_date="2024-12-31",
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        no_live=True,
    )

    assert result["summary"]["collected_record_count"] == 0
    assert result["summary"]["real_record_count"] == 0
