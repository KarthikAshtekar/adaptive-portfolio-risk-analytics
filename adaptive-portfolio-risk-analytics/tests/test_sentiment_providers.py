"""Offline provider contract tests for Phase 4A.5."""

from __future__ import annotations

from pathlib import Path

from src.sentiment.providers import (
    AlphaVantageNewsProvider,
    EarningsCallProvider,
    GDELTProvider,
    NORMALIZED_SENTIMENT_COLUMNS,
    RBIProvider,
)


def test_rbi_provider_normalizes_mocked_rss() -> None:
    xml = """<?xml version="1.0"?>
    <rss><channel><item>
      <title>Monetary Policy Statement</title>
      <link>https://rbi.example/policy</link>
      <pubDate>Fri, 05 Apr 2024 08:00:00 GMT</pubDate>
      <description>Inflationary pressure may remain elevated.</description>
    </item></channel></rss>"""
    provider = RBIProvider(
        feeds_enabled=True,
        feed_urls=["mock://rbi"],
        feed_loader=lambda _: xml,
    )

    records = provider.normalize(provider.fetch("2024-01-01", "2024-12-31"))

    assert set(NORMALIZED_SENTIMENT_COLUMNS).issubset(records.columns)
    assert records.loc[0, "document_type"] == "monetary_policy_statement"
    assert records.loc[0, "provider"] == "rbi"
    assert provider.validate(records).diagnostics["valid_record_count"] == 1


def test_earnings_provider_loads_local_transcript_fixture() -> None:
    manifest = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "sentiment"
        / "earnings_calls"
        / "manifest.csv"
    )
    provider = EarningsCallProvider(manifest)

    records = provider.normalize(provider.fetch("2024-01-01", "2026-12-31"))

    assert len(records) == 2
    assert set(records["document_type"]) == {"earnings_call"}
    assert set(records["quarter"]) == {"Q4 FY2024", "Q2 FY2025"}
    assert provider.validate(records).invalid_records.empty


def test_gdelt_provider_handles_fixture_json() -> None:
    fixture = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "sentiment"
        / "provider_fixtures"
        / "gdelt_sample.json"
    )
    provider = GDELTProvider(enabled=True, fixture_path=fixture)

    records = provider.normalize(
        provider.fetch(
            "2024-01-01",
            "2026-12-31",
            query="India inflation",
        )
    )

    assert len(records) == 5
    assert records["url"].str.startswith("https://").all()
    assert records["raw_metadata"].str.contains("India inflation").all()


def test_alpha_vantage_skips_safely_without_api_key() -> None:
    provider = AlphaVantageNewsProvider(api_key="", enabled=True)

    assert provider.fetch("2024-01-01", "2024-12-31") == []
    assert provider.last_diagnostics["status"] == "missing_api_key"
    assert provider.normalize([]).empty
