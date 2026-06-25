"""Offline tests for the official RBI fetcher helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.sentiment import REAL_RBI_DOCUMENT_TYPES, REAL_RBI_MANIFEST_COLUMNS
from src.sentiment.rbi_official_fetcher import (
    build_manifest_record,
    build_rbi_document_id,
    classify_rbi_document_type,
    extract_rbi_text,
    extract_rbi_text_with_diagnostics,
    fetch_rbi_document_index,
    filter_rbi_entries_by_keywords,
    update_rbi_manifest,
)


RSS_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>RBI</title>
    <item>
      <title>Minutes of the Monetary Policy Committee Meeting June 2026</title>
      <link>https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=1</link>
      <description>MPC minutes discuss inflation and liquidity.</description>
      <pubDate>Sat, 06 Jun 2026 10:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Financial Stability Report June 2026</title>
      <link>https://www.rbi.org.in/Scripts/PublicationReportDetails.aspx?ID=2</link>
      <description>Financial stability report.</description>
      <pubDate>Mon, 22 Jun 2026 10:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Unrelated circular</title>
      <link>https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=3</link>
      <description>Operational circular.</description>
      <pubDate>Mon, 01 Jan 2018 10:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""


def test_rss_fixture_parses_entries_and_filters_dates(tmp_path: Path) -> None:
    feed = tmp_path / "rbi_feed.xml"
    feed.write_text(RSS_FIXTURE, encoding="utf-8")

    entries = fetch_rbi_document_index(
        "2026-06-01",
        "2026-06-10",
        sources={"press_releases": str(feed)},
    )

    assert len(entries) == 1
    assert entries[0]["publication_date"] == "2026-06-06"
    assert "Monetary Policy Committee" in entries[0]["title"]
    assert entries[0]["source_channel"] == "press_releases"


def test_keyword_filtering_uses_title_summary_and_url(tmp_path: Path) -> None:
    feed = tmp_path / "rbi_feed.xml"
    feed.write_text(RSS_FIXTURE, encoding="utf-8")

    entries = fetch_rbi_document_index(
        "2026-06-01",
        "2026-06-30",
        sources={"publications": str(feed)},
        keywords="financial stability",
    )

    assert len(entries) == 1
    assert entries[0]["title"] == "Financial Stability Report June 2026"

    all_entries = fetch_rbi_document_index(
        "2026-06-01",
        "2026-06-30",
        sources={"publications": str(feed)},
    )
    assert len(filter_rbi_entries_by_keywords(all_entries, "liquidity")) == 1


def test_document_type_classifier_maps_required_types() -> None:
    assert (
        classify_rbi_document_type(
            "Minutes of the Monetary Policy Committee Meeting",
            "https://www.rbi.org.in/pressreleases",
        )
        == "mpc_minutes"
    )
    assert (
        classify_rbi_document_type(
            "Monetary Policy Statement, 2026-27",
            "https://www.rbi.org.in",
        )
        == "monetary_policy_statement"
    )
    assert (
        classify_rbi_document_type("Deputy Governor speech on inflation", "")
        == "governor_speech"
    )
    assert (
        classify_rbi_document_type("Financial Stability Report", "")
        == "financial_stability_report"
    )
    assert classify_rbi_document_type("Annual Report 2025-26", "") == "annual_report"
    assert (
        classify_rbi_document_type(
            "RBI releases data", "https://www.rbi.org.in/pressreleases"
        )
        == "press_release"
    )
    assert (
        classify_rbi_document_type(
            "Statement on Developmental and Regulatory Policies",
            "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=1",
        )
        == "press_release"
    )


def test_classifier_does_not_emit_invalid_document_types() -> None:
    titles = [
        "Macro report",
        "Notification",
        "Publication",
        "Monetary policy",
        "Unknown update",
    ]

    produced = {classify_rbi_document_type(title, "") for title in titles}

    assert produced <= set(REAL_RBI_DOCUMENT_TYPES)
    assert "macro_report" not in produced
    assert "monetary_policy" not in produced


def test_html_extraction_returns_clean_text_without_navigation() -> None:
    html = b"""
    <html>
      <head><title>RBI Policy</title><style>.x{}</style></head>
      <body>
        <nav>Skip this menu</nav>
        <h1>Monetary Policy Statement</h1>
        <script>alert('skip')</script>
        <p>Inflation remains elevated. Liquidity is being monitored.</p>
        <footer>Skip this footer</footer>
      </body>
    </html>
    """

    text = extract_rbi_text({"content": html, "content_type": "text/html"})

    assert "Monetary Policy Statement" in text
    assert "Inflation remains elevated" in text
    assert "Skip this menu" not in text
    assert "alert" not in text


def test_pdf_unavailable_path_creates_manual_fallback_diagnostic() -> None:
    extraction = extract_rbi_text_with_diagnostics(
        {"content": b"%PDF-1.4\nnot a real pdf", "content_type": "application/pdf"}
    )

    assert extraction["text"] == ""
    assert extraction["extraction_method"] == "manual_required"
    assert extraction["warning"]


def test_manifest_update_preserves_required_columns(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    document_id = build_rbi_document_id(
        "2026-06-06",
        "mpc_minutes",
        "Minutes of the Monetary Policy Committee Meeting",
    )
    text_file = raw / f"{document_id}.txt"
    text_file.write_text("Policy remains vigilant.", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    record = build_manifest_record(
        document_id=document_id,
        publication_date="2026-06-06",
        document_type="mpc_minutes",
        title="Minutes of the Monetary Policy Committee Meeting",
        local_path=f"raw/{document_id}.txt",
        source_url="https://www.rbi.org.in/official",
        retrieval_date="2026-06-25",
        notes=(
            "fetched_by=rbi_official_fetcher; source_channel=press_releases; "
            "extraction_method=html"
        ),
    )

    result = update_rbi_manifest([record], manifest)

    frame = pd.read_csv(manifest)
    assert tuple(frame.columns) == REAL_RBI_MANIFEST_COLUMNS
    assert result["added_count"] == 1
    assert result["validation"]["summary"]["valid_document_count"] == 1
