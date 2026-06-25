"""RBI official fetcher tests for irrelevant press-release exclusion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts import fetch_rbi_documents
from src.sentiment.rbi_official_fetcher import rbi_entry_exclude_keyword_match


def _entry(title: str, url_suffix: str = "1") -> dict[str, object]:
    return {
        "publication_date": "2026-06-10",
        "title": title,
        "summary": "",
        "source_url": (
            "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?"
            f"prid={url_suffix}"
        ),
        "source_channel": "press_releases",
        "feed_url": "https://www.rbi.org.in/pressreleases_rss.xml",
    }


def test_t_bill_auction_result_is_excluded() -> None:
    matched = rbi_entry_exclude_keyword_match(
        _entry("Result of Treasury Bills Auction Result")
    )

    assert matched in {"treasury bills", "auction result"}


def test_state_government_securities_auction_result_is_excluded() -> None:
    matched = rbi_entry_exclude_keyword_match(
        _entry("Result of State Government Securities Auction Result")
    )

    assert matched == "state government securities"


def test_customer_liability_amendment_direction_is_excluded() -> None:
    matched = rbi_entry_exclude_keyword_match(
        _entry("Customer Liability in Digital Transactions Amendment Direction")
    )

    assert matched in {"customer liability", "digital transactions"}


def test_mpc_minutes_are_not_excluded() -> None:
    assert (
        rbi_entry_exclude_keyword_match(
            _entry("Minutes of the Monetary Policy Committee Meeting")
        )
        == ""
    )


def test_monetary_policy_statement_is_not_excluded() -> None:
    assert (
        rbi_entry_exclude_keyword_match(_entry("Monetary Policy Statement, 2026-27"))
        == ""
    )


def test_governor_speech_is_not_excluded() -> None:
    assert (
        rbi_entry_exclude_keyword_match(
            _entry("Governor speech on monetary policy transmission")
        )
        == ""
    )


def test_keyword_exclusion_writes_diagnostics_without_download(
    tmp_path: Path,
    monkeypatch,
) -> None:
    excluded = _entry("Result of State Government Securities Auction Result", "2")
    valid = _entry("Monetary Policy Statement, 2026-27", "3")
    monkeypatch.setattr(
        fetch_rbi_documents.fetcher,
        "fetch_rbi_document_index",
        lambda *args, **kwargs: [excluded, valid],
    )
    calls = {"downloads": 0}

    def fake_download(url: str) -> dict[str, object]:
        calls["downloads"] += 1
        return {
            "url": url,
            "ok": True,
            "content": b"<html><body><p>Monetary Policy Statement text.</p></body></html>",
            "http_status": 200,
            "content_type": "text/html",
            "error": "",
        }

    monkeypatch.setattr(fetch_rbi_documents.fetcher, "download_rbi_document", fake_download)

    summary = fetch_rbi_documents.run_fetch(
        from_date="2026-06-01",
        to_date="2026-06-30",
        output_dir=tmp_path / "rbi_real" / "raw",
        manifest_path=tmp_path / "rbi_real" / "manifest.csv",
        diagnostics_dir=tmp_path / "diagnostics",
        keywords="",
        request_delay_seconds=0,
    )

    diagnostics = pd.read_csv(tmp_path / "diagnostics" / "fetch_diagnostics.csv")
    excluded_row = diagnostics.loc[
        diagnostics["skip_reason"].eq("excluded_irrelevant_press_release")
    ].iloc[0]

    assert summary["excluded_irrelevant_press_releases"] == 1
    assert summary["downloaded_documents"] == 1
    assert calls["downloads"] == 1
    assert excluded_row["download_status"] == "skipped"
    assert str(excluded_row["excluded_by_keyword"]).lower() == "true"
    assert excluded_row["exclude_keyword_matched"] == "state government securities"
    assert str(excluded_row["included_in_manifest"]).lower() == "false"
