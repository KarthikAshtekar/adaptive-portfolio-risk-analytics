"""No-internet tests for targeted RBI MPC / policy-document fetching."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts import fetch_rbi_documents


def _entry(
    title: str,
    *,
    publication_date: str = "2026-06-20",
    url_id: str = "1",
    source_channel: str = "press_releases",
) -> dict[str, object]:
    return {
        "publication_date": publication_date,
        "title": title,
        "summary": "",
        "source_url": (
            "https://www.rbi.org.in/commonman/English/Scripts/"
            f"PressReleases.aspx?Id={url_id}"
        ),
        "source_channel": source_channel,
        "feed_url": "https://www.rbi.org.in/commonman/English/Scripts/PressReleases.aspx",
        "candidate_source_page": (
            "https://www.rbi.org.in/commonman/English/Scripts/PressReleases.aspx"
        ),
    }


def _fake_download(url: str) -> dict[str, object]:
    return {
        "url": url,
        "ok": True,
        "content": (
            "<html><body><p>The Reserve Bank reviewed inflation, liquidity, "
            "growth, policy repo rate, and financial conditions for the "
            "current policy assessment."
            "</p></body></html>"
        ).encode("utf-8"),
        "http_status": 200,
        "content_type": "text/html",
        "error": "",
    }


def test_mpc_minutes_title_is_prioritized_over_generic_press_release(
    tmp_path: Path,
    monkeypatch,
) -> None:
    entries = [
        _entry("Generic Press Release on Banking Regulation", url_id="1"),
        _entry(
            "Minutes of the Monetary Policy Committee Meeting, June 4 to 6, 2026",
            url_id="2",
        ),
    ]
    monkeypatch.setattr(
        fetch_rbi_documents.fetcher,
        "fetch_rbi_document_index",
        lambda *args, **kwargs: entries,
    )
    monkeypatch.setattr(
        fetch_rbi_documents.fetcher,
        "download_rbi_document",
        _fake_download,
    )

    summary = fetch_rbi_documents.run_fetch(
        from_date="2026-06-01",
        to_date="2026-06-30",
        output_dir=tmp_path / "rbi_real" / "raw",
        manifest_path=tmp_path / "rbi_real" / "manifest.csv",
        diagnostics_dir=tmp_path / "diagnostics",
        keywords="",
        target_policy_docs=True,
        max_documents=1,
        request_delay_seconds=0,
    )
    manifest = pd.read_csv(tmp_path / "rbi_real" / "manifest.csv")

    assert summary["downloaded_documents"] == 1
    assert manifest.loc[0, "document_type"] == "mpc_minutes"


def test_monetary_policy_statement_title_is_prioritized(
    tmp_path: Path,
    monkeypatch,
) -> None:
    entries = [
        _entry("Generic Press Release on Banking Regulation", url_id="1"),
        _entry(
            "Monetary Policy Statement, 2026-27 Resolution of the Monetary Policy Committee",
            url_id="2",
        ),
    ]
    monkeypatch.setattr(
        fetch_rbi_documents.fetcher,
        "fetch_rbi_document_index",
        lambda *args, **kwargs: entries,
    )
    monkeypatch.setattr(
        fetch_rbi_documents.fetcher,
        "download_rbi_document",
        _fake_download,
    )

    summary = fetch_rbi_documents.run_fetch(
        from_date="2026-06-01",
        to_date="2026-06-30",
        output_dir=tmp_path / "rbi_real" / "raw",
        manifest_path=tmp_path / "rbi_real" / "manifest.csv",
        diagnostics_dir=tmp_path / "diagnostics",
        keywords="",
        target_policy_docs=True,
        max_documents=1,
        request_delay_seconds=0,
    )
    manifest = pd.read_csv(tmp_path / "rbi_real" / "manifest.csv")
    diagnostics = pd.read_csv(tmp_path / "diagnostics" / "fetch_diagnostics.csv")

    assert summary["downloaded_documents"] == 1
    assert manifest.loc[0, "document_type"] == "monetary_policy_statement"
    assert "Monetary Policy Statement" in diagnostics.loc[0, "matched_policy_phrase"]


def test_target_document_types_exclude_governor_speeches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    entries = [
        _entry("Governor's Statement on Monetary Policy", url_id="1"),
        _entry(
            "Minutes of the Monetary Policy Committee Meeting, June 4 to 6, 2026",
            url_id="2",
        ),
    ]
    monkeypatch.setattr(
        fetch_rbi_documents.fetcher,
        "fetch_rbi_document_index",
        lambda *args, **kwargs: entries,
    )
    monkeypatch.setattr(
        fetch_rbi_documents.fetcher,
        "download_rbi_document",
        _fake_download,
    )

    summary = fetch_rbi_documents.run_fetch(
        from_date="2026-06-01",
        to_date="2026-06-30",
        output_dir=tmp_path / "rbi_real" / "raw",
        manifest_path=tmp_path / "rbi_real" / "manifest.csv",
        diagnostics_dir=tmp_path / "diagnostics",
        keywords="",
        target_policy_docs=True,
        target_document_types="mpc_minutes,monetary_policy_statement",
        request_delay_seconds=0,
    )
    diagnostics = pd.read_csv(tmp_path / "diagnostics" / "fetch_diagnostics.csv")

    assert summary["downloaded_documents"] == 1
    assert "governor_speech" not in pd.read_csv(
        tmp_path / "rbi_real" / "manifest.csv"
    )["document_type"].tolist()
    skipped = diagnostics.loc[diagnostics["document_type"].eq("governor_speech")]
    assert skipped.loc[skipped.index[0], "skip_reason"] == "non_target_document_type"
    assert str(skipped.loc[skipped.index[0], "target_document_type_match"]).lower() == "false"


def test_skipped_non_target_documents_appear_in_diagnostics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        fetch_rbi_documents.fetcher,
        "fetch_rbi_document_index",
        lambda *args, **kwargs: [
            _entry("Governor's Statement on Monetary Policy", url_id="1"),
        ],
    )

    summary = fetch_rbi_documents.run_fetch(
        from_date="2026-06-01",
        to_date="2026-06-30",
        output_dir=tmp_path / "rbi_real" / "raw",
        manifest_path=tmp_path / "rbi_real" / "manifest.csv",
        diagnostics_dir=tmp_path / "diagnostics",
        keywords="",
        target_policy_docs=True,
        target_document_types="mpc_minutes,monetary_policy_statement",
        request_delay_seconds=0,
    )
    diagnostics = pd.read_csv(tmp_path / "diagnostics" / "fetch_diagnostics.csv")

    assert summary["downloaded_documents"] == 0
    assert diagnostics.loc[0, "download_status"] == "skipped"
    assert diagnostics.loc[0, "skip_reason"] == "non_target_document_type"


def test_targeted_mode_keeps_annual_report_index_filter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    annual_index_text = """
    Annual Report - Reserve Bank of India
    Skip to main content
    Change Language
    Search the Website
    Note : To read the chapter of your choice, please click on the links below.
    2026 2025 2024 2023 Archives Top Back to previous page
    """
    monkeypatch.setattr(
        fetch_rbi_documents.fetcher,
        "fetch_rbi_document_index",
        lambda *args, **kwargs: [
            _entry("Annual Report", url_id="annual", source_channel="publications"),
        ],
    )
    monkeypatch.setattr(
        fetch_rbi_documents.fetcher,
        "download_rbi_document",
        lambda url: {
            "url": url,
            "ok": True,
            "content": f"<html><body>{annual_index_text}</body></html>".encode(
                "utf-8"
            ),
            "http_status": 200,
            "content_type": "text/html",
            "error": "",
        },
    )

    summary = fetch_rbi_documents.run_fetch(
        from_date="2026-06-01",
        to_date="2026-06-30",
        output_dir=tmp_path / "rbi_real" / "raw",
        manifest_path=tmp_path / "rbi_real" / "manifest.csv",
        diagnostics_dir=tmp_path / "diagnostics",
        keywords="annual report",
        target_policy_docs=True,
        request_delay_seconds=0,
    )

    assert summary["downloaded_documents"] == 0
    assert summary["skipped_index_pages"] == 1


def test_targeted_mode_keeps_irrelevant_press_release_exclusion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        fetch_rbi_documents.fetcher,
        "fetch_rbi_document_index",
        lambda *args, **kwargs: [
            _entry("Result of Treasury Bills Auction Result", url_id="auction"),
        ],
    )

    summary = fetch_rbi_documents.run_fetch(
        from_date="2026-06-01",
        to_date="2026-06-30",
        output_dir=tmp_path / "rbi_real" / "raw",
        manifest_path=tmp_path / "rbi_real" / "manifest.csv",
        diagnostics_dir=tmp_path / "diagnostics",
        keywords="",
        target_policy_docs=True,
        request_delay_seconds=0,
    )
    diagnostics = pd.read_csv(tmp_path / "diagnostics" / "fetch_diagnostics.csv")

    assert summary["excluded_irrelevant_press_releases"] == 1
    assert diagnostics.loc[0, "skip_reason"] == "excluded_irrelevant_press_release"
