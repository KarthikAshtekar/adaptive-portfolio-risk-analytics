"""RBI official fetcher tests for index/archive page exclusion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts import fetch_rbi_documents
from src.sentiment.rbi_official_fetcher import (
    classify_rbi_document_type,
    is_rbi_index_or_navigation_page,
)


ANNUAL_REPORT_INDEX_TEXT = """
Annual Report - Reserve Bank of India
Skip to main content
Selected Selected
Change Language
Search the Website
Annual Report
Note : To read the chapter of your choice, please click on the links below.
You can also read past reports by accessing the archives in the right panel.
2026 2025 2024 2023 2022 2021 2020 Archives Top Back to previous page
"""


VALID_MPC_MINUTES_TEXT = """
The Monetary Policy Committee met on June 4, 2026 to review the evolving
macroeconomic and financial conditions. Members discussed inflation, liquidity,
growth, external sector developments, and the policy repo rate. The minutes
record individual assessments and votes of members based on available data.
The Committee noted that monetary policy would remain focused on durable
alignment of inflation with the target while supporting growth.
"""


VALID_MONETARY_POLICY_STATEMENT_TEXT = """
The Monetary Policy Statement sets out the assessment of inflation and growth
conditions. The Reserve Bank reviewed liquidity conditions, financial markets,
credit growth, and external spillovers. The policy stance remains data dependent
and the statement records the policy decision and rationale for the decision.
"""


VALID_GOVERNOR_SPEECH_TEXT = """
Governor speech on monetary policy communication and financial stability. The
speech discusses inflation expectations, liquidity management, banking sector
resilience, and the role of credible central bank communication in anchoring
market expectations. It includes substantive remarks rather than archive links.
"""


def test_annual_report_archive_text_is_detected_as_index_page() -> None:
    is_index_page, reason = is_rbi_index_or_navigation_page(
        "Annual Report",
        ANNUAL_REPORT_INDEX_TEXT,
        "https://www.rbi.org.in/Scripts/AnnualReportPublications.aspx",
    )

    assert is_index_page is True
    assert reason == "annual_report_index_page"


def test_annual_report_archive_is_not_classified_as_annual_report() -> None:
    document_type = classify_rbi_document_type(
        "Annual Report",
        "https://www.rbi.org.in/Scripts/AnnualReportPublications.aspx",
        ANNUAL_REPORT_INDEX_TEXT,
    )

    assert document_type != "annual_report"
    assert document_type == "unknown"


def test_index_page_is_excluded_from_manifest_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    entry = {
        "publication_date": "2026-06-01",
        "title": "Annual Report",
        "summary": "Annual Report archive",
        "source_url": "https://www.rbi.org.in/Scripts/AnnualReportPublications.aspx",
        "source_channel": "publications",
        "feed_url": "https://www.rbi.org.in/Publication_rss.xml",
    }
    monkeypatch.setattr(
        fetch_rbi_documents.fetcher,
        "fetch_rbi_document_index",
        lambda *args, **kwargs: [entry],
    )
    monkeypatch.setattr(
        fetch_rbi_documents.fetcher,
        "download_rbi_document",
        lambda url: {
            "url": url,
            "ok": True,
            "content": f"<html><body>{ANNUAL_REPORT_INDEX_TEXT}</body></html>".encode("utf-8"),
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
        request_delay_seconds=0,
    )

    manifest = pd.read_csv(tmp_path / "rbi_real" / "manifest.csv")
    diagnostics = pd.read_csv(tmp_path / "diagnostics" / "fetch_diagnostics.csv")

    assert summary["downloaded_documents"] == 0
    assert summary["skipped_index_pages"] == 1
    assert manifest.empty
    assert diagnostics.loc[0, "download_status"] == "skipped"
    assert (
        diagnostics.loc[0, "is_index_page"] is True
        or str(diagnostics.loc[0, "is_index_page"]).lower() == "true"
    )
    assert diagnostics.loc[0, "skip_reason"] == "index_or_navigation_page"
    assert (
        diagnostics.loc[0, "included_in_manifest"] is False
        or str(diagnostics.loc[0, "included_in_manifest"]).lower() == "false"
    )


def test_valid_mpc_minutes_are_not_excluded() -> None:
    is_index_page, reason = is_rbi_index_or_navigation_page(
        "Minutes of the Monetary Policy Committee Meeting",
        VALID_MPC_MINUTES_TEXT,
        "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=1",
    )

    assert is_index_page is False
    assert reason == ""
    assert (
        classify_rbi_document_type(
            "Minutes of the Monetary Policy Committee Meeting",
            "",
            VALID_MPC_MINUTES_TEXT,
        )
        == "mpc_minutes"
    )


def test_valid_monetary_policy_statement_is_not_excluded() -> None:
    is_index_page, reason = is_rbi_index_or_navigation_page(
        "Monetary Policy Statement, 2026-27",
        VALID_MONETARY_POLICY_STATEMENT_TEXT,
        "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=2",
    )

    assert is_index_page is False
    assert reason == ""
    assert (
        classify_rbi_document_type(
            "Monetary Policy Statement, 2026-27",
            "",
            VALID_MONETARY_POLICY_STATEMENT_TEXT,
        )
        == "monetary_policy_statement"
    )


def test_valid_governor_speech_is_not_excluded() -> None:
    is_index_page, reason = is_rbi_index_or_navigation_page(
        "Governor speech on monetary policy",
        VALID_GOVERNOR_SPEECH_TEXT,
        "https://www.rbi.org.in/Scripts/BS_SpeechesView.aspx?Id=1",
    )

    assert is_index_page is False
    assert reason == ""
    assert (
        classify_rbi_document_type(
            "Governor speech on monetary policy",
            "",
            VALID_GOVERNOR_SPEECH_TEXT,
        )
        == "governor_speech"
    )


def test_substantive_policy_page_with_sidebar_boilerplate_is_not_excluded() -> None:
    text = (
        VALID_MONETARY_POLICY_STATEMENT_TEXT
        + "\n"
        + "\n".join([VALID_MPC_MINUTES_TEXT] * 8)
        + "\nAnnual Report\nArchives\n2026\n2025\n2024\n2023\n"
        + "Note : To read the chapter of your choice, please click on the links below."
    )

    is_index_page, reason = is_rbi_index_or_navigation_page(
        "Edited Transcript of the Reserve Bank of India's Post-Monetary Policy Press Conference",
        text,
        "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=2",
    )

    assert is_index_page is False
    assert reason == ""
    assert (
        classify_rbi_document_type(
            "Edited Transcript of the Reserve Bank of India's Post-Monetary Policy Press Conference",
            "",
            text,
        )
        == "monetary_policy_statement"
    )
