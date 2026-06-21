"""Corpus intake validator tests for RBI, earnings, and news."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.sentiment import validate_corpus_manifest, validate_nlp_corpus_intake


def test_missing_manifests_require_manual_action(tmp_path: Path) -> None:
    result = validate_nlp_corpus_intake(
        rbi_manifest=tmp_path / "rbi.csv",
        earnings_manifest=tmp_path / "earnings.csv",
        news_manifest=tmp_path / "news.csv",
    )

    assert result["manual_action_required"] is True
    assert result["intake_status"]["manifest_exists"].eq(False).all()
    assert result["intake_status"]["corpus_status"].eq(
        "manual_action_required"
    ).all()


def test_valid_small_corpora_pass_intake_validation(tmp_path: Path) -> None:
    (tmp_path / "rbi.txt").write_text("Policy remains vigilant.", encoding="utf-8")
    (tmp_path / "earnings.txt").write_text(
        "Demand remains resilient.", encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "document_id": "rbi-1",
                "publication_date": "2024-01-05",
                "document_type": "mpc_minutes",
                "title": "MPC Minutes",
                "local_path": "rbi.txt",
                "source_url": "https://www.rbi.org.in/real",
                "retrieval_date": "2024-01-06",
                "language": "en",
                "notes": "Official public document",
            }
        ]
    ).to_csv(tmp_path / "rbi.csv", index=False)
    pd.DataFrame(
        [
            {
                "document_id": "earnings-1",
                "company": "Real Bank",
                "ticker": "REALBANK.NS",
                "sector": "Banking",
                "quarter": "Q1 FY2024",
                "publication_date": "2024-04-20",
                "title": "Real Bank earnings call",
                "local_path": "earnings.txt",
                "source_url": "https://investor.realbank.test/call",
                "retrieval_date": "2024-04-21",
                "language": "en",
                "notes": "Legally available transcript",
            }
        ]
    ).to_csv(tmp_path / "earnings.csv", index=False)
    pd.DataFrame(
        [
            {
                "record_id": "news-1",
                "publication_time": "2024-05-01T09:00:00Z",
                "source": "Official Agency",
                "provider": "local_manifest",
                "document_type": "financial_news",
                "entity": "India",
                "ticker": "",
                "sector": "Macro",
                "country": "IN",
                "title": "Inflation outlook update",
                "text": "Inflation risks remain elevated.",
                "url": "https://agency.test/inflation",
                "language": "en",
                "retrieval_time": "2024-05-01T10:00:00Z",
                "notes": "Permitted summary",
            }
        ]
    ).to_csv(tmp_path / "news.csv", index=False)

    result = validate_nlp_corpus_intake(
        rbi_manifest=tmp_path / "rbi.csv",
        earnings_manifest=tmp_path / "earnings.csv",
        news_manifest=tmp_path / "news.csv",
    )

    assert result["all_corpora_ready"] is True
    assert result["manual_action_required"] is False
    assert result["valid_real_records_by_corpus"] == {
        "rbi": 1,
        "earnings": 1,
        "news": 1,
    }


def test_duplicates_missing_files_bad_dates_and_news_time_are_flagged(
    tmp_path: Path,
) -> None:
    rbi = pd.DataFrame(
        [
            {
                "document_id": "duplicate",
                "publication_date": "bad-date",
                "document_type": "mpc_minutes",
                "title": "Minutes",
                "local_path": "missing.txt",
                "source_url": "https://www.rbi.org.in/a",
                "retrieval_date": "2024-01-02",
                "language": "en",
                "notes": "",
            },
            {
                "document_id": "duplicate",
                "publication_date": "2024-01-01",
                "document_type": "mpc_minutes",
                "title": "Other Minutes",
                "local_path": "missing-2.txt",
                "source_url": "https://www.rbi.org.in/b",
                "retrieval_date": "2024-01-02",
                "language": "en",
                "notes": "",
            },
        ]
    )
    rbi.to_csv(tmp_path / "rbi.csv", index=False)
    rbi_result = validate_corpus_manifest("rbi", tmp_path / "rbi.csv")
    errors = " ".join(rbi_result["rows"]["validation_errors"].tolist())

    assert "duplicate document_id" in errors
    assert "local file not found" in errors
    assert "invalid publication_date" in errors

    news = pd.DataFrame(
        [
            {
                "record_id": "news-1",
                "publication_time": "2024-05-02T10:00:00Z",
                "source": "Source",
                "provider": "Provider",
                "document_type": "financial_news",
                "entity": "India",
                "ticker": "",
                "sector": "Macro",
                "country": "IN",
                "title": "Article",
                "text": "Forward-looking policy discussion.",
                "url": "https://source.test/article",
                "language": "en",
                "retrieval_time": "2024-05-01T10:00:00Z",
                "notes": "",
            }
        ]
    )
    news.to_csv(tmp_path / "news.csv", index=False)
    news_result = validate_corpus_manifest("news", tmp_path / "news.csv")
    assert news_result["rows"]["validation_errors"].str.contains(
        "publication_time after retrieval_time"
    ).any()
