"""Tests for manifest-driven RBI document ingestion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.sentiment import RBIDocument, load_rbi_documents


def test_loads_text_and_flags_one_bad_row_without_failing_all(tmp_path: Path) -> None:
    (tmp_path / "good.txt").write_text(
        "Inflation remains elevated and policy will remain vigilant.",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "document_id": "good_doc",
                "publication_date": "2024-01-05",
                "title": "Good",
                "document_type": "mpc_minutes",
                "local_path": "good.txt",
                "source_url": "https://example.test/good",
            },
            {
                "document_id": "bad doc id",
                "publication_date": "invalid",
                "title": "Bad",
                "document_type": "other",
                "local_path": "missing.txt",
                "source_url": "",
            },
        ]
    ).to_csv(tmp_path / "manifest.csv", index=False)

    documents = load_rbi_documents(tmp_path / "manifest.csv")

    assert documents["load_status"].tolist() == ["loaded", "error"]
    assert "Inflation remains elevated" in documents.iloc[0]["text"]
    assert documents.iloc[1]["document_type"] == "unknown"
    assert documents.iloc[1]["error"]


def test_csv_document_prefers_text_column(tmp_path: Path) -> None:
    pd.DataFrame({"text": ["First sentence.", "Second sentence."]}).to_csv(
        tmp_path / "document.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "document_id": "csv_doc",
                "publication_date": "2024-01-05",
                "title": "CSV",
                "document_type": "macro_report",
                "file_path": "document.csv",
            }
        ]
    ).to_csv(tmp_path / "manifest.csv", index=False)

    documents = load_rbi_documents(tmp_path / "manifest.csv")

    assert documents.iloc[0]["text"] == "First sentence.\nSecond sentence."


def test_rbi_document_schema_matches_public_contract() -> None:
    document = RBIDocument(
        document_id="doc_1",
        publication_date=pd.Timestamp("2024-01-05"),
        document_type="mpc_minutes",
        title="Minutes",
        text="Policy will remain data dependent.",
        source_url="https://example.test/doc_1",
        local_path="doc_1.txt",
        language="en",
    )

    assert document.local_path == "doc_1.txt"
    assert document.source_url.endswith("doc_1")
    assert document.language == "en"


def test_ingestion_records_diagnostics(tmp_path: Path) -> None:
    (tmp_path / "doc.txt").write_text("Policy remains vigilant.", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "document_id": "doc",
                "publication_date": "2024-01-05",
                "document_type": "press_release",
                "title": "Release",
                "local_path": "doc.txt",
                "source_url": "",
            }
        ]
    ).to_csv(tmp_path / "manifest.csv", index=False)

    documents = load_rbi_documents(tmp_path / "manifest.csv")

    assert documents.attrs["diagnostics"]["loaded_document_count"] == 1
    assert documents.attrs["diagnostics"]["error_document_count"] == 0
