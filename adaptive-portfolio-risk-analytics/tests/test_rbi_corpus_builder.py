"""Tests for the Phase 4A.3 real-RBI corpus builder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.sentiment import (
    REAL_RBI_MANIFEST_COLUMNS,
    build_rbi_manifest_from_directory,
    load_real_rbi_corpus,
    validate_rbi_manifest,
)


def _manifest_row(local_path: str, **overrides) -> dict[str, str]:
    row = {
        "document_id": "rbi_mpc_20240105",
        "publication_date": "2024-01-05",
        "document_type": "mpc_minutes",
        "title": "MPC Minutes January 2024",
        "local_path": local_path,
        "source_url": "https://www.rbi.org.in/example",
        "retrieval_date": "2026-06-21",
        "language": "en",
        "notes": "Public RBI communication",
    }
    row.update(overrides)
    return row


def test_manifest_builder_creates_exact_manifest_from_local_directory(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "rbi_mpc_2024-01-05.txt").write_text(
        "Inflation remains elevated. Policy will remain data dependent.",
        encoding="utf-8",
    )

    manifest = build_rbi_manifest_from_directory(
        raw,
        tmp_path / "manifest.csv",
    )

    assert tuple(manifest.columns) == REAL_RBI_MANIFEST_COLUMNS
    assert manifest.iloc[0]["publication_date"] == "2024-01-05"
    assert manifest.iloc[0]["document_type"] == "mpc_minutes"


def test_validator_catches_missing_required_columns(tmp_path: Path) -> None:
    pd.DataFrame([{"document_id": "doc"}]).to_csv(
        tmp_path / "manifest.csv",
        index=False,
    )

    result = validate_rbi_manifest(tmp_path / "manifest.csv")

    assert "source_url" in result["missing_required_columns"]
    assert result["is_valid"] is False


def test_validator_flags_missing_files_duplicates_and_empty_documents(
    tmp_path: Path,
) -> None:
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    manifest = pd.DataFrame(
        [
            _manifest_row("missing.txt"),
            _manifest_row(
                "empty.txt",
                title="Empty",
                source_url="https://www.rbi.org.in/empty",
            ),
            _manifest_row(
                "missing-2.txt",
                title="Duplicate ID",
                source_url="https://www.rbi.org.in/duplicate",
            ),
        ]
    )
    manifest.to_csv(tmp_path / "manifest.csv", index=False)

    result = validate_rbi_manifest(tmp_path / "manifest.csv")

    assert result["summary"]["invalid_document_count"] == 3
    assert result["summary"]["duplicate_record_count"] >= 2
    assert result["invalid_documents"]["validation_errors"].str.contains(
        "local file not found|document text is empty|duplicate document_id",
        regex=True,
    ).any()


def test_real_corpus_loader_returns_only_valid_documents(
    tmp_path: Path,
) -> None:
    (tmp_path / "valid.txt").write_text(
        "The inflation outlook remains uncertain. Policy will remain vigilant.",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            _manifest_row("valid.txt"),
            _manifest_row(
                "missing.txt",
                document_id="missing_doc",
                title="Missing",
                source_url="https://www.rbi.org.in/missing",
            ),
        ]
    ).to_csv(tmp_path / "manifest.csv", index=False)

    documents = load_real_rbi_corpus(tmp_path / "manifest.csv")

    assert documents["document_id"].tolist() == ["rbi_mpc_20240105"]
    assert documents["corpus_type"].eq("real_rbi").all()
    assert documents.attrs["diagnostics"]["invalid_document_count"] == 1
