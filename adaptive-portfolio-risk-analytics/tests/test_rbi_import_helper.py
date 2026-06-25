"""Tests for RBI real-corpus bootstrap and import helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.bootstrap_rbi_real_corpus import bootstrap_rbi_real_corpus
from scripts.import_rbi_text_document import build_parser, import_rbi_text_document
from src.sentiment import REAL_RBI_DOCUMENT_TYPES, REAL_RBI_MANIFEST_COLUMNS
from src.sentiment.corpus_intake import RBI_DOCUMENT_TYPES as INTAKE_RBI_DOCUMENT_TYPES
from src.sentiment.schema import RBI_DOCUMENT_TYPES as SCHEMA_RBI_DOCUMENT_TYPES


EXPECTED_REAL_RBI_DOCUMENT_TYPES = (
    "mpc_minutes",
    "monetary_policy_statement",
    "governor_speech",
    "press_release",
    "financial_stability_report",
    "annual_report",
    "unknown",
)


def test_real_rbi_document_type_contract_is_exact() -> None:
    assert REAL_RBI_DOCUMENT_TYPES == EXPECTED_REAL_RBI_DOCUMENT_TYPES
    assert SCHEMA_RBI_DOCUMENT_TYPES == EXPECTED_REAL_RBI_DOCUMENT_TYPES
    assert INTAKE_RBI_DOCUMENT_TYPES == set(EXPECTED_REAL_RBI_DOCUMENT_TYPES)

    parser = build_parser()
    document_type_action = next(
        action for action in parser._actions if action.dest == "document_type"
    )
    assert tuple(document_type_action.choices) == EXPECTED_REAL_RBI_DOCUMENT_TYPES


def test_bootstrap_creates_empty_manifest_and_directories(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "rbi_real"

    result = bootstrap_rbi_real_corpus(corpus_dir=corpus_dir)

    assert Path(result["raw_dir"]).is_dir()
    assert Path(result["processed_dir"]).is_dir()
    manifest = Path(result["manifest_path"])
    assert manifest.is_file()
    frame = pd.read_csv(manifest)
    assert tuple(frame.columns) == REAL_RBI_MANIFEST_COLUMNS
    assert frame.empty


def test_import_helper_adds_valid_rbi_text_file(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "rbi_real"
    bootstrap_rbi_real_corpus(corpus_dir=corpus_dir)
    input_text = tmp_path / "source.txt"
    input_text.write_text(
        "Inflation remains elevated. Policy will remain data dependent.",
        encoding="utf-8",
    )

    result = import_rbi_text_document(
        document_id="RBI_MPC_2026_06",
        publication_date="2026-06-06",
        document_type="mpc_minutes",
        title="MPC Minutes June 2026",
        source_url="https://www.rbi.org.in/example",
        input_text_file=input_text,
        retrieval_date="2026-06-23",
        manifest_path=corpus_dir / "manifest.csv",
    )

    manifest = pd.read_csv(corpus_dir / "manifest.csv")
    assert manifest["document_id"].tolist() == ["RBI_MPC_2026_06"]
    assert (corpus_dir / "raw" / "RBI_MPC_2026_06.txt").is_file()
    assert result["validation"]["summary"]["valid_document_count"] == 1


def test_import_helper_blocks_duplicate_without_overwrite(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "rbi_real"
    bootstrap_rbi_real_corpus(corpus_dir=corpus_dir)
    input_text = tmp_path / "source.txt"
    input_text.write_text(
        "Inflation remains elevated. Policy will remain vigilant.",
        encoding="utf-8",
    )
    kwargs = {
        "document_id": "RBI_DUPLICATE",
        "publication_date": "2026-06-06",
        "document_type": "mpc_minutes",
        "title": "MPC Minutes June 2026",
        "source_url": "https://www.rbi.org.in/example",
        "input_text_file": input_text,
        "retrieval_date": "2026-06-23",
        "manifest_path": corpus_dir / "manifest.csv",
    }
    import_rbi_text_document(**kwargs)

    with pytest.raises(ValueError, match="already exists"):
        import_rbi_text_document(**kwargs)

    overwritten = import_rbi_text_document(
        **{**kwargs, "title": "Updated MPC Minutes"},
        overwrite=True,
    )
    manifest = pd.read_csv(corpus_dir / "manifest.csv")
    assert len(manifest) == 1
    assert manifest.iloc[0]["title"] == "Updated MPC Minutes"
    assert overwritten["overwrote_existing"] is True


def test_import_helper_rejects_invalid_type_and_empty_text(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "rbi_real"
    bootstrap_rbi_real_corpus(corpus_dir=corpus_dir)
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    text = tmp_path / "source.txt"
    text.write_text("Policy will remain vigilant.", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid document_type"):
        import_rbi_text_document(
            document_id="RBI_BAD_TYPE",
            publication_date="2026-06-06",
            document_type="blog_post",
            title="Bad Type",
            source_url="https://www.rbi.org.in/example",
            input_text_file=text,
            retrieval_date="2026-06-23",
            manifest_path=corpus_dir / "manifest.csv",
        )

    with pytest.raises(ValueError, match="empty"):
        import_rbi_text_document(
            document_id="RBI_EMPTY",
            publication_date="2026-06-06",
            document_type="mpc_minutes",
            title="Empty",
            source_url="https://www.rbi.org.in/example",
            input_text_file=empty,
            retrieval_date="2026-06-23",
            manifest_path=corpus_dir / "manifest.csv",
        )
