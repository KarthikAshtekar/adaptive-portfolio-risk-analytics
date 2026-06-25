"""Corpus status tests for RBI policy-core document coverage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.check_rbi_corpus_status import build_rbi_corpus_status
from src.sentiment.rbi_corpus_builder import REAL_RBI_MANIFEST_COLUMNS


def _write_manifest(tmp_path: Path, document_types: list[str]) -> Path:
    corpus_dir = tmp_path / "rbi_real"
    raw_dir = corpus_dir / "raw"
    raw_dir.mkdir(parents=True)
    rows = []
    for index, document_type in enumerate(document_types, start=1):
        document_id = f"rbi_test_{index}"
        text_path = raw_dir / f"{document_id}.txt"
        text_path.write_text(
            (
                "Reserve Bank of India monetary policy document. "
                "The text discusses inflation, liquidity, growth, policy repo "
                "rate, financial conditions, and macroeconomic assessment."
            ),
            encoding="utf-8",
        )
        rows.append(
            {
                "document_id": document_id,
                "publication_date": f"2026-06-{index:02d}",
                "document_type": document_type,
                "title": f"{document_type} test document {index}",
                "local_path": f"raw/{document_id}.txt",
                "source_url": f"https://www.rbi.org.in/test/{index}",
                "retrieval_date": "2026-06-25",
                "language": "en",
                "notes": "test",
            }
        )
    manifest = corpus_dir / "manifest.csv"
    pd.DataFrame(rows, columns=REAL_RBI_MANIFEST_COLUMNS).to_csv(
        manifest,
        index=False,
    )
    return manifest


def test_policy_core_documents_are_counted_correctly(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        [
            "mpc_minutes",
            "mpc_minutes",
            "monetary_policy_statement",
            "governor_speech",
            "financial_stability_report",
        ],
    )

    status, diagnostics = build_rbi_corpus_status(manifest)

    assert status["mpc_minutes_count"] == 2
    assert status["monetary_policy_statement_count"] == 1
    assert status["governor_speech_count"] == 1
    assert status["financial_stability_report_count"] == 1
    assert status["policy_core_documents"] == 3
    assert (
        diagnostics.loc[
            diagnostics["metric"].eq("policy_core_documents"),
            "actual",
        ].iloc[0]
        == 3
    )


def test_manual_action_remains_yes_when_policy_core_documents_below_threshold(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(
        tmp_path,
        [
            "mpc_minutes",
            "monetary_policy_statement",
            "governor_speech",
            "governor_speech",
            "financial_stability_report",
            "press_release",
            "governor_speech",
            "financial_stability_report",
            "press_release",
            "governor_speech",
        ],
    )

    status, _ = build_rbi_corpus_status(manifest)

    assert status["valid_document_count"] == 10
    assert status["policy_core_documents"] == 2
    assert status["minimum_requirements_passed"] is False
    assert status["manual_action_required"] is True
