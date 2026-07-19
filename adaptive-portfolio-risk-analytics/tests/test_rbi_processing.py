"""Tests for stable RBI sentence extraction."""

from __future__ import annotations

import pandas as pd

from src.sentiment import (
    clean_rbi_sentence,
    split_rbi_documents_into_sentences,
)


def test_sentence_processing_preserves_order_metadata_and_stable_ids() -> None:
    documents = pd.DataFrame(
        [
            {
                "document_id": "doc_1",
                "publication_date": pd.Timestamp("2024-01-05"),
                "document_type": "mpc_minutes",
                "title": "Minutes",
                "source": "Fixture",
                "load_status": "loaded",
                "manifest_order": 0,
                "text": (
                    "Reserve Bank of India\n"
                    "Inflation remains elevated and policy will remain vigilant. "
                    "Growth could improve going forward as demand recovers.\n"
                    "Page 1"
                ),
            }
        ]
    )

    first = split_rbi_documents_into_sentences(documents)
    second = split_rbi_documents_into_sentences(documents)

    assert first["sentence_id"].tolist() == ["doc_1_s0000", "doc_1_s0001"]
    assert first["sentence_order"].tolist() == [0, 1]
    assert first["document_type"].eq("mpc_minutes").all()
    assert first["publication_date"].eq(pd.Timestamp("2024-01-05")).all()
    assert first["sentence"].equals(first["sentence_text"])
    pd.testing.assert_frame_equal(first, second)


def test_clean_rbi_sentence_removes_excess_whitespace() -> None:
    assert clean_rbi_sentence("  Policy   remains\tvigilant.  ") == ("Policy remains vigilant.")
