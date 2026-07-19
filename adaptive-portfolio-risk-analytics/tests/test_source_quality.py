"""Source-quality and provenance scoring tests."""

from __future__ import annotations

import pandas as pd

from src.sentiment import score_source_quality


def _official_record(**overrides) -> dict[str, object]:
    record = {
        "record_id": "rbi-1",
        "publication_time": "2026-06-01T09:00:00Z",
        "retrieval_time": "2026-06-01T10:00:00Z",
        "provider": "rbi",
        "source": "Reserve Bank of India",
        "entity": "Reserve Bank of India",
        "ticker": "",
        "sector": "central_bank",
        "query": "monetary policy",
        "title": "Monetary Policy Statement",
        "url": "https://www.rbi.org.in/policy/statement",
        "language": "en",
        "raw_metadata": "{}",
        "possible_reaction_data": False,
    }
    record.update(overrides)
    return record


def test_official_record_is_high_quality_and_real_candidate() -> None:
    scored = score_source_quality(pd.DataFrame([_official_record()]))
    row = scored.iloc[0]

    assert row["official_source"]
    assert row["source_quality_label"] == "high"
    assert row["is_real_provider_data"]


def test_reaction_warning_reduces_source_quality_score() -> None:
    records = pd.DataFrame(
        [
            _official_record(record_id="clean"),
            _official_record(
                record_id="reaction",
                url="https://www.rbi.org.in/policy/reaction",
                possible_reaction_data=True,
            ),
        ]
    )
    scored = score_source_quality(records).set_index("record_id")

    assert (
        scored.loc["reaction", "source_quality_score"] < scored.loc["clean", "source_quality_score"]
    )
    assert "possible reaction data" in scored.loc["reaction", "source_quality_warning"]
