"""Real NLP validation report tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.validate_real_nlp_signal import (
    VERDICT_C,
    validate_real_nlp_signal,
)


def test_sparse_records_generate_insufficiency_report(tmp_path: Path) -> None:
    records = pd.DataFrame(
        [
            {
                "record_id": "real-1",
                "timestamp": "2026-06-01T09:00:00Z",
                "publication_time": "2026-06-01T09:00:00Z",
                "retrieval_time": "2026-06-01T10:00:00Z",
                "source": "Reserve Bank of India",
                "provider": "rbi",
                "document_type": "monetary_policy_statement",
                "entity": "Reserve Bank of India",
                "ticker": "",
                "sector": "central_bank",
                "country": "IN",
                "title": "Policy statement",
                "text": "Inflation risk remains elevated.",
                "url": "https://www.rbi.org.in/policy/real-1",
                "language": "en",
                "raw_metadata": "{}",
            }
        ]
    )
    input_path = tmp_path / "records.csv"
    output = tmp_path / "report"
    records.to_csv(input_path, index=False)

    result = validate_real_nlp_signal(
        input_records=input_path,
        start_date="2026-01-01",
        end_date="2026-06-21",
        output_dir=output,
    )

    assert result["summary"]["verdict"] == VERDICT_C
    assert result["summary"]["predictiveness_claim"] is False
    assert (output / "report.html").is_file()
    assert (output / "summary.md").is_file()
    assert "Insufficient real-data coverage" in (output / "report.html").read_text(encoding="utf-8")
