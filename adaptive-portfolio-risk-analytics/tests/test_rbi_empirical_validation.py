"""Tests for the real-RBI empirical validation runner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.sentiment import EMPIRICAL_OUTPUT_FILES, run_rbi_empirical_validation


def test_empirical_validation_writes_metadata_complete_outputs(
    tmp_path: Path,
) -> None:
    (tmp_path / "minutes.txt").write_text(
        "Inflation remains elevated and upside risks remain uncertain. "
        "Policy will remain vigilant going forward.",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "document_id": "rbi_minutes_20240105",
                "publication_date": "2024-01-05",
                "document_type": "mpc_minutes",
                "title": "MPC Minutes",
                "local_path": "minutes.txt",
                "source_url": "https://www.rbi.org.in/minutes",
                "retrieval_date": "2026-06-21",
                "language": "en",
                "notes": "",
            }
        ]
    ).to_csv(tmp_path / "manifest.csv", index=False)
    index = pd.bdate_range("2024-01-05", periods=10)
    returns = pd.DataFrame({"A": 0.0, "B": 0.0}, index=index)
    rule = pd.Series("Normal", index=index)
    hmm = pd.Series("Risk-On", index=index)
    output = tmp_path / "output"

    result = run_rbi_empirical_validation(
        tmp_path / "manifest.csv",
        returns,
        rule,
        hmm,
        output,
        decision_lag=1,
        lookback_window=3,
    )

    assert {path.name for path in output.glob("*.csv")} == set(
        EMPIRICAL_OUTPUT_FILES
    )
    for key in (
        "rbi_documents",
        "rbi_sentence_scores",
        "macro_stance_index",
        "macro_regime_comparison",
        "disagreement_dates",
        "coverage_diagnostics",
        "corpus_diagnostics",
    ):
        assert result[key]["corpus_type"].eq("real_rbi").all()
        assert result[key]["decision_lag"].eq(1).all()
        assert result[key]["lookback_window"].eq(3).all()

    macro = result["macro_stance_index"]
    assert macro.loc[index[0], "decision_macro_label"] == (
        "insufficient_macro_data"
    )
    assert macro.loc[index[1], "decision_source_date"] < index[1]
