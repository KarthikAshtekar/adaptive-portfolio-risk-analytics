"""Tests for sparse, lagged RBI macro stance index construction."""

from __future__ import annotations

import pandas as pd

from src.sentiment import build_current_macro_summary, build_macro_stance_index


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sentence_id": "doc_1_s0000",
                "document_id": "doc_1",
                "publication_date": pd.Timestamp("2024-01-05"),
                "stance_label": "hawkish",
                "certainty_label": "uncertain",
                "time_label": "forward_looking",
            },
            {
                "sentence_id": "doc_1_s0001",
                "document_id": "doc_1",
                "publication_date": pd.Timestamp("2024-01-05"),
                "stance_label": "neutral",
                "certainty_label": "certain",
                "time_label": "current",
            },
        ]
    )


def test_index_has_required_columns_and_one_session_lag() -> None:
    market_index = pd.bdate_range("2024-01-05", periods=5)
    macro = build_macro_stance_index(
        _scores(),
        market_index,
        lookback_window=2,
        decision_lag=1,
    )

    required = {
        "hawkish_share",
        "dovish_share",
        "uncertainty_share",
        "forward_looking_share",
        "net_stance_score",
        "macro_risk_score",
        "macro_label",
        "decision_macro_label",
        "document_count",
        "sentence_count",
        "coverage_flag",
    }
    assert required.issubset(macro.columns)
    assert macro.loc["2024-01-05", "decision_macro_label"] == "insufficient_macro_data"
    assert macro.loc["2024-01-08", "decision_macro_label"] == "risk_off_macro"
    assert macro.loc["2024-01-08", "decision_source_date"] < pd.Timestamp(
        "2024-01-08"
    )


def test_day_t_document_does_not_change_day_t_decision() -> None:
    market_index = pd.bdate_range("2024-01-05", periods=5)
    baseline = build_macro_stance_index(
        _scores(),
        market_index,
        lookback_window=1,
        decision_lag=1,
    )
    changed_scores = pd.concat(
        [
            _scores(),
            pd.DataFrame(
                [
                    {
                        "sentence_id": "doc_2_s0000",
                        "document_id": "doc_2",
                        "publication_date": pd.Timestamp("2024-01-08"),
                        "stance_label": "dovish",
                        "certainty_label": "certain",
                        "time_label": "forward_looking",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    changed = build_macro_stance_index(
        changed_scores,
        market_index,
        lookback_window=1,
        decision_lag=1,
    )

    assert (
        baseline.loc["2024-01-08", "decision_macro_label"]
        == changed.loc["2024-01-08", "decision_macro_label"]
    )


def test_sparse_publications_produce_coverage_warning() -> None:
    market_index = pd.bdate_range("2024-01-05", periods=8)
    macro = build_macro_stance_index(
        _scores(),
        market_index,
        lookback_window=2,
        decision_lag=1,
    )

    current = build_current_macro_summary(macro, "Normal")

    assert current["macro_sentiment_coverage"] == 0
    assert current["coverage_status"].startswith("Insufficient")
    assert "coverage is insufficient" in current["macro_sentiment_warning"]
