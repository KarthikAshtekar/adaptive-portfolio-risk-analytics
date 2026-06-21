"""Tests for sentiment and quantitative-regime confirmation analytics."""

from __future__ import annotations

import pandas as pd

from src.sentiment import (
    calculate_sentiment_confirmation_score,
    compare_sentiment_to_regimes,
)


def test_sentiment_regime_agreement_and_confusion_matrices() -> None:
    index = pd.bdate_range("2024-01-01", periods=4)
    signal = pd.DataFrame(
        {
            "decision_sentiment_label": [
                "risk_on",
                "neutral",
                "risk_off",
                "risk_on",
            ],
            "article_count": [1, 1, 1, 1],
            "decision_article_count": [1, 1, 1, 1],
        },
        index=index,
    )
    rule = pd.Series(["Calm", "Normal", "Stress", "Crisis"], index=index)
    hmm = pd.Series(["Risk-On", "Normal", "Risk-Off", "Risk-Off"], index=index)

    comparison = compare_sentiment_to_regimes(signal, rule, hmm)

    assert comparison["agreement_with_rule_based"] == 0.75
    assert comparison["agreement_with_hmm"] == 0.75
    assert comparison["risk_off_agreement_rule_based"] == 0.5
    assert not comparison["confusion_matrix_rule_based"].empty
    assert len(comparison["dates_of_major_disagreement"]) == 1


def test_confirmation_status_handles_agreement_disagreement_and_insufficient_data() -> None:
    assert (
        calculate_sentiment_confirmation_score(
            "Stress",
            "risk_off",
            article_count=2,
        )
        == "Confirmed Risk-Off"
    )
    assert (
        calculate_sentiment_confirmation_score(
            "Stress",
            "risk_on",
            article_count=2,
        )
        == "Quant-Sentiment Disagreement"
    )
    assert (
        calculate_sentiment_confirmation_score(
            "Stress",
            "unknown",
            article_count=0,
        )
        == "Insufficient Sentiment Data"
    )

