"""Tests for dependency-light sentiment scoring."""

from __future__ import annotations

import pandas as pd

from src.sentiment import score_sentiment_records


def test_lexicon_scorer_labels_risk_off_headline_negative() -> None:
    records = pd.DataFrame(
        [{"title": "Banking stress triggers selloff", "text": "Panic after downgrade"}]
    )

    scored = score_sentiment_records(records)

    assert scored.iloc[0]["sentiment_score"] < 0
    assert scored.iloc[0]["sentiment_label"] == "risk_off"


def test_lexicon_scorer_labels_risk_on_headline_positive() -> None:
    records = pd.DataFrame(
        [{"title": "Market rally gains momentum", "text": "Strong earnings support growth"}]
    )

    scored = score_sentiment_records(records)

    assert scored.iloc[0]["sentiment_score"] > 0
    assert scored.iloc[0]["sentiment_label"] == "risk_on"
    assert scored.iloc[0]["model_name"] == "phase4a_lexicon"

