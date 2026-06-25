"""GDELT/news signal construction tests."""

from __future__ import annotations

import pandas as pd

from src.sentiment.composite_index import (
    VALID_COMPOSITE_NLP_LABELS,
    build_composite_nlp_risk_index,
)
from src.sentiment.scoring import score_sentiment_records


def test_real_like_gdelt_title_produces_sentiment_and_risk_scores() -> None:
    records = pd.DataFrame(
        [
            {
                "provider": "gdelt",
                "document_type": "news",
                "title": "Inflation shock and rate hike concerns hit India outlook",
                "text": "",
                "publication_time": "2026-04-01T09:00:00Z",
                "decision_available_date": "2026-04-02T00:00:00Z",
                "is_ex_ante_valid": True,
                "possible_reaction_data": False,
            }
        ]
    )

    scored = score_sentiment_records(records)

    row = scored.iloc[0]
    assert row["sentiment_score"] < 0
    assert row["sentiment_label"] == "risk_off"
    assert row["risk_score"] > 0
    assert row["risk_label"] == "risk_off"
    assert row["scoring_method_used"] == "lexicon"
    assert row["model_name"] == "phase4a_lexicon"
    assert row["model_version"] == "1.0"


def test_news_only_gdelt_records_produce_monitoring_composite_labels() -> None:
    market_index = pd.bdate_range("2026-04-01", periods=8)
    records = pd.DataFrame(
        [
            {
                "provider": "gdelt",
                "document_type": "news",
                "title": "Inflation shock and rate hike concerns hit India outlook",
                "text": "",
                "publication_time": "2026-04-01T09:00:00Z",
                "decision_available_date": "2026-04-02T00:00:00Z",
                "is_ex_ante_valid": True,
                "possible_reaction_data": False,
            }
        ]
    )
    scored = score_sentiment_records(records)

    composite = build_composite_nlp_risk_index(
        news_sentiment=scored,
        market_index=market_index,
        decision_lag=1,
    )

    valid = composite["decision_composite_nlp_label"].isin(
        VALID_COMPOSITE_NLP_LABELS
    )
    assert valid.any()
    first_valid = composite.loc[valid].iloc[0]
    assert first_valid["decision_source_mix"] == "news_only"
    assert first_valid["decision_coverage_score"] == 1 / 3
    assert first_valid["decision_composite_nlp_label"] == "nlp_risk_off"
    assert bool(composite["commentary_only"].all()) is True
