"""Composite source-mix labeling for RBI and news monitoring."""

from __future__ import annotations

import pandas as pd

from src.sentiment.composite_index import (
    build_composite_nlp_risk_index,
)
from src.sentiment.scoring import score_sentiment_records


def _news_records(score_text: str = "Inflation shock and rate hike concerns") -> pd.DataFrame:
    records = pd.DataFrame(
        [
            {
                "provider": "gdelt",
                "document_type": "news",
                "title": score_text,
                "text": "",
                "publication_time": "2026-04-01T09:00:00Z",
                "decision_available_date": "2026-04-02T00:00:00Z",
                "is_ex_ante_valid": True,
                "possible_reaction_data": False,
            }
        ]
    )
    return score_sentiment_records(records)


def _rbi_macro(index: pd.DatetimeIndex, score: float = 0.4) -> pd.DataFrame:
    frame = pd.DataFrame(index=index)
    frame["macro_risk_score"] = pd.NA
    frame.loc[index[1], "macro_risk_score"] = score
    return frame


def _covered_mix(result: pd.DataFrame) -> set[str]:
    covered = result.loc[result["decision_composite_nlp_label"].ne("insufficient_nlp_data")]
    return set(covered["decision_source_mix"].dropna().astype(str))


def test_news_only_source_mix_remains_news_only() -> None:
    index = pd.bdate_range("2026-04-01", periods=8)

    result = build_composite_nlp_risk_index(
        news_sentiment=_news_records(),
        market_index=index,
        decision_lag=1,
    )

    assert "news_only" in _covered_mix(result)


def test_rbi_only_source_mix_is_labeled_rbi_only() -> None:
    index = pd.bdate_range("2026-04-01", periods=8)

    result = build_composite_nlp_risk_index(
        rbi_macro_index=_rbi_macro(index),
        market_index=index,
        decision_lag=1,
    )

    assert "rbi_only" in _covered_mix(result)


def test_rbi_and_news_source_mix_is_labeled_explicitly() -> None:
    index = pd.bdate_range("2026-04-01", periods=8)

    result = build_composite_nlp_risk_index(
        rbi_macro_index=_rbi_macro(index),
        news_sentiment=_news_records(),
        market_index=index,
        decision_lag=1,
    )

    assert "rbi_and_news" in _covered_mix(result)
    assert bool(result["commentary_only"].all()) is True
