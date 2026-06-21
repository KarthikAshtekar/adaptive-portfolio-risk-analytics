"""Composite ex-ante NLP risk index tests."""

from __future__ import annotations

import pandas as pd
import pytest

from src.sentiment.composite_index import build_composite_nlp_risk_index
from src.sentiment.nlp_regime_comparison import compare_composite_nlp_to_regimes


def _records(score: float, date: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_available_date": date,
                "sentiment_score": score,
                "is_ex_ante_valid": True,
                "possible_reaction_data": False,
            }
        ]
    )


def test_composite_index_rejects_market_returns_as_input() -> None:
    returns = pd.Series(
        0.01,
        index=pd.bdate_range("2024-01-01", periods=10),
    )

    with pytest.raises(TypeError, match="market returns are not inputs"):
        build_composite_nlp_risk_index(market_index=returns)


def test_composite_index_marks_poor_coverage_insufficient() -> None:
    index = pd.bdate_range("2024-01-01", periods=10)

    result = build_composite_nlp_risk_index(
        earnings_sentiment=_records(-0.8, "2024-01-03"),
        market_index=index,
        decision_lag=1,
    )

    assert set(result["decision_composite_nlp_label"]) == {
        "insufficient_nlp_data"
    }


def test_composite_index_is_lagged_and_compares_with_regimes() -> None:
    index = pd.bdate_range("2024-01-01", periods=12)
    result = build_composite_nlp_risk_index(
        earnings_sentiment=_records(-0.8, "2024-01-03"),
        news_sentiment=_records(-0.6, "2024-01-03"),
        market_index=index,
        decision_lag=1,
    )
    regimes = pd.Series("Stress", index=index)

    comparison = compare_composite_nlp_to_regimes(result, regimes, regimes)

    covered = result["decision_composite_nlp_label"].ne(
        "insufficient_nlp_data"
    )
    assert covered.any()
    first_covered = result.index[covered][0]
    assert result.loc[first_covered, "decision_source_date"] < first_covered
    assert comparison["agreement_with_rule_based"] == 1.0
