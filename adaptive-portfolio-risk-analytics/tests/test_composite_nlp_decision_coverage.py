"""Decision-label coverage tests for composite NLP monitoring."""

from __future__ import annotations

import pandas as pd

from src.sentiment.composite_index import (
    VALID_COMPOSITE_NLP_LABELS,
    build_composite_nlp_risk_index,
)
from src.sentiment.coverage import calculate_nlp_coverage
from src.sentiment.nlp_regime_comparison import compare_composite_nlp_to_regimes
from src.sentiment.scoring import score_sentiment_records


def _gdelt_records() -> pd.DataFrame:
    dates = pd.bdate_range("2026-04-01", "2026-06-21")[::2][:25]
    rows: list[dict[str, object]] = []
    for idx, day in enumerate(dates):
        for copy_id in range(2):
            rows.append(
                {
                    "record_id": f"gdelt-{idx}-{copy_id}",
                    "provider": "gdelt",
                    "document_type": "news",
                    "source": "economictimes.indiatimes.com",
                    "title": "Inflation shock and rate hike concerns hit India outlook",
                    "text": "",
                    "url": f"https://economictimes.indiatimes.com/news/{idx}-{copy_id}",
                    "language": "en",
                    "publication_time": day.tz_localize("UTC").isoformat(),
                    "decision_available_date": (day + pd.Timedelta(days=1))
                    .tz_localize("UTC")
                    .isoformat(),
                    "retrieval_time": "2026-06-23T10:00:00Z",
                    "is_real_provider_data": True,
                    "is_ex_ante_valid": True,
                    "possible_reaction_data": False,
                    "source_quality_label": "high",
                }
            )
    return pd.DataFrame(rows)


def test_news_only_records_have_positive_decision_label_coverage() -> None:
    market_index = pd.bdate_range("2026-04-01", "2026-06-21")
    records = _gdelt_records()
    scored = score_sentiment_records(records)

    composite = build_composite_nlp_risk_index(
        news_sentiment=scored,
        market_index=market_index,
        decision_lag=1,
    )
    coverage = calculate_nlp_coverage(
        records,
        composite_index=composite,
        start_date="2026-04-01",
        end_date="2026-06-21",
        min_records=50,
        min_distinct_dates=20,
        min_coverage_ratio=0.20,
    )

    assert coverage["record_count"] == 50
    assert coverage["distinct_publication_dates"] == 25
    assert coverage["decision_label_coverage"] > 0
    assert coverage["coverage_quality"] == "limited"
    assert coverage["source_families"] == ["news"]
    assert coverage["source_diversity_limited"] is True
    assert "news_only" in set(composite["decision_source_mix"].dropna())


def test_regime_comparison_only_outputs_valid_decision_label_dates() -> None:
    market_index = pd.bdate_range("2026-04-01", "2026-06-21")
    scored = score_sentiment_records(_gdelt_records())
    composite = build_composite_nlp_risk_index(
        news_sentiment=scored,
        market_index=market_index,
        decision_lag=1,
    )
    regimes = pd.Series("Stress", index=market_index)

    comparison = compare_composite_nlp_to_regimes(
        composite,
        regimes,
        regimes,
    )
    table = comparison["comparison_table"]

    assert not table.empty
    assert set(table["composite_nlp_label"]).issubset(set(VALID_COMPOSITE_NLP_LABELS))
    assert len(table) == int(
        composite["decision_composite_nlp_label"].isin(VALID_COMPOSITE_NLP_LABELS).sum()
    )
    assert comparison["predictiveness_claim"] is False
