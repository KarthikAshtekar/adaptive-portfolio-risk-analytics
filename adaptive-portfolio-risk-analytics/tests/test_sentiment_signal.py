"""Tests for market alignment and lagged daily sentiment decisions."""

from __future__ import annotations

import pandas as pd

from src.sentiment import build_alignment_checks, build_daily_sentiment_signal


def _scored_records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-05 18:00:00",
                    "2024-01-08 11:00:00",
                    "2024-01-09 11:00:00",
                ]
            ),
            "source": ["Sample"] * 3,
            "title": ["Friday", "Monday", "Tuesday"],
            "text": [""] * 3,
            "sentiment_score": [-1.0, 1.0, -1.0],
        }
    )


def test_daily_signal_aligns_and_applies_one_session_lag() -> None:
    market_index = pd.bdate_range("2024-01-05", periods=5)
    signal = build_daily_sentiment_signal(
        _scored_records(),
        market_index,
        lookback_window=1,
        decision_lag=1,
    )

    assert signal.index.equals(market_index)
    assert signal.loc["2024-01-08", "decision_sentiment_label"] == "risk_off"
    assert signal.loc["2024-01-09", "decision_sentiment_label"] == "risk_on"
    assert signal["decision_lag"].eq(1).all()


def test_decision_for_day_t_does_not_use_day_t_score() -> None:
    market_index = pd.bdate_range("2024-01-05", periods=5)
    baseline = _scored_records()
    altered = baseline.copy()
    altered.loc[
        altered["timestamp"].dt.date == pd.Timestamp("2024-01-09").date(), "sentiment_score"
    ] = 1.0

    baseline_signal = build_daily_sentiment_signal(
        baseline,
        market_index,
        lookback_window=1,
        decision_lag=1,
    )
    altered_signal = build_daily_sentiment_signal(
        altered,
        market_index,
        lookback_window=1,
        decision_lag=1,
    )

    assert (
        baseline_signal.loc["2024-01-09", "decision_sentiment_label"]
        == altered_signal.loc["2024-01-09", "decision_sentiment_label"]
    )
    checks = build_alignment_checks(
        baseline,
        baseline_signal,
        market_index,
        decision_lag=1,
    )
    assert checks["all_checks_passed"] is True
