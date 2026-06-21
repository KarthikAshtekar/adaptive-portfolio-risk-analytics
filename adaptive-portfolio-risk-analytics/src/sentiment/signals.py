"""Daily observed and lagged sentiment signal construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .alignment import assign_records_to_market_dates, validate_market_index


def _classify_score(score: float, threshold: float) -> str:
    if not np.isfinite(score):
        return "unknown"
    if score > threshold:
        return "risk_on"
    if score < -threshold:
        return "risk_off"
    return "neutral"


def build_daily_sentiment_signal(
    scored_records: pd.DataFrame,
    market_index: pd.DatetimeIndex,
    aggregation: str = "mean",
    lookback_window: int = 5,
    decision_lag: int = 1,
    *,
    neutral_threshold: float = 0.15,
) -> pd.DataFrame:
    """Build market-aligned observed and lagged sentiment signals.

    Records are assigned to the first market date on or after their calendar
    date. The observed rolling label is then shifted by ``decision_lag`` market
    sessions, so a decision on day ``t`` cannot use day-``t`` records.
    """
    index = validate_market_index(market_index)
    if aggregation not in {"mean", "median"}:
        raise ValueError("aggregation must be 'mean' or 'median'")
    if int(lookback_window) <= 0:
        raise ValueError("lookback_window must be positive")
    if int(decision_lag) < 1:
        raise ValueError("decision_lag must be at least 1 for Phase 4A")
    if "sentiment_score" not in scored_records:
        raise ValueError("scored_records must contain sentiment_score")

    assigned = assign_records_to_market_dates(scored_records, index)
    assigned["sentiment_score"] = pd.to_numeric(
        assigned["sentiment_score"],
        errors="coerce",
    )
    assigned = assigned.loc[assigned["sentiment_score"].notna()].copy()

    signal = pd.DataFrame(index=index)
    signal.index.name = "date"
    if assigned.empty:
        daily_scores = pd.Series(dtype=float)
        counts = pd.Series(dtype=int)
        latest_timestamps = pd.Series(dtype="datetime64[ns]")
    else:
        grouped = assigned.groupby("market_date", sort=True)
        daily_scores = getattr(grouped["sentiment_score"], aggregation)()
        counts = grouped.size()
        latest_timestamps = grouped["timestamp"].max()

    signal["daily_sentiment_score"] = daily_scores.reindex(index)
    signal["article_count"] = counts.reindex(index, fill_value=0).astype(int)
    signal["latest_article_timestamp"] = latest_timestamps.reindex(index)
    signal["rolling_sentiment_score"] = (
        signal["daily_sentiment_score"]
        .rolling(int(lookback_window), min_periods=1)
        .mean()
    )
    signal["rolling_article_count"] = (
        signal["article_count"]
        .rolling(int(lookback_window), min_periods=1)
        .sum()
        .astype(int)
    )
    signal["observed_sentiment_label"] = signal["rolling_sentiment_score"].map(
        lambda score: _classify_score(score, neutral_threshold)
    )
    signal["decision_sentiment_score"] = signal["rolling_sentiment_score"].shift(
        int(decision_lag)
    )
    signal["decision_sentiment_label"] = (
        signal["observed_sentiment_label"]
        .shift(int(decision_lag))
        .fillna("unknown")
        .astype("object")
    )
    signal["decision_article_count"] = (
        signal["rolling_article_count"].shift(int(decision_lag)).fillna(0).astype(int)
    )
    signal["decision_source_timestamp"] = signal["latest_article_timestamp"].shift(
        int(decision_lag)
    )
    signal["decision_lag"] = int(decision_lag)
    return signal

