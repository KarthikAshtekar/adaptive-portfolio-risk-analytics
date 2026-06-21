"""Timestamp and market-index alignment helpers for sentiment records."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_market_index(market_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return a sorted, unique, timezone-naive market index."""
    if not isinstance(market_index, pd.DatetimeIndex):
        raise TypeError("market_index must be a DatetimeIndex")
    if market_index.empty:
        raise ValueError("market_index must not be empty")
    index = market_index
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    return pd.DatetimeIndex(index).sort_values().drop_duplicates()


def assign_records_to_market_dates(
    records_df: pd.DataFrame,
    market_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Assign each record to the first market date on or after its calendar date."""
    index = validate_market_index(market_index)
    if "timestamp" not in records_df:
        raise ValueError("records_df must contain timestamp")

    records = records_df.copy()
    timestamps = pd.to_datetime(records["timestamp"], errors="coerce", utc=True)
    records["timestamp"] = timestamps.dt.tz_convert(None)
    records = records.loc[records["timestamp"].notna()].copy()
    normalized = records["timestamp"].dt.normalize().to_numpy(dtype="datetime64[ns]")
    positions = index.values.searchsorted(normalized, side="left")
    valid = positions < len(index)
    records = records.loc[valid].copy()
    records["market_date"] = index.take(positions[valid])
    return records.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def build_alignment_checks(
    scored_records: pd.DataFrame,
    sentiment_signal: pd.DataFrame,
    market_index: pd.DatetimeIndex,
    *,
    decision_lag: int,
) -> dict[str, object]:
    """Audit timestamp validity, index alignment, lagging, and look-ahead safety."""
    index = validate_market_index(market_index)
    assigned = assign_records_to_market_dates(scored_records, index)
    invalid_timestamp_count = int(
        pd.to_datetime(
            scored_records.get("timestamp", pd.Series(dtype="object")),
            errors="coerce",
        ).isna().sum()
    )
    aligned_index = sentiment_signal.index.equals(index)
    expected_labels = (
        sentiment_signal["observed_sentiment_label"]
        .shift(int(decision_lag))
        .fillna("unknown")
        .astype(str)
    )
    lag_matches = expected_labels.equals(
        sentiment_signal["decision_sentiment_label"].astype(str)
    )

    decision_timestamps = pd.to_datetime(
        sentiment_signal.get(
            "decision_source_timestamp",
            pd.Series(pd.NaT, index=index),
        ),
        errors="coerce",
    )
    decision_dates = pd.Series(index, index=index)
    no_future_timestamp = bool(
        (
            decision_timestamps.isna()
            | (decision_timestamps.dt.normalize() < decision_dates)
        ).all()
    )
    records_before_end = bool(
        assigned.empty
        or assigned["market_date"].le(index.max()).all()
    )
    checks = pd.DataFrame(
        [
            {"check": "valid_input_timestamps", "passed": invalid_timestamp_count == 0},
            {"check": "signal_index_matches_market_index", "passed": aligned_index},
            {"check": "decision_labels_match_configured_lag", "passed": lag_matches},
            {"check": "decision_sources_precede_decision_date", "passed": no_future_timestamp},
            {"check": "records_do_not_exceed_market_index", "passed": records_before_end},
        ]
    )
    return {
        "checks": checks,
        "all_checks_passed": bool(checks["passed"].all()),
        "assigned_records": assigned,
        "invalid_timestamp_count": invalid_timestamp_count,
        "unassigned_record_count": int(max(len(scored_records) - len(assigned), 0)),
        "decision_lag": int(decision_lag),
    }
