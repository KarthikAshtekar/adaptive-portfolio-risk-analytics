"""Timestamp and reaction-data checks for ex-ante textual records."""

from __future__ import annotations

import re

import pandas as pd


REACTION_DATA_PATTERNS = {
    "market fell": r"\bmarket\s+fell\b",
    "stock plunged after": r"\bstock\s+plunged\s+after\b",
    "shares rallied after": r"\bshares\s+rallied\s+after\b",
    "Nifty dropped": r"\bnifty\s+dropped\b",
    "Sensex crashed": r"\bsensex\s+crashed\b",
    "volatility spiked": r"\bvolatility\s+spiked\b",
}


def validate_ex_ante_records(records_df: pd.DataFrame) -> pd.DataFrame:
    """Mark records valid only when publication/retrieval provenance is usable."""
    if not isinstance(records_df, pd.DataFrame):
        raise TypeError("records_df must be a pandas DataFrame")
    frame = records_df.copy()
    for column in ("publication_time", "retrieval_time"):
        if column not in frame:
            frame[column] = pd.NaT
        frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    reasons: list[str] = []
    for row in frame.itertuples(index=False):
        row_reasons: list[str] = []
        if pd.isna(row.publication_time):
            row_reasons.append("missing publication_time")
        if pd.isna(row.retrieval_time):
            row_reasons.append("missing retrieval_time")
        if (
            pd.notna(row.publication_time)
            and pd.notna(row.retrieval_time)
            and row.publication_time > row.retrieval_time
        ):
            row_reasons.append("publication_time after retrieval_time")
        reasons.append("; ".join(row_reasons))
    frame["ex_ante_validation_errors"] = reasons
    frame["is_ex_ante_valid"] = frame["ex_ante_validation_errors"].eq("")
    return frame


def flag_reaction_data_leakage(records_df: pd.DataFrame) -> pd.DataFrame:
    """Flag possible post-event market-reaction language without deleting it."""
    if not isinstance(records_df, pd.DataFrame):
        raise TypeError("records_df must be a pandas DataFrame")
    frame = records_df.copy()
    for column in ("title", "text"):
        if column not in frame:
            frame[column] = ""
    warnings: list[str] = []
    for row in frame.itertuples(index=False):
        combined = f"{getattr(row, 'title', '')} {getattr(row, 'text', '')}"
        matched = [
            label
            for label, pattern in REACTION_DATA_PATTERNS.items()
            if re.search(pattern, combined, flags=re.IGNORECASE)
        ]
        warnings.append("; ".join(matched))
    frame["reaction_warning_reason"] = warnings
    frame["possible_reaction_data"] = frame[
        "reaction_warning_reason"
    ].ne("")
    return frame


def apply_publication_lag(
    records_df: pd.DataFrame,
    lag_days: int = 1,
) -> pd.DataFrame:
    """Attach the earliest decision date after the configured publication lag."""
    if int(lag_days) < 1:
        raise ValueError("lag_days must be at least 1")
    frame = (
        records_df.copy()
        if "is_ex_ante_valid" in records_df
        else validate_ex_ante_records(records_df)
    )
    publication = pd.to_datetime(
        frame["publication_time"], errors="coerce", utc=True
    )
    frame["decision_available_date"] = (
        publication.dt.normalize() + pd.Timedelta(days=int(lag_days))
    )
    frame.loc[
        ~frame["is_ex_ante_valid"], "decision_available_date"
    ] = pd.NaT
    frame["publication_lag_days"] = int(lag_days)
    return frame
