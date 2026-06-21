"""Local-file ingestion for timestamped sentiment records."""

from __future__ import annotations

from pathlib import Path
from typing import IO

import pandas as pd


REQUIRED_INPUT_COLUMNS = ("timestamp", "source", "title", "text")
OPTIONAL_INPUT_COLUMNS = ("ticker", "url")
STANDARD_INPUT_COLUMNS = REQUIRED_INPUT_COLUMNS + OPTIONAL_INPUT_COLUMNS


def load_local_sentiment_csv(
    source: str | Path | IO[bytes] | IO[str],
) -> pd.DataFrame:
    """Load, validate, deduplicate, and sort a local sentiment CSV.

    Timestamps are converted to UTC and then made timezone-naive so they align
    consistently with the repository's daily market indexes.
    """
    if hasattr(source, "seek"):
        source.seek(0)
    frame = pd.read_csv(source)
    frame.columns = [str(column).strip().lower() for column in frame.columns]

    for column in STANDARD_INPUT_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA

    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame["timestamp"] = timestamps.dt.tz_convert(None)
    for column in ("source", "title", "text", "ticker", "url"):
        frame[column] = frame[column].astype("string").str.strip()

    frame["source"] = frame["source"].fillna("unknown_source")
    frame["title"] = frame["title"].fillna("")
    frame["text"] = frame["text"].fillna("")
    frame["text"] = frame["text"].where(frame["text"].ne(""), frame["title"])
    frame["title"] = frame["title"].where(frame["title"].ne(""), frame["text"])

    valid = (
        frame["timestamp"].notna()
        & frame["source"].ne("")
        & (frame["title"].ne("") | frame["text"].ne(""))
    )
    clean = frame.loc[valid, list(STANDARD_INPUT_COLUMNS)].copy()
    clean = clean.drop_duplicates(
        subset=["title", "source", "timestamp"],
        keep="first",
    )
    clean = clean.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    clean["market"] = "IN"
    clean["sentiment_score"] = float("nan")
    clean["sentiment_label"] = "unknown"
    clean["model_name"] = "unscored"
    clean["model_version"] = "1.0"
    clean["metadata"] = [{} for _ in range(len(clean))]
    return clean
