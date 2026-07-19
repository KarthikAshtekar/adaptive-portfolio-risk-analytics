"""Common contracts for optional external sentiment providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import pandas as pd


NORMALIZED_SENTIMENT_COLUMNS = (
    "record_id",
    "timestamp",
    "publication_time",
    "retrieval_time",
    "source",
    "provider",
    "document_type",
    "entity",
    "ticker",
    "sector",
    "country",
    "title",
    "text",
    "url",
    "language",
    "raw_metadata",
)


def stable_record_id(*values: object) -> str:
    """Build a deterministic identifier without exposing provider payloads."""
    joined = "|".join(str(value or "").strip() for value in values)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]


def serialize_metadata(value: object) -> str:
    """Serialize provider metadata consistently for CSV-safe persistence."""
    if isinstance(value, str):
        try:
            json.loads(value)
            return value
        except json.JSONDecodeError:
            return json.dumps({"value": value}, ensure_ascii=False)
    return json.dumps(value or {}, default=str, ensure_ascii=False, sort_keys=True)


def _missing_text(value: object) -> bool:
    if value is None or pd.isna(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none", "<na>"}


def normalized_frame(
    records: list[dict[str, Any]] | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a schema-stable normalized provider frame."""
    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records or [])
    for column in NORMALIZED_SENTIMENT_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    frame = frame.loc[
        :,
        [
            *NORMALIZED_SENTIMENT_COLUMNS,
            *[column for column in frame.columns if column not in NORMALIZED_SENTIMENT_COLUMNS],
        ],
    ].copy()
    for column in ("timestamp", "publication_time", "retrieval_time"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    frame["raw_metadata"] = frame["raw_metadata"].map(serialize_metadata)
    return frame


@dataclass(frozen=True)
class ProviderValidation:
    """Row-level provider validation without aborting a mixed batch."""

    records: pd.DataFrame
    valid_records: pd.DataFrame
    invalid_records: pd.DataFrame
    diagnostics: dict[str, object]


class SentimentProvider(ABC):
    """Base interface for API, feed, and local sentiment providers."""

    provider_name = "unknown"

    def __init__(self) -> None:
        self.last_diagnostics: dict[str, object] = {}

    @abstractmethod
    def fetch(
        self,
        start_date,
        end_date,
        query=None,
        symbols=None,
        sectors=None,
        limit=None,
    ):
        """Fetch provider-native records. Live access must remain optional."""

    @abstractmethod
    def normalize(self, raw_records) -> pd.DataFrame:
        """Normalize provider-native records into the shared schema."""

    def validate(self, records: pd.DataFrame) -> ProviderValidation:
        """Validate provenance and timestamps while retaining invalid rows."""
        frame = normalized_frame(records)
        errors: list[str] = []
        required_text = (
            "record_id",
            "source",
            "provider",
            "document_type",
            "title",
            "url",
        )
        for row in frame.itertuples(index=False):
            row_errors: list[str] = []
            for column in required_text:
                if _missing_text(getattr(row, column, "")):
                    row_errors.append(f"missing {column}")
            if pd.isna(row.publication_time):
                row_errors.append("missing publication_time")
            if pd.isna(row.retrieval_time):
                row_errors.append("missing retrieval_time")
            if (
                pd.notna(row.publication_time)
                and pd.notna(row.retrieval_time)
                and row.publication_time > row.retrieval_time
            ):
                row_errors.append("publication_time after retrieval_time")
            if _missing_text(row.text) and _missing_text(row.title):
                row_errors.append("missing title and text")
            errors.append("; ".join(row_errors))
        frame["provider_validation_errors"] = errors
        frame["provider_record_valid"] = frame["provider_validation_errors"].eq("")
        valid = frame.loc[frame["provider_record_valid"]].copy()
        invalid = frame.loc[~frame["provider_record_valid"]].copy()
        diagnostics = {
            "provider": self.provider_name,
            "normalized_record_count": int(len(frame)),
            "valid_record_count": int(len(valid)),
            "invalid_record_count": int(len(invalid)),
        }
        return ProviderValidation(
            records=frame,
            valid_records=valid,
            invalid_records=invalid,
            diagnostics=diagnostics,
        )
