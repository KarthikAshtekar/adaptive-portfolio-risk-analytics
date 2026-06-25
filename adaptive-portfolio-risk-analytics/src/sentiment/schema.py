"""Data contracts for the Phase 4A sentiment confirmation layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


SENTIMENT_LABELS = ("risk_on", "neutral", "risk_off", "unknown")
RBI_DOCUMENT_TYPES = (
    "mpc_minutes",
    "monetary_policy_statement",
    "governor_speech",
    "press_release",
    "financial_stability_report",
    "annual_report",
    "unknown",
)
RBI_STANCE_LABELS = ("hawkish", "neutral", "dovish")
RBI_CERTAINTY_LABELS = ("certain", "uncertain", "neutral")
RBI_TIME_LABELS = (
    "forward_looking",
    "backward_looking",
    "current",
    "unknown",
)


@dataclass(frozen=True)
class SentimentRecord:
    """One timestamped market-news record and its optional score metadata."""

    timestamp: pd.Timestamp
    source: str
    title: str
    text: str
    ticker: str | None = None
    market: str = "IN"
    sentiment_score: float | None = None
    sentiment_label: str = "unknown"
    model_name: str = "unscored"
    model_version: str = "1.0"
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        timestamp = pd.Timestamp(self.timestamp)
        if pd.isna(timestamp):
            raise ValueError("timestamp must be valid")
        if not str(self.source).strip():
            raise ValueError("source must not be empty")
        if not str(self.title).strip() and not str(self.text).strip():
            raise ValueError("title or text must not be empty")
        if self.sentiment_label not in SENTIMENT_LABELS:
            raise ValueError(
                f"sentiment_label must be one of {SENTIMENT_LABELS}"
            )
        object.__setattr__(self, "timestamp", timestamp)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable record dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class RBIDocument:
    """One locally stored RBI document plus manifest and load metadata."""

    document_id: str
    publication_date: pd.Timestamp
    document_type: str
    title: str
    text: str
    source_url: str | None = None
    local_path: str | None = None
    language: str = "en"
    source: str = "RBI"
    load_status: str = "loaded"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        publication_date = pd.Timestamp(self.publication_date)
        if pd.isna(publication_date):
            raise ValueError("publication_date must be valid")
        if not str(self.document_id).strip():
            raise ValueError("document_id must not be empty")
        if self.document_type not in RBI_DOCUMENT_TYPES:
            raise ValueError(
                f"document_type must be one of {RBI_DOCUMENT_TYPES}"
            )
        if not str(self.language).strip():
            raise ValueError("language must not be empty")
        object.__setattr__(self, "publication_date", publication_date)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable document dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class RBISentenceScore:
    """One sentence-level RBI stance, certainty, and time classification."""

    document_id: str
    publication_date: pd.Timestamp
    sentence_id: str
    sentence: str
    stance_label: str
    stance_score: float
    certainty_label: str
    certainty_score: float
    time_label: str
    time_score: float
    model_name: str
    model_version: str
    sentence_order: int = 0
    hawkish_score: float = 0.0
    dovish_score: float = 0.0
    uncertainty_score: float = 0.0
    forward_looking_score: float = 0.0
    scoring_method: str = "lexicon"
    fallback_used: bool = False
    fallback_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stance_label not in RBI_STANCE_LABELS:
            raise ValueError(f"stance_label must be one of {RBI_STANCE_LABELS}")
        if self.certainty_label not in RBI_CERTAINTY_LABELS:
            raise ValueError(
                f"certainty_label must be one of {RBI_CERTAINTY_LABELS}"
            )
        if self.time_label not in RBI_TIME_LABELS:
            raise ValueError(f"time_label must be one of {RBI_TIME_LABELS}")
        if int(self.sentence_order) < 0:
            raise ValueError("sentence_order must be non-negative")
        object.__setattr__(self, "publication_date", pd.Timestamp(self.publication_date))

    @property
    def sentence_text(self) -> str:
        """Compatibility alias used by the DataFrame scoring pipeline."""
        return self.sentence

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable sentence-score dictionary."""
        return asdict(self)
