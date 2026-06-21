"""Source-quality and real-versus-fixture provenance diagnostics."""

from __future__ import annotations

import json
from urllib.parse import urlparse

import numpy as np
import pandas as pd


KNOWN_FINANCIAL_DOMAINS = {
    "reuters.com",
    "bloomberg.com",
    "moneycontrol.com",
    "business-standard.com",
    "economictimes.indiatimes.com",
    "livemint.com",
    "cnbc.com",
    "ft.com",
    "wsj.com",
    "alpha-vantage.com",
    "alphavantage.co",
}
OFFICIAL_DOMAINS = {
    "rbi.org.in",
    "sebi.gov.in",
    "nseindia.com",
    "bseindia.com",
}
SUPPORTED_LANGUAGES = {"en", "eng", "english", "hi", "hin", "hindi"}
FIXTURE_MARKERS = {
    "do_not_use_placeholder",
    "synthetic fixture",
    "fixture for offline",
    "placeholder",
    "test only",
    "example.com",
    '"source_kind": "fixture"',
    '"source_kind": "local_fixture"',
}


def _text(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series("", index=frame.index, dtype="string")
    return frame[column].fillna("").astype(str).str.strip()


def _domain(url: object) -> str:
    value = str(url or "").strip()
    if not value:
        return ""
    host = urlparse(value).netloc.lower().removeprefix("www.")
    return host.split(":", 1)[0]


def _metadata_text(value: object) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str).lower()
    return str(value or "").lower()


def classify_data_provenance(records_df: pd.DataFrame) -> pd.DataFrame:
    """Mark obvious fixtures/placeholders so they cannot count as real evidence."""
    if not isinstance(records_df, pd.DataFrame):
        raise TypeError("records_df must be a pandas DataFrame")
    frame = records_df.copy()
    metadata = _text(frame, "raw_metadata").map(_metadata_text)
    urls = _text(frame, "url").str.lower()
    fixture = pd.Series(False, index=frame.index)
    reasons = pd.Series("", index=frame.index, dtype="string")
    for marker in FIXTURE_MARKERS:
        matched = metadata.str.contains(marker, regex=False) | urls.str.contains(
            marker, regex=False
        )
        fixture |= matched
        reasons = reasons.mask(
            matched & reasons.eq(""),
            f"fixture marker detected: {marker}",
        )
    frame["is_real_provider_data"] = ~fixture
    frame["data_provenance"] = np.where(
        fixture, "fixture_or_placeholder", "real_candidate"
    )
    frame["data_provenance_warning"] = reasons
    return frame


def score_source_quality(records_df: pd.DataFrame) -> pd.DataFrame:
    """Score auditable source properties without using market outcomes."""
    if not isinstance(records_df, pd.DataFrame):
        raise TypeError("records_df must be a pandas DataFrame")
    frame = classify_data_provenance(records_df)
    provider = _text(frame, "provider").str.lower()
    source = _text(frame, "source").str.lower()
    domains = _text(frame, "url").map(_domain)

    frame["official_source"] = (
        provider.eq("rbi")
        | source.str.contains("reserve bank of india", regex=False)
        | domains.isin(OFFICIAL_DOMAINS)
    )
    frame["known_financial_source"] = domains.isin(KNOWN_FINANCIAL_DOMAINS)
    publication = pd.to_datetime(
        frame.get(
            "publication_time",
            pd.Series(pd.NaT, index=frame.index),
        ),
        errors="coerce",
        utc=True,
    )
    frame["has_publication_time"] = publication.notna()
    frame["has_url"] = _text(frame, "url").ne("")
    entity_or_topic = pd.Series(False, index=frame.index)
    for column in ("entity", "ticker", "query", "sector"):
        entity_or_topic |= _text(frame, column).ne("")
    frame["has_entity_or_topic"] = entity_or_topic
    frame["language_supported"] = (
        _text(frame, "language").str.lower().isin(SUPPORTED_LANGUAGES)
    )

    duplicate_key = (
        _text(frame, "url").str.lower()
        + "|"
        + publication.astype(str)
        + "|"
        + _text(frame, "title").str.lower()
    )
    frame["duplicate_risk"] = duplicate_key.duplicated(keep=False)
    if "possible_reaction_data" not in frame:
        frame["possible_reaction_data"] = False
    frame["reaction_data_warning"] = (
        frame["possible_reaction_data"].fillna(False).astype(bool)
    )

    positive_dimensions = [
        "official_source",
        "known_financial_source",
        "has_publication_time",
        "has_url",
        "has_entity_or_topic",
        "language_supported",
    ]
    positive = frame[positive_dimensions].astype(float).sum(axis=1)
    safe_risks = (
        (~frame["duplicate_risk"]).astype(float)
        + (~frame["reaction_data_warning"]).astype(float)
    )
    frame["source_quality_score"] = (positive + safe_risks) / 8.0
    evidence_present = (
        _text(frame, "source").ne("")
        | _text(frame, "provider").ne("")
        | frame["has_url"]
    )
    frame["source_quality_label"] = np.select(
        [
            ~evidence_present,
            frame["source_quality_score"].ge(0.75),
            frame["source_quality_score"].ge(0.50),
        ],
        ["unknown", "high", "medium"],
        default="low",
    )
    warnings: list[str] = []
    for row in frame.itertuples(index=False):
        row_warnings: list[str] = []
        if row.duplicate_risk:
            row_warnings.append("possible duplicate")
        if row.reaction_data_warning:
            row_warnings.append("possible reaction data")
        if not row.has_publication_time:
            row_warnings.append("missing publication time")
        if not row.has_url:
            row_warnings.append("missing source URL")
        if row.data_provenance == "fixture_or_placeholder":
            row_warnings.append("fixture or placeholder data")
        warnings.append("; ".join(row_warnings))
    frame["source_quality_warning"] = warnings
    return frame
