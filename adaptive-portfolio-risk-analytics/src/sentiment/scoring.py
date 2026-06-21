"""Dependency-light lexicon scoring for market-news records."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


RISK_OFF_TERMS = (
    "crash",
    "selloff",
    "recession",
    "inflation shock",
    "rate hike",
    "default",
    "downgrade",
    "war",
    "liquidity crisis",
    "volatility spike",
    "drawdown",
    "panic",
    "weak earnings",
    "banking stress",
)

RISK_ON_TERMS = (
    "rally",
    "growth",
    "rate cut",
    "easing",
    "strong earnings",
    "upgrade",
    "recovery",
    "bullish",
    "stabilization",
    "soft landing",
    "liquidity support",
    "record high",
    "momentum",
)

LEXICON_MODEL_NAME = "phase4a_lexicon"
LEXICON_MODEL_VERSION = "1.0"


def _normalize_text(value: object) -> str:
    text = str(value or "").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s-]", " ", text)).strip()


def _term_count(text: str, terms: tuple[str, ...]) -> int:
    return sum(text.count(term) for term in terms)


def _label_score(score: float, neutral_threshold: float) -> str:
    if not np.isfinite(score):
        return "unknown"
    if score > neutral_threshold:
        return "risk_on"
    if score < -neutral_threshold:
        return "risk_off"
    return "neutral"


def score_sentiment_records(
    records_df: pd.DataFrame,
    method: str = "lexicon",
    *,
    neutral_threshold: float = 0.15,
) -> pd.DataFrame:
    """Score records using the Phase 4A lexicon convention.

    Positive scores are risk-on, negative scores are risk-off, and near-zero
    scores are neutral. VADER and FinBERT remain future extension points.
    """
    if method != "lexicon":
        raise ValueError("Phase 4A supports method='lexicon' only")
    if not isinstance(records_df, pd.DataFrame):
        raise TypeError("records_df must be a pandas DataFrame")
    if neutral_threshold < 0:
        raise ValueError("neutral_threshold must be non-negative")

    scored = records_df.copy()
    for column in ("title", "text"):
        if column not in scored:
            scored[column] = ""

    scores: list[float] = []
    for row in scored.itertuples(index=False):
        title = _normalize_text(getattr(row, "title", ""))
        text = _normalize_text(getattr(row, "text", ""))
        combined = f"{title} {text}".strip()
        positive = _term_count(combined, RISK_ON_TERMS)
        negative = _term_count(combined, RISK_OFF_TERMS)
        total = positive + negative
        score = 0.0 if total == 0 else (positive - negative) / total
        scores.append(float(np.clip(score, -1.0, 1.0)))

    scored["sentiment_score"] = scores
    scored["sentiment_label"] = [
        _label_score(score, neutral_threshold) for score in scores
    ]
    scored["model_name"] = LEXICON_MODEL_NAME
    scored["model_version"] = LEXICON_MODEL_VERSION
    return scored

