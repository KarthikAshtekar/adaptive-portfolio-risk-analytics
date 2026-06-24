"""Optional local FinBERT adapter with deterministic lexicon fallback."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .scoring import (
    LEXICON_MODEL_NAME,
    LEXICON_MODEL_VERSION,
    add_risk_scores_from_sentiment,
    score_sentiment_records,
)


def _fallback_labels(labels: pd.Series) -> pd.Series:
    return labels.map(
        {
            "risk_on": "positive",
            "risk_off": "negative",
            "neutral": "neutral",
            "unknown": "unknown",
        }
    ).fillna("unknown")


def score_with_finbert(
    records_df: pd.DataFrame,
    model_name: str = "ProsusAI/finbert",
    fallback: str = "lexicon",
    *,
    pipeline_factory: Callable[..., object] | None = None,
    local_files_only: bool = True,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Score text locally with FinBERT, falling back without network access."""
    if fallback != "lexicon":
        raise ValueError("fallback must be 'lexicon'")
    if not isinstance(records_df, pd.DataFrame):
        raise TypeError("records_df must be a pandas DataFrame")
    scored = score_sentiment_records(records_df, method="lexicon")
    scored["finbert_label"] = _fallback_labels(scored["sentiment_label"])
    scored["finbert_score"] = scored["sentiment_score"].abs()
    scored["scoring_method_used"] = "lexicon_fallback"
    scored["model_name"] = LEXICON_MODEL_NAME
    scored["model_version"] = LEXICON_MODEL_VERSION
    scored["fallback_used"] = True
    scored["fallback_reason"] = "FinBERT not attempted"
    if scored.empty:
        return scored

    try:
        factory = pipeline_factory
        if factory is None:
            from transformers import pipeline

            factory = pipeline
        classifier = factory(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            model_kwargs={"local_files_only": local_files_only},
        )
        texts = (
            scored["title"].fillna("").astype(str)
            + ". "
            + scored["text"].fillna("").astype(str)
        ).str.strip().tolist()
        outputs = classifier(
            texts,
            truncation=True,
            batch_size=int(batch_size),
        )
        if len(outputs) != len(scored):
            raise ValueError("FinBERT output length does not match input")
        labels: list[str] = []
        confidences: list[float] = []
        sentiment_scores: list[float] = []
        sentiment_labels: list[str] = []
        for output in outputs:
            item = output[0] if isinstance(output, list) else output
            label = str(item.get("label", "neutral")).strip().lower()
            aliases = {
                "label_0": "positive",
                "label_1": "negative",
                "label_2": "neutral",
            }
            label = aliases.get(label, label)
            if label not in {"positive", "negative", "neutral"}:
                label = "neutral"
            confidence = float(item.get("score", 0.0))
            signed_score = (
                confidence
                if label == "positive"
                else -confidence
                if label == "negative"
                else 0.0
            )
            labels.append(label)
            confidences.append(confidence)
            sentiment_scores.append(float(np.clip(signed_score, -1.0, 1.0)))
            sentiment_labels.append(
                "risk_on"
                if label == "positive"
                else "risk_off"
                if label == "negative"
                else "neutral"
            )
        scored["finbert_label"] = labels
        scored["finbert_score"] = confidences
        scored["sentiment_score"] = sentiment_scores
        scored["sentiment_label"] = sentiment_labels
        scored = add_risk_scores_from_sentiment(scored)
        scored["scoring_method_used"] = "finbert"
        scored["model_name"] = model_name
        scored["model_version"] = "local_huggingface"
        scored["fallback_used"] = False
        scored["fallback_reason"] = pd.NA
    except Exception as exc:
        scored["fallback_reason"] = str(exc)
    return scored
