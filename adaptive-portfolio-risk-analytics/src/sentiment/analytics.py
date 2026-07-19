"""Comparison analytics between sentiment and quantitative regimes."""

from __future__ import annotations

import numpy as np
import pandas as pd


RISK_ON_QUANT_REGIMES = {"calm", "normal", "risk-on", "risk_on"}
NEUTRAL_QUANT_REGIMES = {"normal"}
RISK_OFF_QUANT_REGIMES = {"stress", "crisis", "risk-off", "risk_off"}


def _coerce_regimes(values: pd.Series | None, name: str) -> pd.Series:
    if values is None:
        return pd.Series(dtype="object", name=name)
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    if not isinstance(values.index, pd.DatetimeIndex):
        raise ValueError(f"{name} index must be a DatetimeIndex")
    result = values.sort_index().astype("object")
    result.name = name
    return result


def sentiment_agrees_with_regime(
    sentiment_label: object,
    quantitative_regime: object,
) -> bool | float:
    """Return whether one sentiment label confirms one quantitative regime."""
    sentiment = str(sentiment_label or "unknown").strip().lower()
    regime = str(quantitative_regime or "unknown").strip().lower()
    if sentiment == "unknown" or regime in {"", "unknown", "nan", "none"}:
        return np.nan
    if sentiment == "risk_on":
        return regime in RISK_ON_QUANT_REGIMES
    if sentiment == "neutral":
        return regime in NEUTRAL_QUANT_REGIMES
    if sentiment == "risk_off":
        return regime in RISK_OFF_QUANT_REGIMES
    return False


def _agreement_rate(values: pd.Series) -> float:
    valid = values.dropna()
    return float(valid.astype(bool).mean()) if not valid.empty else np.nan


def _lead_lag_table(
    sentiment_labels: pd.Series,
    rule_regimes: pd.Series,
    hmm_regimes: pd.Series,
    max_shift: int = 5,
) -> pd.DataFrame:
    rows = []
    for shift in range(-int(max_shift), int(max_shift) + 1):
        shifted = sentiment_labels.shift(shift)
        rule_agreement = pd.concat(
            [shifted.rename("sentiment"), rule_regimes.rename("quant")],
            axis=1,
            join="inner",
        ).apply(
            lambda row: sentiment_agrees_with_regime(row["sentiment"], row["quant"]),
            axis=1,
        )
        hmm_agreement = pd.concat(
            [shifted.rename("sentiment"), hmm_regimes.rename("quant")],
            axis=1,
            join="inner",
        ).apply(
            lambda row: sentiment_agrees_with_regime(row["sentiment"], row["quant"]),
            axis=1,
        )
        rows.append(
            {
                "sentiment_shift_market_days": shift,
                "rule_based_agreement_rate": _agreement_rate(rule_agreement),
                "hmm_agreement_rate": _agreement_rate(hmm_agreement),
            }
        )
    return pd.DataFrame(rows)


def compare_sentiment_to_regimes(
    sentiment_signal: pd.DataFrame,
    rule_based_regimes: pd.Series,
    hmm_regimes: pd.Series | None,
) -> dict[str, object]:
    """Compare lagged sentiment decisions with trading-safe quantitative regimes."""
    if "decision_sentiment_label" not in sentiment_signal:
        raise ValueError("sentiment_signal must contain decision_sentiment_label")
    sentiment = sentiment_signal["decision_sentiment_label"].astype("object")
    rule = _coerce_regimes(rule_based_regimes, "rule_based_regime").reindex(sentiment_signal.index)
    hmm = _coerce_regimes(hmm_regimes, "hmm_regime").reindex(sentiment_signal.index)

    comparison = pd.DataFrame(index=sentiment_signal.index)
    comparison.index.name = "date"
    comparison["sentiment_label"] = sentiment
    comparison["rule_based_regime"] = rule
    comparison["hmm_regime"] = hmm
    comparison["article_count"] = sentiment_signal.get("article_count", 0)
    comparison["decision_article_count"] = sentiment_signal.get(
        "decision_article_count",
        0,
    )
    comparison["agreement_rule_based"] = comparison.apply(
        lambda row: sentiment_agrees_with_regime(
            row["sentiment_label"],
            row["rule_based_regime"],
        ),
        axis=1,
    )
    comparison["agreement_hmm"] = comparison.apply(
        lambda row: sentiment_agrees_with_regime(
            row["sentiment_label"],
            row["hmm_regime"],
        ),
        axis=1,
    )

    stress_mask = comparison["rule_based_regime"].astype(str).str.lower().isin({"stress", "crisis"})
    stress_with_sentiment = stress_mask & comparison["sentiment_label"].astype(str).ne("unknown")
    risk_off_rule_agreement = (
        float(
            comparison.loc[stress_with_sentiment, "sentiment_label"]
            .astype(str)
            .str.lower()
            .eq("risk_off")
            .mean()
        )
        if stress_with_sentiment.any()
        else np.nan
    )
    hmm_stress_mask = (
        comparison["hmm_regime"]
        .astype(str)
        .str.lower()
        .isin({"stress", "crisis", "risk-off", "risk_off"})
    )
    hmm_stress_with_sentiment = hmm_stress_mask & comparison["sentiment_label"].astype(str).ne(
        "unknown"
    )
    risk_off_hmm_agreement = (
        float(
            comparison.loc[hmm_stress_with_sentiment, "sentiment_label"]
            .astype(str)
            .str.lower()
            .eq("risk_off")
            .mean()
        )
        if hmm_stress_with_sentiment.any()
        else np.nan
    )

    disagreement_mask = comparison[["agreement_rule_based", "agreement_hmm"]].eq(False).any(axis=1)
    disagreement = comparison.loc[disagreement_mask].copy()
    disagreement.insert(0, "date", disagreement.index)

    valid_sentiment = comparison["sentiment_label"].astype(str).ne("unknown")
    distribution = (
        comparison.loc[valid_sentiment, "sentiment_label"]
        .value_counts()
        .rename_axis("sentiment_label")
        .reset_index(name="number_of_days")
    )
    if not distribution.empty:
        distribution["percentage_of_days"] = (
            distribution["number_of_days"] / distribution["number_of_days"].sum()
        )

    article_count = pd.to_numeric(
        sentiment_signal.get(
            "article_count",
            pd.Series(0, index=sentiment_signal.index),
        ),
        errors="coerce",
    ).fillna(0)
    coverage_ratio = float(article_count.gt(0).mean()) if len(article_count) else 0.0
    decision_coverage_ratio = float(valid_sentiment.mean()) if len(valid_sentiment) else 0.0

    return {
        "agreement_with_rule_based": _agreement_rate(comparison["agreement_rule_based"]),
        "agreement_with_hmm": _agreement_rate(comparison["agreement_hmm"]),
        "risk_off_agreement_rule_based": risk_off_rule_agreement,
        "risk_off_agreement_hmm": risk_off_hmm_agreement,
        "lead_lag_diagnostics": _lead_lag_table(sentiment, rule, hmm),
        "dates_of_major_disagreement": disagreement.reset_index(drop=True),
        "confusion_matrix_rule_based": pd.crosstab(
            comparison["sentiment_label"],
            comparison["rule_based_regime"],
        ),
        "confusion_matrix_hmm": pd.crosstab(
            comparison["sentiment_label"],
            comparison["hmm_regime"],
        ),
        "sentiment_distribution": distribution,
        "article_coverage_ratio": coverage_ratio,
        "decision_coverage_ratio": decision_coverage_ratio,
        "comparison_table": comparison,
    }


def calculate_sentiment_confirmation_score(
    quantitative_regime: object,
    sentiment_label: object,
    *,
    article_count: int | float | None = None,
) -> str:
    """Classify current sentiment as confirmation, disagreement, or insufficient."""
    regime = str(quantitative_regime or "Unknown").strip().lower()
    sentiment = str(sentiment_label or "unknown").strip().lower()
    if (
        sentiment == "unknown"
        or regime in {"", "unknown", "nan", "none"}
        or (article_count is not None and float(article_count) <= 0)
    ):
        return "Insufficient Sentiment Data"
    if sentiment == "risk_off" and regime in RISK_OFF_QUANT_REGIMES:
        return "Confirmed Risk-Off"
    if sentiment == "risk_on" and regime in RISK_ON_QUANT_REGIMES:
        return "Confirmed Risk-On"
    if sentiment == "neutral" and regime in NEUTRAL_QUANT_REGIMES:
        return "Confirmed Neutral"
    return "Quant-Sentiment Disagreement"
