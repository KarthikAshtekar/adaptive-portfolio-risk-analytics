"""Composite, decision-lagged NLP risk index for monitoring only."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd


COMPOSITE_NLP_LABELS = (
    "nlp_risk_on",
    "nlp_neutral",
    "nlp_risk_off",
    "insufficient_nlp_data",
)


def _market_dates(market_index) -> pd.DatetimeIndex:
    if not isinstance(market_index, pd.DatetimeIndex):
        raise TypeError(
            "market_index must be a DatetimeIndex; market returns are not inputs"
        )
    if market_index.empty:
        raise ValueError("market_index must not be empty")
    index = pd.DatetimeIndex(market_index).sort_values().drop_duplicates()
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    return index


def _align_text_scores(
    records: pd.DataFrame | None,
    index: pd.DatetimeIndex,
) -> pd.Series:
    if records is None or not isinstance(records, pd.DataFrame) or records.empty:
        return pd.Series(np.nan, index=index, dtype=float)
    frame = records.copy()
    if "is_ex_ante_valid" in frame:
        frame = frame.loc[frame["is_ex_ante_valid"].fillna(False)]
    if "possible_reaction_data" in frame:
        frame = frame.loc[~frame["possible_reaction_data"].fillna(False)]
    if frame.empty or "sentiment_score" not in frame:
        return pd.Series(np.nan, index=index, dtype=float)
    dates = pd.to_datetime(
        frame.get("decision_available_date"), errors="coerce", utc=True
    )
    if dates.isna().all():
        return pd.Series(np.nan, index=index, dtype=float)
    dates = dates.dt.tz_convert(None).dt.normalize()
    positions = index.searchsorted(dates.to_numpy(), side="left")
    valid = dates.notna() & (positions < len(index))
    aligned = pd.DataFrame(
        {
            "date": index.take(positions[valid]),
            # Existing sentiment convention is positive risk-on, so invert.
            "risk_score": -pd.to_numeric(
                frame.loc[valid, "sentiment_score"], errors="coerce"
            ).to_numpy(),
        }
    ).dropna()
    daily = aligned.groupby("date")["risk_score"].mean()
    result = daily.reindex(index)
    return result.rolling(21, min_periods=1).mean()


def _align_rbi(
    rbi_macro_index: pd.DataFrame | None,
    index: pd.DatetimeIndex,
) -> pd.Series:
    if (
        rbi_macro_index is None
        or not isinstance(rbi_macro_index, pd.DataFrame)
        or rbi_macro_index.empty
    ):
        return pd.Series(np.nan, index=index, dtype=float)
    column = (
        "macro_risk_score"
        if "macro_risk_score" in rbi_macro_index
        else "decision_macro_risk_score"
    )
    if column not in rbi_macro_index:
        return pd.Series(np.nan, index=index, dtype=float)
    values = pd.to_numeric(rbi_macro_index[column], errors="coerce")
    return values.reindex(index)


def _label(score: float, coverage: float) -> str:
    if coverage < (2 / 3) or not np.isfinite(score):
        return "insufficient_nlp_data"
    if score >= 0.15:
        return "nlp_risk_off"
    if score <= -0.15:
        return "nlp_risk_on"
    return "nlp_neutral"


def build_composite_nlp_risk_index(
    rbi_macro_index=None,
    earnings_sentiment=None,
    news_sentiment=None,
    market_index=None,
    decision_lag: int = 1,
) -> pd.DataFrame:
    """Combine textual components without using returns or reaction features."""
    if int(decision_lag) < 1:
        raise ValueError("decision_lag must be at least 1")
    index = _market_dates(market_index)
    result = pd.DataFrame(index=index)
    result.index.name = "date"
    result["rbi_macro_risk_score"] = _align_rbi(rbi_macro_index, index)
    result["earnings_sector_risk_score"] = _align_text_scores(
        earnings_sentiment, index
    )
    result["news_geopolitical_risk_score"] = _align_text_scores(
        news_sentiment, index
    )
    component_columns = [
        "rbi_macro_risk_score",
        "earnings_sector_risk_score",
        "news_geopolitical_risk_score",
    ]
    available = result[component_columns].notna()
    result["coverage_score"] = available.sum(axis=1) / len(component_columns)
    result["composite_nlp_risk_score"] = result[component_columns].mean(
        axis=1, skipna=True
    )
    result["source_mix"] = [
        json.dumps(
            [
                column.replace("_risk_score", "")
                for column in component_columns
                if bool(available.loc[date, column])
            ]
        )
        for date in result.index
    ]
    result["composite_nlp_label"] = [
        _label(score, coverage)
        for score, coverage in zip(
            result["composite_nlp_risk_score"],
            result["coverage_score"],
        )
    ]
    for column in [
        *component_columns,
        "composite_nlp_risk_score",
        "coverage_score",
        "source_mix",
        "composite_nlp_label",
    ]:
        result[f"decision_{column}"] = result[column].shift(int(decision_lag))
    result["decision_composite_nlp_label"] = result[
        "decision_composite_nlp_label"
    ].fillna("insufficient_nlp_data")
    result["decision_source_date"] = pd.Series(
        result.index, index=result.index
    ).shift(int(decision_lag))
    result["decision_lag"] = int(decision_lag)
    result["commentary_only"] = True
    return result
