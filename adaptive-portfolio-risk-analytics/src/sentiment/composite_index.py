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
VALID_COMPOSITE_NLP_LABELS = COMPOSITE_NLP_LABELS[:3]
COMPONENT_COLUMNS = [
    "rbi_macro_risk_score",
    "earnings_sector_risk_score",
    "news_geopolitical_risk_score",
]
COMPONENT_SOURCE_NAMES = {
    "rbi_macro_risk_score": "rbi_macro",
    "earnings_sector_risk_score": "earnings",
    "news_geopolitical_risk_score": "news",
}


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


def _eligible_text_frame(records: pd.DataFrame | None) -> pd.DataFrame:
    if records is None or not isinstance(records, pd.DataFrame) or records.empty:
        return pd.DataFrame()
    frame = records.copy()
    if "is_ex_ante_valid" in frame:
        frame = frame.loc[frame["is_ex_ante_valid"].fillna(False)]
    if "possible_reaction_data" in frame:
        frame = frame.loc[~frame["possible_reaction_data"].fillna(False)]
    return frame.copy()


def _record_available_dates(frame: pd.DataFrame) -> pd.Series:
    if "decision_available_date" in frame:
        source = frame["decision_available_date"]
    elif "publication_time" in frame:
        source = frame["publication_time"]
    else:
        source = pd.Series(pd.NaT, index=frame.index)
    return pd.to_datetime(source, errors="coerce", utc=True)


def _align_records_to_market(
    records: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    if records.empty:
        return records.copy()
    frame = records.copy()
    dates = _record_available_dates(frame)
    if dates.isna().all():
        aligned = frame.iloc[0:0].copy()
        aligned["_market_date"] = pd.NaT
        return aligned
    normalized = dates.dt.tz_convert(None).dt.normalize()
    positions = index.searchsorted(normalized.to_numpy(), side="left")
    valid = normalized.notna() & (positions < len(index))
    aligned = frame.loc[valid].copy()
    aligned["_market_date"] = index.take(positions[valid])
    return aligned


def _risk_scores(frame: pd.DataFrame) -> pd.Series:
    if "risk_score" in frame:
        return pd.to_numeric(frame["risk_score"], errors="coerce")
    if "sentiment_score" in frame:
        # Sentiment convention: positive is risk-on, negative is risk-off.
        # NLP risk convention: positive is risk-off pressure.
        return -pd.to_numeric(frame["sentiment_score"], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _sentiment_scores(frame: pd.DataFrame) -> pd.Series:
    if "sentiment_score" in frame:
        return pd.to_numeric(frame["sentiment_score"], errors="coerce")
    if "risk_score" in frame:
        return -pd.to_numeric(frame["risk_score"], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _align_text_scores(
    records: pd.DataFrame | None,
    index: pd.DatetimeIndex,
) -> pd.Series:
    frame = _eligible_text_frame(records)
    if frame.empty:
        return pd.Series(np.nan, index=index, dtype=float)
    aligned_records = _align_records_to_market(frame, index)
    if aligned_records.empty:
        return pd.Series(np.nan, index=index, dtype=float)
    aligned = pd.DataFrame(
        {
            "date": aligned_records["_market_date"].to_numpy(),
            "risk_score": _risk_scores(aligned_records).to_numpy(),
        }
    ).dropna()
    daily = aligned.groupby("date")["risk_score"].mean()
    result = daily.reindex(index)
    return result.rolling(21, min_periods=1).mean()


def _align_text_sentiment(
    records: pd.DataFrame | None,
    index: pd.DatetimeIndex,
) -> pd.Series:
    frame = _eligible_text_frame(records)
    if frame.empty:
        return pd.Series(np.nan, index=index, dtype=float)
    aligned_records = _align_records_to_market(frame, index)
    if aligned_records.empty:
        return pd.Series(np.nan, index=index, dtype=float)
    aligned = pd.DataFrame(
        {
            "date": aligned_records["_market_date"].to_numpy(),
            "sentiment_score": _sentiment_scores(aligned_records).to_numpy(),
        }
    ).dropna()
    daily = aligned.groupby("date")["sentiment_score"].mean()
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


def _source_mix(active_sources: list[str]) -> str:
    if not active_sources:
        return "none"
    unique = list(dict.fromkeys(active_sources))
    source_set = set(unique)
    if source_set == {"rbi_macro"}:
        return "rbi_only"
    if source_set == {"news"}:
        return "news_only"
    if {"rbi_macro", "news"}.issubset(source_set):
        return "rbi_and_news"
    if len(unique) == 1:
        return f"{unique[0]}_only"
    return "multi_source"


def _label(score: float, source_mix: str) -> str:
    if not np.isfinite(score) or source_mix == "none":
        return "insufficient_nlp_data"
    if source_mix not in {"news_only", "rbi_only", "rbi_and_news", "multi_source"}:
        return "insufficient_nlp_data"
    if score >= 0.15:
        return "nlp_risk_off"
    if score <= -0.15:
        return "nlp_risk_on"
    return "nlp_neutral"


def _label_reason(
    *,
    label: str,
    source_mix: str,
    score: float,
) -> str:
    if label in VALID_COMPOSITE_NLP_LABELS:
        return ""
    if source_mix == "none":
        return "no_valid_nlp_source_in_rolling_window"
    if source_mix not in {"news_only", "rbi_only", "rbi_and_news", "multi_source"}:
        return f"unsupported_single_source_mix:{source_mix}"
    if not np.isfinite(score):
        return "missing_composite_score"
    return "insufficient_nlp_data"


def _component_label(score: float) -> str:
    if not np.isfinite(score):
        return "insufficient_nlp_data"
    if score >= 0.15:
        return "nlp_risk_off"
    if score <= -0.15:
        return "nlp_risk_on"
    return "nlp_neutral"


def build_daily_nlp_signal(
    records: pd.DataFrame,
    market_index,
    *,
    decision_lag: int = 1,
    rolling_window: int = 21,
) -> pd.DataFrame:
    """Build auditable daily NLP diagnostics from news/GDELT records.

    The sign convention is explicit: positive `news_geopolitical_risk_score`
    means risk-off/news stress, negative values mean risk-on/supportive news.
    The output is a monitoring signal only and has no allocation side effects.
    """
    if int(decision_lag) < 1:
        raise ValueError("decision_lag must be at least 1")
    if int(rolling_window) < 1:
        raise ValueError("rolling_window must be at least 1")
    index = _market_dates(market_index)
    result = pd.DataFrame({"date": index})
    result["raw_record_count"] = 0
    result["valid_record_count"] = 0
    result["english_record_count"] = 0
    result["high_quality_record_count"] = 0
    result["mean_sentiment_score"] = np.nan
    result["mean_risk_score"] = np.nan
    result["news_geopolitical_risk_score"] = np.nan

    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame()
    if not frame.empty:
        aligned_all = _align_records_to_market(frame, index)
        if not aligned_all.empty:
            raw_counts = aligned_all.groupby("_market_date").size()
            result["raw_record_count"] = (
                result["date"].map(raw_counts).fillna(0).astype(int)
            )

        valid_frame = _eligible_text_frame(frame)
        if not valid_frame.empty:
            valid_frame = valid_frame.copy()
            valid_frame["_sentiment_score"] = _sentiment_scores(valid_frame)
            valid_frame["_risk_score"] = _risk_scores(valid_frame)
            valid_frame = valid_frame.loc[
                valid_frame["_sentiment_score"].notna()
                | valid_frame["_risk_score"].notna()
            ].copy()
            aligned_valid = _align_records_to_market(valid_frame, index)
            if not aligned_valid.empty:
                grouped = aligned_valid.groupby("_market_date")
                valid_counts = grouped.size()
                result["valid_record_count"] = (
                    result["date"].map(valid_counts).fillna(0).astype(int)
                )
                language = (
                    aligned_valid.get(
                        "language",
                        pd.Series("", index=aligned_valid.index),
                    )
                    .fillna("")
                    .astype(str)
                    .str.lower()
                )
                english_counts = (
                    aligned_valid.loc[
                        language.isin({"en", "eng", "english"})
                    ]
                    .groupby("_market_date")
                    .size()
                )
                result["english_record_count"] = (
                    result["date"].map(english_counts).fillna(0).astype(int)
                )
                quality = (
                    aligned_valid.get(
                        "source_quality_label",
                        pd.Series("", index=aligned_valid.index),
                    )
                    .fillna("")
                    .astype(str)
                    .str.lower()
                )
                high_counts = (
                    aligned_valid.loc[quality.eq("high")]
                    .groupby("_market_date")
                    .size()
                )
                result["high_quality_record_count"] = (
                    result["date"].map(high_counts).fillna(0).astype(int)
                )
                daily_sentiment = grouped["_sentiment_score"].mean()
                daily_risk = grouped["_risk_score"].mean()
                result["mean_sentiment_score"] = result["date"].map(
                    daily_sentiment
                )
                result["mean_risk_score"] = result["date"].map(daily_risk)
                result["news_geopolitical_risk_score"] = (
                    result.set_index("date")["mean_risk_score"]
                    .rolling(int(rolling_window), min_periods=1)
                    .mean()
                    .reindex(index)
                    .to_numpy()
                )

    valid_days = result["valid_record_count"].gt(0).astype(float)
    rolling_observations = (
        pd.Series(1.0, index=index)
        .rolling(int(rolling_window), min_periods=1)
        .sum()
    )
    rolling_valid_days = (
        pd.Series(valid_days.to_numpy(), index=index)
        .rolling(int(rolling_window), min_periods=1)
        .sum()
    )
    result["rolling_article_day_coverage"] = (
        rolling_valid_days / rolling_observations.replace(0, np.nan)
    ).fillna(0.0).to_numpy()
    raw_source_mix = np.where(
        pd.to_numeric(
            result["news_geopolitical_risk_score"], errors="coerce"
        ).notna(),
        "news_only",
        "none",
    )
    result["raw_source_mix"] = raw_source_mix
    result["coverage_score"] = np.where(raw_source_mix == "news_only", 1 / 3, 0.0)
    result["raw_nlp_label"] = [
        _label(score, source_mix)
        for score, source_mix in zip(
            pd.to_numeric(
                result["news_geopolitical_risk_score"], errors="coerce"
            ),
            result["raw_source_mix"],
        )
    ]
    raw_reasons = [
        _label_reason(label=label, source_mix=source_mix, score=score)
        for label, source_mix, score in zip(
            result["raw_nlp_label"],
            result["raw_source_mix"],
            pd.to_numeric(
                result["news_geopolitical_risk_score"], errors="coerce"
            ),
        )
    ]
    lag = int(decision_lag)
    result["decision_nlp_label"] = (
        result["raw_nlp_label"]
        .shift(lag)
        .fillna("insufficient_nlp_data")
    )
    result["decision_news_geopolitical_risk_score"] = result[
        "news_geopolitical_risk_score"
    ].shift(lag)
    result["source_mix"] = result["raw_source_mix"].shift(lag).fillna("none")
    result["decision_source_date"] = pd.Series(index, index=result.index).shift(
        lag
    )
    shifted_reasons = pd.Series(raw_reasons, index=result.index).shift(lag)
    result["insufficient_reason"] = np.where(
        result["decision_nlp_label"].isin(VALID_COMPOSITE_NLP_LABELS),
        "",
        shifted_reasons.fillna("decision_lag_no_prior_signal"),
    )
    ordered = [
        "date",
        "raw_record_count",
        "valid_record_count",
        "english_record_count",
        "high_quality_record_count",
        "mean_sentiment_score",
        "mean_risk_score",
        "news_geopolitical_risk_score",
        "raw_nlp_label",
        "decision_nlp_label",
        "coverage_score",
        "source_mix",
        "insufficient_reason",
        "rolling_article_day_coverage",
        "raw_source_mix",
        "decision_news_geopolitical_risk_score",
        "decision_source_date",
    ]
    return result.loc[:, ordered]


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
    result["news_sentiment_score"] = _align_text_sentiment(
        news_sentiment, index
    )
    result["news_risk_label"] = [
        _component_label(score)
        for score in result["news_geopolitical_risk_score"]
    ]
    component_columns = COMPONENT_COLUMNS
    available = result[component_columns].notna()
    result["coverage_score"] = available.sum(axis=1) / len(component_columns)
    result["composite_nlp_risk_score"] = result[component_columns].mean(
        axis=1, skipna=True
    )
    result["source_mix"] = [
        _source_mix(
            [
                COMPONENT_SOURCE_NAMES[column]
                for column in component_columns
                if bool(available.loc[date, column])
            ]
        )
        for date in result.index
    ]
    result["source_mix_components"] = [
        json.dumps(
            [
                COMPONENT_SOURCE_NAMES[column]
                for column in component_columns
                if bool(available.loc[date, column])
            ],
            sort_keys=True,
        )
        for date in result.index
    ]
    result["composite_nlp_label"] = [
        _label(score, source_mix)
        for score, source_mix in zip(
            result["composite_nlp_risk_score"],
            result["source_mix"],
        )
    ]
    result["insufficient_reason"] = [
        _label_reason(label=label, source_mix=source_mix, score=score)
        for label, source_mix, score in zip(
            result["composite_nlp_label"],
            result["source_mix"],
            result["composite_nlp_risk_score"],
        )
    ]
    for column in [
        *component_columns,
        "news_sentiment_score",
        "news_risk_label",
        "composite_nlp_risk_score",
        "coverage_score",
        "source_mix",
        "source_mix_components",
        "composite_nlp_label",
        "insufficient_reason",
    ]:
        result[f"decision_{column}"] = result[column].shift(int(decision_lag))
    result["decision_composite_nlp_label"] = result[
        "decision_composite_nlp_label"
    ].fillna("insufficient_nlp_data")
    result["decision_insufficient_reason"] = result[
        "decision_insufficient_reason"
    ].fillna("decision_lag_no_prior_signal")
    result["decision_source_date"] = pd.Series(
        result.index, index=result.index
    ).shift(int(decision_lag))
    result["decision_lag"] = int(decision_lag)
    result["commentary_only"] = True
    return result
