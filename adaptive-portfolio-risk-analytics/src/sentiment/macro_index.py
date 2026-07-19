"""Market-aligned RBI macro stance index with explicit decision lagging."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .alignment import validate_market_index


MACRO_LABELS = (
    "risk_on_macro",
    "neutral_macro",
    "risk_off_macro",
    "insufficient_macro_data",
)


def _assign_to_market_dates(
    scored_sentences: pd.DataFrame,
    market_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    assigned = scored_sentences.copy()
    dates = pd.to_datetime(
        assigned["publication_date"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None)
    assigned["publication_date"] = dates
    assigned = assigned.loc[dates.notna()].copy()
    normalized = assigned["publication_date"].dt.normalize().to_numpy(dtype="datetime64[ns]")
    positions = market_index.values.searchsorted(normalized, side="left")
    valid = positions < len(market_index)
    assigned = assigned.loc[valid].copy()
    assigned["market_date"] = market_index.take(positions[valid])
    return assigned


def _macro_label(
    macro_risk_score: float,
    sentence_count: int,
    *,
    risk_off_threshold: float,
    risk_on_threshold: float,
) -> str:
    if sentence_count <= 0 or not np.isfinite(macro_risk_score):
        return "insufficient_macro_data"
    if macro_risk_score >= risk_off_threshold:
        return "risk_off_macro"
    if macro_risk_score <= risk_on_threshold:
        return "risk_on_macro"
    return "neutral_macro"


def build_macro_stance_index(
    scored_sentences: pd.DataFrame,
    market_index: pd.DatetimeIndex,
    lookback_window: int = 63,
    decision_lag: int = 1,
    *,
    risk_off_threshold: float = 0.25,
    risk_on_threshold: float = -0.20,
) -> pd.DataFrame:
    """Build observed and trading-safe RBI macro stance confirmation signals."""
    index = validate_market_index(market_index)
    if int(lookback_window) <= 0:
        raise ValueError("lookback_window must be positive")
    if int(decision_lag) < 1:
        raise ValueError("decision_lag must be at least 1")
    required = {
        "document_id",
        "publication_date",
        "stance_label",
        "certainty_label",
        "time_label",
    }
    missing = required.difference(scored_sentences.columns)
    if missing:
        raise ValueError(f"scored_sentences are missing required columns: {sorted(missing)}")

    assigned = _assign_to_market_dates(scored_sentences, index)
    daily = pd.DataFrame(index=index)
    daily.index.name = "date"
    for label, column in (
        ("hawkish", "hawkish_count"),
        ("dovish", "dovish_count"),
    ):
        values = assigned["stance_label"].astype(str).str.lower().eq(label).astype(int)
        daily[column] = values.groupby(assigned["market_date"]).sum().reindex(index, fill_value=0)
    daily["uncertainty_count"] = (
        assigned["certainty_label"]
        .astype(str)
        .str.lower()
        .eq("uncertain")
        .astype(int)
        .groupby(assigned["market_date"])
        .sum()
        .reindex(index, fill_value=0)
    )
    daily["forward_looking_count"] = (
        assigned["time_label"]
        .astype(str)
        .str.lower()
        .eq("forward_looking")
        .astype(int)
        .groupby(assigned["market_date"])
        .sum()
        .reindex(index, fill_value=0)
    )
    daily["daily_sentence_count"] = (
        assigned.groupby("market_date").size().reindex(index, fill_value=0).astype(int)
    )
    daily["daily_document_count"] = (
        assigned.groupby("market_date")["document_id"]
        .nunique()
        .reindex(index, fill_value=0)
        .astype(int)
    )
    daily["latest_publication_date"] = (
        assigned.groupby("market_date")["publication_date"].max().reindex(index)
    )

    rolling = (
        daily[
            [
                "hawkish_count",
                "dovish_count",
                "uncertainty_count",
                "forward_looking_count",
                "daily_sentence_count",
            ]
        ]
        .rolling(int(lookback_window), min_periods=1)
        .sum()
    )
    result = pd.DataFrame(index=index)
    result.index.name = "date"
    denominator = rolling["daily_sentence_count"].replace(0, np.nan)
    result["hawkish_share"] = rolling["hawkish_count"] / denominator
    result["dovish_share"] = rolling["dovish_count"] / denominator
    result["uncertainty_share"] = rolling["uncertainty_count"] / denominator
    result["forward_looking_share"] = rolling["forward_looking_count"] / denominator
    result["net_stance_score"] = result["hawkish_share"] - result["dovish_share"]
    result["macro_risk_score"] = result["net_stance_score"] + result["uncertainty_share"]
    result["document_count"] = (
        daily["daily_document_count"].rolling(int(lookback_window), min_periods=1).sum().astype(int)
    )
    result["sentence_count"] = rolling["daily_sentence_count"].astype(int)
    result["coverage_flag"] = np.where(
        result["sentence_count"].gt(0),
        "covered",
        "insufficient_macro_data",
    )
    result["macro_label"] = [
        _macro_label(
            risk_score,
            int(sentence_count),
            risk_off_threshold=float(risk_off_threshold),
            risk_on_threshold=float(risk_on_threshold),
        )
        for risk_score, sentence_count in zip(
            result["macro_risk_score"],
            result["sentence_count"],
        )
    ]
    result["decision_macro_label"] = (
        result["macro_label"].shift(int(decision_lag)).fillna("insufficient_macro_data")
    )
    result["decision_macro_risk_score"] = result["macro_risk_score"].shift(int(decision_lag))
    for column in (
        "hawkish_share",
        "dovish_share",
        "uncertainty_share",
        "forward_looking_share",
        "net_stance_score",
    ):
        result[f"decision_{column}"] = result[column].shift(int(decision_lag))
    result["decision_document_count"] = (
        result["document_count"].shift(int(decision_lag)).fillna(0).astype(int)
    )
    result["decision_sentence_count"] = (
        result["sentence_count"].shift(int(decision_lag)).fillna(0).astype(int)
    )
    result["decision_coverage_flag"] = (
        result["coverage_flag"].shift(int(decision_lag)).fillna("insufficient_macro_data")
    )
    publication_days = (
        daily["latest_publication_date"].astype("int64") // 86_400_000_000_000
    ).astype(float)
    publication_days.loc[daily["latest_publication_date"].isna()] = np.nan
    rolling_latest = pd.to_datetime(
        publication_days.rolling(
            int(lookback_window),
            min_periods=1,
        ).max(),
        unit="D",
        errors="coerce",
    )
    rolling_latest = pd.Series(rolling_latest, index=index)
    result["latest_publication_date"] = rolling_latest
    result["decision_source_date"] = rolling_latest.shift(int(decision_lag))
    result["lookback_window"] = int(lookback_window)
    result["decision_lag"] = int(decision_lag)
    return result


def macro_confirmation_status(
    quantitative_regime: object,
    macro_label: object,
    *,
    sentence_count: int | float = 0,
) -> str:
    """Describe macro confirmation without changing quantitative decisions."""
    regime = str(quantitative_regime or "unknown").strip().lower()
    label = str(macro_label or "insufficient_macro_data").strip().lower()
    if sentence_count <= 0 or label == "insufficient_macro_data":
        return "Insufficient Macro Data"
    if label == "risk_off_macro" and regime in {
        "stress",
        "crisis",
        "risk-off",
        "risk_off",
    }:
        return "Confirmed Risk-Off"
    if label == "risk_on_macro" and regime in {
        "calm",
        "normal",
        "risk-on",
        "risk_on",
    }:
        return "Confirmed Risk-On"
    if label == "neutral_macro" and regime == "normal":
        return "Confirmed Neutral"
    return "Quant-Macro Disagreement"


def build_current_macro_summary(
    macro_index: pd.DataFrame,
    quantitative_regime: object,
) -> dict[str, object]:
    """Return the compact Manager View macro-sentiment payload."""
    if macro_index.empty:
        return {
            "quantitative_regime": str(quantitative_regime or "Unknown"),
            "macro_sentiment_label": "insufficient_macro_data",
            "macro_sentiment_confirmation": "Insufficient Macro Data",
            "macro_sentiment_coverage": 0,
            "macro_sentiment_sentence_coverage": 0,
            "coverage_status": "Insufficient: 0 RBI documents in rolling window",
            "macro_sentiment_warning": "No aligned RBI macro sentences are available.",
            "stance_summary": "No macro stance evidence",
            "uncertainty_share": np.nan,
            "last_document_date": None,
        }
    current = macro_index.iloc[-1]
    sentence_count = int(current.get("decision_sentence_count", 0) or 0)
    document_count = int(current.get("decision_document_count", 0) or 0)
    label = str(current.get("decision_macro_label", "insufficient_macro_data"))
    confirmation = macro_confirmation_status(
        quantitative_regime,
        label,
        sentence_count=sentence_count,
    )
    warning = (
        "RBI macro-sentiment coverage is insufficient for the current decision date."
        if document_count <= 0 or sentence_count <= 0
        else None
    )
    source_dates = pd.to_datetime(
        macro_index.get(
            "decision_source_date",
            pd.Series(pd.NaT, index=macro_index.index),
        ),
        errors="coerce",
    ).dropna()
    last_document_date = source_dates.max() if not source_dates.empty else None
    net_stance = current.get("decision_net_stance_score")
    stance_summary = (
        "Hawkish tilt"
        if pd.notna(net_stance) and float(net_stance) > 0.10
        else "Dovish tilt"
        if pd.notna(net_stance) and float(net_stance) < -0.10
        else "Neutral stance"
    )
    return {
        "quantitative_regime": str(quantitative_regime or "Unknown"),
        "macro_sentiment_label": label,
        "macro_sentiment_confirmation": confirmation,
        "macro_sentiment_coverage": document_count,
        "macro_sentiment_sentence_coverage": sentence_count,
        "coverage_status": (
            f"Covered: {document_count} RBI document(s), "
            f"{sentence_count} sentence(s) in rolling window"
            if document_count > 0 and sentence_count > 0
            else "Insufficient: 0 RBI documents in rolling window"
        ),
        "macro_sentiment_warning": warning,
        "stance_summary": stance_summary,
        "uncertainty_share": current.get("decision_uncertainty_share"),
        "last_document_date": last_document_date,
    }


def plot_macro_stance_index(macro_index: pd.DataFrame) -> go.Figure:
    """Plot macro risk and its hawkish, dovish, and uncertainty components."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=macro_index.index,
            y=macro_index["decision_macro_risk_score"],
            name="Lagged macro risk score",
            line={"color": "#8A3B3B", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=macro_index.index,
            y=macro_index["hawkish_share"],
            name="Hawkish share",
            line={"color": "#B56A3B", "width": 1},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=macro_index.index,
            y=-macro_index["dovish_share"],
            name="Dovish share (-)",
            line={"color": "#3F7C73", "width": 1},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=macro_index.index,
            y=macro_index["uncertainty_share"],
            name="Uncertainty share",
            line={"color": "#7A5AA6", "width": 1, "dash": "dot"},
        )
    )
    fig.add_hline(y=0.0, line_color="#6F768A", line_dash="dot")
    fig.update_layout(
        title="RBI Macro Stance Index",
        xaxis_title="Date",
        yaxis_title="Macro risk / stance share",
        template="plotly_white",
        legend={"orientation": "h"},
    )
    return fig


def plot_macro_stance_shares(macro_index: pd.DataFrame) -> go.Figure:
    """Plot hawkish and dovish rolling sentence shares."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=macro_index.index,
            y=macro_index["hawkish_share"],
            name="Hawkish share",
            line={"color": "#B56A3B", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=macro_index.index,
            y=macro_index["dovish_share"],
            name="Dovish share",
            line={"color": "#3F7C73", "width": 2},
        )
    )
    fig.update_layout(
        title="Hawkish vs Dovish Share",
        xaxis_title="Date",
        yaxis_title="Rolling sentence share",
        yaxis={"range": [0, 1]},
        template="plotly_white",
        legend={"orientation": "h"},
    )
    return fig


def plot_macro_uncertainty_share(macro_index: pd.DataFrame) -> go.Figure:
    """Plot uncertainty and forward-looking rolling shares."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=macro_index.index,
            y=macro_index["uncertainty_share"],
            name="Uncertainty share",
            line={"color": "#7A5AA6", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=macro_index.index,
            y=macro_index["forward_looking_share"],
            name="Forward-looking share",
            line={"color": "#2E4780", "width": 1.5, "dash": "dot"},
        )
    )
    fig.update_layout(
        title="Uncertainty and Forward-Looking Language",
        xaxis_title="Date",
        yaxis_title="Rolling sentence share",
        yaxis={"range": [0, 1]},
        template="plotly_white",
        legend={"orientation": "h"},
    )
    return fig


def plot_macro_regime_timeline(
    macro_index: pd.DataFrame,
    regimes: pd.Series,
    *,
    title: str,
) -> go.Figure:
    """Plot the lagged macro-risk score against a quantitative regime scale."""
    regime_map = {
        "Crisis": 2,
        "Stress": 1,
        "Risk-Off": 1,
        "Normal": 0,
        "Unknown": 0,
        "Calm": -1,
        "Risk-On": -1,
    }
    aligned = regimes.reindex(macro_index.index).astype(str)
    numeric = aligned.map(regime_map).fillna(0)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=macro_index.index,
            y=macro_index["decision_macro_risk_score"],
            name="Lagged macro-risk score",
            line={"color": "#8A3B3B", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=numeric.index,
            y=numeric.values,
            name="Quantitative regime scale",
            mode="lines",
            line={"color": "#2E4780", "width": 1.5, "dash": "dot"},
            text=aligned,
            hovertemplate="%{x}<br>Regime: %{text}<extra></extra>",
        )
    )
    fig.add_hline(y=0.0, line_color="#6F768A", line_dash="dot")
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Risk-off (+) / risk-on (-)",
        template="plotly_white",
        legend={"orientation": "h"},
    )
    return fig
