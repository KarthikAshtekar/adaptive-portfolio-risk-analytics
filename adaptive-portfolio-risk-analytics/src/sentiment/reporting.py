"""Presentation helpers for sentiment confirmation diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .analytics import calculate_sentiment_confirmation_score


def build_current_sentiment_summary(
    sentiment_signal: pd.DataFrame,
    quantitative_regime: object,
    *,
    stale_after_days: int = 7,
) -> dict[str, object]:
    """Build the compact Manager View sentiment confirmation payload."""
    if sentiment_signal.empty:
        return {
            "quantitative_regime": str(quantitative_regime or "Unknown"),
            "sentiment_label": "unknown",
            "confirmation_status": "Insufficient Sentiment Data",
            "article_coverage": 0,
            "coverage_ratio": 0.0,
            "last_sentiment_date": None,
            "sentiment_warning": "No aligned sentiment observations are available.",
        }

    current = sentiment_signal.iloc[-1]
    article_coverage = int(current.get("decision_article_count", 0) or 0)
    sentiment_label = str(current.get("decision_sentiment_label", "unknown"))
    status = calculate_sentiment_confirmation_score(
        quantitative_regime,
        sentiment_label,
        article_count=article_coverage,
    )
    article_dates = sentiment_signal.index[
        pd.to_numeric(sentiment_signal["article_count"], errors="coerce")
        .fillna(0)
        .gt(0)
    ]
    last_date = article_dates.max() if len(article_dates) else None
    warning = None
    if article_coverage <= 0:
        warning = "Sentiment coverage is insufficient for the current decision date."
    elif last_date is not None:
        age_days = int((sentiment_signal.index.max().normalize() - last_date.normalize()).days)
        if age_days > int(stale_after_days):
            warning = f"Sentiment data is stale by {age_days} calendar days."

    coverage_ratio = float(
        pd.to_numeric(sentiment_signal["article_count"], errors="coerce")
        .fillna(0)
        .gt(0)
        .mean()
    )
    return {
        "quantitative_regime": str(quantitative_regime or "Unknown"),
        "sentiment_label": sentiment_label,
        "confirmation_status": status,
        "article_coverage": article_coverage,
        "coverage_ratio": coverage_ratio,
        "last_sentiment_date": last_date,
        "sentiment_warning": warning,
    }


def sentiment_commentary(status: str) -> str:
    """Return non-prescriptive recommendation commentary for one status."""
    if status == "Confirmed Risk-Off":
        return "Sentiment confirms the quantitative stress signal."
    if status == "Confirmed Risk-On":
        return "Sentiment confirms the quantitative risk-on signal."
    if status == "Confirmed Neutral":
        return "Sentiment confirms a neutral quantitative regime."
    if status == "Quant-Sentiment Disagreement":
        return (
            "Sentiment disagrees with the quantitative regime, so recommendation "
            "confidence is not upgraded."
        )
    return (
        "Sentiment coverage is insufficient; the strategy recommendation remains "
        "based on quantitative evidence."
    )


def plot_daily_sentiment_signal(signal: pd.DataFrame) -> go.Figure:
    """Plot daily and rolling observed sentiment scores."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=signal.index,
            y=signal["daily_sentiment_score"],
            name="Daily score",
            marker_color="#A3BEFA",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=signal.index,
            y=signal["rolling_sentiment_score"],
            mode="lines",
            name="Rolling observed score",
            line={"color": "#2E4780", "width": 2},
        )
    )
    fig.add_hline(y=0.0, line_color="#6F768A", line_dash="dot")
    fig.update_layout(
        title="Daily Market Sentiment",
        xaxis_title="Date",
        yaxis_title="Risk-on (+) / Risk-off (-)",
        template="plotly_white",
        legend={"orientation": "h"},
    )
    return fig


def plot_sentiment_regime_timeline(
    signal: pd.DataFrame,
    regimes: pd.Series,
    *,
    title: str,
) -> go.Figure:
    """Plot lagged sentiment score against categorical quantitative regimes."""
    regime_map = {
        "Risk-Off": -2,
        "Crisis": -2,
        "Stress": -1,
        "Unknown": 0,
        "Normal": 0,
        "Calm": 1,
        "Risk-On": 1,
    }
    aligned = regimes.reindex(signal.index).astype(str)
    numeric = aligned.map(regime_map).fillna(0)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=signal.index,
            y=signal["decision_sentiment_score"],
            mode="lines",
            name="Lagged sentiment score",
            line={"color": "#2E4780", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=numeric.index,
            y=numeric.values,
            mode="lines",
            name="Quantitative regime scale",
            line={"color": "#B56A3B", "width": 1.5, "dash": "dot"},
            text=aligned,
            hovertemplate="%{x}<br>Regime: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Confirmation scale",
        template="plotly_white",
        legend={"orientation": "h"},
    )
    return fig

