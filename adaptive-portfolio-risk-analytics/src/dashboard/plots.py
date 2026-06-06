"""Plotting utilities for the Streamlit dashboard."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram


# ============================================================
# PERFORMANCE CHARTS
# ============================================================


def plot_performance_curves(
    curves: dict[str, pd.Series],
) -> go.Figure:
    """Compare portfolio growth curves across strategies."""

    fig = go.Figure()

    for name, series in curves.items():
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=name,
            )
        )

    fig.update_layout(
        title="Portfolio Growth Comparison",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        template="plotly_white",
        hovermode="x unified",
    )

    return fig


def plot_equity_curve(
    portfolio_values: pd.Series,
    title: str = "Equity Curve",
) -> go.Figure:
    """Single portfolio growth curve."""

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=portfolio_values.index,
            y=portfolio_values.values,
            mode="lines",
            name="Portfolio",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        template="plotly_white",
        hovermode="x unified",
    )

    return fig


# ============================================================
# DRAWDOWNS
# ============================================================


def plot_drawdown_curves(
    curves: dict[str, pd.Series],
) -> go.Figure:
    """Compare drawdown curves across strategies."""

    fig = go.Figure()

    for name, series in curves.items():
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=name,
            )
        )

    fig.update_layout(
        title="Drawdown Comparison",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        template="plotly_white",
        hovermode="x unified",
    )

    return fig


def plot_drawdowns(
    drawdown_series: pd.Series,
    title: str = "Drawdown",
) -> go.Figure:
    """Single portfolio drawdown chart."""

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=drawdown_series.index,
            y=drawdown_series.values,
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown",
        template="plotly_white",
        hovermode="x unified",
    )

    return fig


# ============================================================
# VOLATILITY
# ============================================================


def plot_rolling_volatility(
    rolling_vol: pd.Series,
    title: str = "Rolling Volatility",
) -> go.Figure:
    """Rolling volatility chart."""

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol.values,
            mode="lines",
            name="Rolling Volatility",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Volatility",
        template="plotly_white",
        hovermode="x unified",
    )

    return fig


# ============================================================
# CORRELATION
# ============================================================


def plot_correlation_heatmap(
    corr: pd.DataFrame,
) -> go.Figure:
    """Correlation matrix heatmap."""

    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Correlation Heatmap",
        aspect="auto",
    )

    fig.update_layout(
        template="plotly_white",
    )

    return fig


# ============================================================
# CLUSTERING
# ============================================================


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    labels: list[Any] | None = None,
    title: str = "Hierarchical Clustering Dendrogram",
):
    """
    Return a matplotlib figure.

    Streamlit usage:

    fig = plot_dendrogram(...)
    st.pyplot(fig)
    """

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    scipy_dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=45,
        leaf_font_size=10,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_ylabel("Distance")

    plt.tight_layout()

    return fig


# ============================================================
# PORTFOLIO WEIGHTS
# ============================================================


def plot_weight_bar(
    weights: pd.Series,
    title: str = "Portfolio Weights",
) -> go.Figure:
    """Portfolio weights bar chart."""

    fig = go.Figure(
        go.Bar(
            x=weights.index.tolist(),
            y=weights.values.tolist(),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Asset",
        yaxis_title="Weight",
        template="plotly_white",
    )

    return fig


def plot_weights(
    weights: pd.Series,
    title: str = "Portfolio Weights",
) -> go.Figure:
    """Alias."""

    return plot_weight_bar(
        weights,
        title,
    )


def plot_weight_pie(
    weights: pd.Series,
    title: str = "Portfolio Allocation",
) -> go.Figure:
    """Portfolio allocation pie chart."""

    fig = go.Figure(
        go.Pie(
            labels=weights.index.tolist(),
            values=weights.values.tolist(),
            hole=0.35,
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
    )

    return fig


def plot_weight_comparison(
    comparison_df: pd.DataFrame,
    left_col: str = "HRP Weight",
    right_col: str = "HERC Weight",
    title: str = "HRP vs HERC Weight Comparison",
) -> go.Figure:
    """Plot grouped portfolio weights for two allocation schemes."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=comparison_df["Asset"].tolist(),
            y=comparison_df[left_col].tolist(),
            name=left_col,
        )
    )
    fig.add_trace(
        go.Bar(
            x=comparison_df["Asset"].tolist(),
            y=comparison_df[right_col].tolist(),
            name=right_col,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Asset",
        yaxis_title="Weight",
        barmode="group",
        template="plotly_white",
    )

    return fig


# ============================================================
# STRATEGY COMPARISON
# ============================================================


def plot_strategy_comparison(
    strategy_metrics: dict[str, dict[str, float]],
    metric_name: str = "cagr",
) -> go.Figure:
    """Compare a metric across strategies."""

    strategies = list(strategy_metrics.keys())

    values = [
        strategy_metrics[strategy].get(
            metric_name,
            0.0,
        )
        for strategy in strategies
    ]

    fig = go.Figure(
        go.Bar(
            x=strategies,
            y=values,
        )
    )

    fig.update_layout(
        title=f"{metric_name.upper()} Comparison",
        xaxis_title="Strategy",
        yaxis_title=metric_name.upper(),
        template="plotly_white",
    )

    return fig


# ============================================================
# STREAMLIT HELPERS
# ============================================================


def format_metric_cards(
    metrics: dict[str, float],
) -> dict[str, str]:
    """
    Format metrics for Streamlit KPI cards.
    """

    formatted = {}

    for key, value in metrics.items():

        key_lower = key.lower()

        if any(
            x in key_lower
            for x in [
                "return",
                "cagr",
                "volatility",
                "drawdown",
            ]
        ):
            formatted[key] = f"{value:.2%}"

        else:
            formatted[key] = f"{value:.2f}"

    return formatted
