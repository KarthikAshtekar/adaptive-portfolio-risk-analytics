"""Plotting utilities for the Streamlit dashboard."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.clustering.dendrograms import DendrogramAnalyzer


def plot_performance_curves(curves: dict[str, pd.Series]) -> go.Figure:
    """Plot portfolio value curves for one or more strategies."""
    fig = go.Figure()
    for name, series in curves.items():
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=name))

    fig.update_layout(
        title="Portfolio Growth Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_drawdown_curves(curves: dict[str, pd.Series]) -> go.Figure:
    """Plot drawdown curves for one or more strategies."""
    fig = go.Figure()
    for name, series in curves.items():
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=name))

    fig.update_layout(
        title="Drawdown Curve",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    """Plot correlation heatmap."""
    fig = px.imshow(
        corr.values,
        x=corr.columns,
        y=corr.index,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Correlation Heatmap",
        aspect="auto",
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    labels: list[Any] | None = None,
    title: str = "Hierarchical Clustering Dendrogram",
) -> go.Figure:
    """Render dendrogram from linkage matrix."""
    return DendrogramAnalyzer.to_plotly_figure(linkage_matrix, labels=labels, title=title)


def plot_weight_bar(weights: pd.Series, title: str = "Latest Portfolio Weights") -> go.Figure:
    """Plot strategy weights as bar chart."""
    fig = go.Figure(
        go.Bar(x=weights.index.tolist(), y=weights.values.tolist(), marker_color="#1f77b4")
    )
    fig.update_layout(
        title=title,
        xaxis_title="Asset",
        yaxis_title="Weight",
        template="plotly_white",
    )
    return fig
