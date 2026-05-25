"""Plotting utilities for Streamlit dashboard."""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def plot_efficient_frontier(
    returns: pd.DataFrame,
    weights_history: pd.DataFrame = None,
    title: str = "Efficient Frontier",
) -> go.Figure:
    """
    Plot efficient frontier.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns
    weights_history : pd.DataFrame, optional
        Historical portfolio weights
    title : str
        Plot title

    Returns
    -------
    go.Figure
        Plotly figure

    TODO: Implement efficient frontier calculation
    """
    fig = go.Figure()

    # TODO: Calculate efficient frontier portfolios
    # TODO: Plot frontier
    # TODO: Add portfolio history

    return fig


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    labels: list = None,
    title: str = "Hierarchical Clustering Dendrogram",
) -> go.Figure:
    """
    Plot hierarchical clustering dendrogram.

    Parameters
    ----------
    linkage_matrix : np.ndarray
        Linkage matrix
    labels : list, optional
        Asset labels
    title : str
        Plot title

    Returns
    -------
    go.Figure
        Plotly figure

    TODO: Implement Plotly dendrogram
    """
    fig = go.Figure()

    # TODO: Create Plotly dendrogram

    return fig


def plot_risk_decomposition(
    risk_contrib: np.ndarray,
    asset_names: list,
    title: str = "Risk Decomposition",
) -> go.Figure:
    """
    Plot portfolio risk decomposition.

    Parameters
    ----------
    risk_contrib : np.ndarray
        Risk contributions by asset
    asset_names : list
        Asset names
    title : str
        Plot title

    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = px.pie(
        values=risk_contrib,
        names=asset_names,
        title=title,
    )

    return fig


def plot_portfolio_returns(
    returns: pd.Series,
    title: str = "Portfolio Returns",
) -> go.Figure:
    """
    Plot cumulative portfolio returns.

    Parameters
    ----------
    returns : pd.Series
        Portfolio returns
    title : str
        Plot title

    Returns
    -------
    go.Figure
        Plotly figure
    """
    cumulative = (1 + returns).cumprod()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=returns.index,
            y=cumulative,
            mode="lines",
            name="Portfolio Value",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        hovermode="x unified",
    )

    return fig


def plot_rolling_metrics(
    returns: pd.Series,
    metric: str = "sharpe",
    window: int = 252,
) -> go.Figure:
    """
    Plot rolling performance metric.

    Parameters
    ----------
    returns : pd.Series
        Portfolio returns
    metric : str
        Metric: 'sharpe', 'volatility', 'correlation'
    window : int
        Rolling window size

    Returns
    -------
    go.Figure
        Plotly figure

    TODO: Implement metric calculation
    """
    fig = go.Figure()

    # TODO: Calculate rolling metrics
    # TODO: Plot time series

    return fig
