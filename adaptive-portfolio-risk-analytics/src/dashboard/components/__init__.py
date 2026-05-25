"""Streamlit dashboard components."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def render_metric_card(title: str, value: str, change: str, color: str = "green"):
    """
    Render a metric card.

    Parameters
    ----------
    title : str
        Metric title
    value : str
        Current value
    change : str
        Change indicator
    color : str
        Color indicator

    TODO: Implement custom styling
    """
    st.metric(title, value, change)


def render_portfolio_summary(portfolio_data: dict):
    """
    Render portfolio summary section.

    Parameters
    ----------
    portfolio_data : dict
        Portfolio metrics

    TODO: Implement full summary display
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Portfolio Value", "1.00M", "+5.2%")
    with col2:
        st.metric("Sharpe Ratio", "1.45", "+0.15")
    with col3:
        st.metric("Max Drawdown", "-8.3%", "+1.2%")
    with col4:
        st.metric("Volatility", "12.5%", "-0.5%")


def render_allocation_table(weights: dict):
    """
    Render portfolio allocation table.

    Parameters
    ----------
    weights : dict
        Asset weights

    TODO: Implement interactive allocation table
    """
    data = {"Asset": list(weights.keys()), "Weight %": list(weights.values())}
    df = pd.DataFrame(data)
    st.dataframe(df)


def render_performance_chart(returns: pd.Series):
    """
    Render cumulative return chart.

    Parameters
    ----------
    returns : pd.Series
        Portfolio returns

    TODO: Implement Plotly chart
    """
    cumulative = (1 + returns).cumprod()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=returns.index,
            y=cumulative,
            mode="lines",
            name="Portfolio",
        )
    )

    fig.update_layout(
        title="Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Value",
    )

    st.plotly_chart(fig)
