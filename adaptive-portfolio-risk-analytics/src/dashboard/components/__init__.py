"""Reusable Streamlit dashboard components."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.modes import net_metric_label


def render_metric_card(
    title: str,
    value: str,
    delta: str | None = None,
) -> None:
    """
    Render a Streamlit metric card.
    """
    st.metric(
        label=title,
        value=value,
        delta=delta,
    )


def render_portfolio_summary(
    metrics: dict[str, str],
) -> None:
    """
    Render KPI metrics in a row.
    """

    columns = st.columns(len(metrics))

    for col, (metric, value) in zip(
        columns,
        metrics.items(),
    ):
        with col:
            st.metric(
                net_metric_label(metric),
                value,
            )


def render_allocation_table(
    weights: pd.Series,
) -> None:
    """
    Render portfolio allocation table.
    """

    df = (
        weights.rename("Weight")
        .reset_index()
        .rename(columns={"index": "Asset"})
    )

    df["Weight (%)"] = df["Weight"] * 100

    st.dataframe(
        df[["Asset", "Weight (%)"]],
        use_container_width=True,
    )
