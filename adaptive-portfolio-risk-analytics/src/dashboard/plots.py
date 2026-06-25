"""Plotting utilities for the Streamlit dashboard."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

from src.dashboard.modes import net_metric_label

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
        title="Net Portfolio Growth Comparison",
        xaxis_title="Date",
        yaxis_title="Net Portfolio Value",
        template="plotly_white",
        hovermode="x unified",
    )

    return fig


def plot_equity_curve(
    portfolio_values: pd.Series,
    title: str = "Net Equity Curve",
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
        yaxis_title="Net Portfolio Value",
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

    fig, ax = plt.subplots(figsize=(12, 6))

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


def plot_risk_contribution_bar(
    risk_contribution_df: pd.DataFrame,
    title: str = "Percentage Risk Contribution",
) -> go.Figure:
    """Plot percentage risk contribution by asset."""
    fig = go.Figure(
        go.Bar(
            x=risk_contribution_df["Asset"].tolist(),
            y=risk_contribution_df["Percentage Risk Contribution"].tolist(),
            name="Percentage Risk Contribution",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Asset",
        yaxis_title="Percentage Risk Contribution",
        template="plotly_white",
    )

    return fig


def plot_weight_vs_risk_contribution(
    risk_contribution_df: pd.DataFrame,
    title: str = "Weight vs Risk Contribution",
) -> go.Figure:
    """Plot grouped capital weights and risk contributions."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=risk_contribution_df["Asset"].tolist(),
            y=risk_contribution_df["Weight"].tolist(),
            name="Weight",
        )
    )
    fig.add_trace(
        go.Bar(
            x=risk_contribution_df["Asset"].tolist(),
            y=risk_contribution_df["Percentage Risk Contribution"].tolist(),
            name="Percentage Risk Contribution",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Asset",
        yaxis_title="Value",
        barmode="group",
        template="plotly_white",
    )

    return fig


def plot_hrp_herc_risk_comparison(
    comparison_df: pd.DataFrame,
    title: str = "HRP vs HERC Risk Contribution Comparison",
) -> go.Figure:
    """Plot grouped HRP and HERC percentage risk contributions."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=comparison_df["Asset"].tolist(),
            y=comparison_df["HRP % Risk Contribution"].tolist(),
            name="HRP % Risk Contribution",
        )
    )
    fig.add_trace(
        go.Bar(
            x=comparison_df["Asset"].tolist(),
            y=comparison_df["HERC % Risk Contribution"].tolist(),
            name="HERC % Risk Contribution",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Asset",
        yaxis_title="Percentage Risk Contribution",
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


def plot_metric_comparison(
    performance_comparison_df: pd.DataFrame,
    metric_name: str,
) -> go.Figure:
    """Plot a selected performance metric across strategies."""
    if metric_name not in performance_comparison_df.columns:
        raise ValueError(f"metric_name '{metric_name}' not found in performance comparison table")

    fig = go.Figure(
        go.Bar(
            x=performance_comparison_df.index.tolist(),
            y=performance_comparison_df[metric_name].tolist(),
        )
    )

    fig.update_layout(
        title=f"{net_metric_label(metric_name)} Comparison",
        xaxis_title="Strategy",
        yaxis_title=net_metric_label(metric_name),
        template="plotly_white",
    )

    return fig


def plot_relative_performance(
    relative_performance_df: pd.DataFrame,
    metric_name: str,
) -> go.Figure:
    """Plot a selected relative performance metric versus the benchmark."""
    if metric_name not in relative_performance_df.columns:
        raise ValueError(f"metric_name '{metric_name}' not found in relative performance table")

    fig = go.Figure(
        go.Bar(
            x=relative_performance_df["strategy"].tolist(),
            y=relative_performance_df[metric_name].tolist(),
        )
    )

    fig.update_layout(
        title=f"{net_metric_label(metric_name)} vs Benchmark",
        xaxis_title="Strategy",
        yaxis_title=net_metric_label(metric_name),
        template="plotly_white",
    )

    return fig


def plot_final_value_comparison(
    performance_comparison_df: pd.DataFrame,
) -> go.Figure:
    """Plot final portfolio values across strategies."""
    if "final_value" not in performance_comparison_df.columns:
        raise ValueError("performance comparison table must contain 'final_value'")

    fig = go.Figure(
        go.Bar(
            x=performance_comparison_df.index.tolist(),
            y=performance_comparison_df["final_value"].tolist(),
        )
    )

    fig.update_layout(
        title="Net Final Value Comparison",
        xaxis_title="Strategy",
        yaxis_title="Net Final Portfolio Value",
        template="plotly_white",
    )

    return fig


def plot_turnover_series(
    turnover_series: pd.Series,
) -> go.Figure:
    """Plot turnover across rebalance dates."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=turnover_series.index,
            y=turnover_series.values,
            name="Turnover",
        )
    )
    fig.update_layout(
        title="Turnover by Rebalance Date",
        xaxis_title="Rebalance Date",
        yaxis_title="Turnover",
        template="plotly_white",
    )
    return fig


def plot_transaction_costs(
    rebalance_log_df: pd.DataFrame,
) -> go.Figure:
    """Plot transaction costs across rebalance dates."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=rebalance_log_df["rebalance_date"],
            y=rebalance_log_df["transaction_cost"],
            name="Transaction Cost",
        )
    )
    fig.update_layout(
        title="Transaction Costs by Rebalance Date",
        xaxis_title="Rebalance Date",
        yaxis_title="Transaction Cost",
        template="plotly_white",
    )
    return fig


def plot_rebalance_events(
    portfolio_values: pd.Series,
    rebalance_log_df: pd.DataFrame,
) -> go.Figure:
    """Plot portfolio value with rebalance event markers."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=portfolio_values.index,
            y=portfolio_values.values,
            mode="lines",
            name="Net Portfolio Value",
        )
    )

    if not rebalance_log_df.empty:
        aligned_values = portfolio_values.reindex(
            pd.to_datetime(rebalance_log_df["rebalance_date"]),
            method="nearest",
        )
        fig.add_trace(
            go.Scatter(
                x=rebalance_log_df["rebalance_date"],
                y=aligned_values.values,
                mode="markers",
                marker=dict(size=9, color="crimson", symbol="diamond"),
                name="Rebalance Event",
                text=rebalance_log_df["rebalance_reason"],
            )
        )

    fig.update_layout(
        title="Net Portfolio Value with Rebalance Events",
        xaxis_title="Date",
        yaxis_title="Net Portfolio Value",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_cost_adjusted_comparison(
    gross_values: pd.Series,
    net_values: pd.Series,
) -> go.Figure:
    """Plot gross and net portfolio values to show cost drag."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=gross_values.index,
            y=gross_values.values,
            mode="lines",
            name="Gross Portfolio Value",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=net_values.index,
            y=net_values.values,
            mode="lines",
            name="Net Portfolio Value",
        )
    )
    fig.update_layout(
        title="Gross vs Net Portfolio Value",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_exposure_series(
    exposure_series: pd.Series,
) -> go.Figure:
    """Plot risky-asset exposure through time."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=exposure_series.index,
            y=exposure_series.values,
            mode="lines",
            name="Risky Exposure",
            fill="tozeroy",
        )
    )
    fig.update_layout(
        title="Risky Portfolio Exposure",
        xaxis_title="Date",
        yaxis_title="Exposure",
        yaxis=dict(range=[0.0, 1.05]),
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_regime_series(
    regime_series: pd.Series,
    title: str = "Volatility Regime Timeline",
) -> go.Figure:
    """Plot categorical regime states over time with an explicit Unknown warm-up."""
    normalized = regime_series.astype(str).str.lower()
    observed = set(normalized.dropna().unique())
    preferred_order = [
        "unknown",
        "risk-on",
        "calm",
        "normal",
        "stress",
        "risk-off",
        "crisis",
    ]
    regime_order = [regime for regime in preferred_order if regime in observed]
    regime_order += sorted(observed - set(regime_order))
    encoded = normalized.map({name: idx for idx, name in enumerate(regime_order)})
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=regime_series.index,
            y=encoded.values,
            mode="lines",
            line=dict(shape="hv"),
            name="Regime",
            text=regime_series.astype(str).values,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Regime",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(regime_order))),
            ticktext=[name.title() for name in regime_order],
        ),
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_hmm_state_probabilities(
    probabilities: pd.DataFrame,
    title: str = "HMM State Probabilities",
) -> go.Figure:
    """Plot HMM state probabilities as a compact multiseries time chart."""
    fig = go.Figure()
    for column in probabilities.columns:
        fig.add_trace(
            go.Scatter(
                x=probabilities.index,
                y=probabilities[column],
                mode="lines",
                name=str(column).replace("_", " ").title(),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Probability",
        yaxis=dict(range=[0.0, 1.0]),
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_base_vs_vol_targeted_growth(
    base_values: pd.Series,
    targeted_values: pd.Series,
) -> go.Figure:
    """Plot base growth versus volatility-targeted growth."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=base_values.index,
            y=base_values.values,
            mode="lines",
            name="Base Strategy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=targeted_values.index,
            y=targeted_values.values,
            mode="lines",
            name="Volatility Targeted",
        )
    )
    fig.update_layout(
        title="Base vs Volatility-Targeted Growth",
        xaxis_title="Date",
        yaxis_title="Growth",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_realized_vs_target_vol(
    realized_vol: pd.Series,
    target_vol: pd.Series,
) -> go.Figure:
    """Plot realized volatility against target volatility."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=realized_vol.index,
            y=realized_vol.values,
            mode="lines",
            name="Realized Volatility",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=target_vol.index,
            y=target_vol.values,
            mode="lines",
            name="Target Volatility",
        )
    )
    fig.update_layout(
        title="Realized vs Target Volatility",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_defensive_allocation(
    exposure_series: pd.Series,
) -> go.Figure:
    """Plot defensive-sleeve allocation through time."""
    defensive_allocation = 1.0 - exposure_series
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=defensive_allocation.index,
            y=defensive_allocation.values,
            mode="lines",
            fill="tozeroy",
            name="Defensive Allocation",
        )
    )
    fig.update_layout(
        title="Defensive Sleeve Allocation",
        xaxis_title="Date",
        yaxis_title="Allocation",
        yaxis=dict(range=[0.0, 1.05]),
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_experiment_metric_by_parameter(
    experiment_results_df: pd.DataFrame,
    parameter,
    metric,
) -> go.Figure:
    """Plot average experiment metric by parameter value."""
    successful = (
        experiment_results_df[experiment_results_df["status"] == "success"]
        if "status" in experiment_results_df.columns
        else experiment_results_df
    )
    grouped = successful.groupby(parameter, dropna=False)[metric].mean().reset_index()
    grouped[parameter] = grouped[parameter].astype(str)

    fig = go.Figure(
        go.Bar(
            x=grouped[parameter].tolist(),
            y=grouped[metric].tolist(),
        )
    )
    fig.update_layout(
        title=f"Average {metric.replace('_', ' ').title()} by {parameter.replace('_', ' ').title()}",
        xaxis_title=parameter.replace("_", " ").title(),
        yaxis_title=metric.replace("_", " ").title(),
        template="plotly_white",
    )
    return fig


def plot_top_experiments(
    experiment_results_df: pd.DataFrame,
    metric,
) -> go.Figure:
    """Plot the top experiment configurations by the chosen metric."""
    successful = (
        experiment_results_df[experiment_results_df["status"] == "success"]
        if "status" in experiment_results_df.columns
        else experiment_results_df
    )
    ranked = successful.sort_values(by=metric, ascending=False).head(10).copy()
    ranked["label"] = (
        ranked["strategy"].astype(str)
        + " | "
        + ranked["covariance_method"].astype(str)
        + " | "
        + ranked["rebalance_mode"].astype(str)
    )

    fig = go.Figure(
        go.Bar(
            x=ranked["label"].tolist(),
            y=ranked[metric].tolist(),
        )
    )
    fig.update_layout(
        title=f"Top Experiment Configurations by {metric.replace('_', ' ').title()}",
        xaxis_title="Configuration",
        yaxis_title=metric.replace("_", " ").title(),
        template="plotly_white",
    )
    return fig


def plot_sensitivity_heatmap(
    experiment_results_df: pd.DataFrame,
    x_param,
    y_param,
    metric,
) -> go.Figure:
    """Plot a sensitivity heatmap for two experiment parameters."""
    successful = (
        experiment_results_df[experiment_results_df["status"] == "success"]
        if "status" in experiment_results_df.columns
        else experiment_results_df
    )
    pivot = pd.pivot_table(
        successful,
        index=y_param,
        columns=x_param,
        values=metric,
        aggfunc="mean",
    )

    fig = px.imshow(
        pivot,
        color_continuous_scale="Blues",
        aspect="auto",
        title=f"{metric.replace('_', ' ').title()} Sensitivity",
    )
    fig.update_layout(
        xaxis_title=x_param.replace("_", " ").title(),
        yaxis_title=y_param.replace("_", " ").title(),
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
                "pain_index",
            ]
        ):
            formatted[key] = f"{value:.2%}"

        else:
            formatted[key] = f"{value:.2f}"

    return formatted
