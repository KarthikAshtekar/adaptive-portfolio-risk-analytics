from __future__ import annotations

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[2]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from datetime import date

import pandas as pd
import streamlit as st

from src.analytics import (
    PerformanceAnalytics,
    RiskAnalytics,
)
from src.backtesting import RollingBacktester

from src.clustering import (
    compute_linkage_matrix,
)

from src.covariance import (
    compute_covariance_matrix,
    compute_correlation_matrix,
    compute_distance_matrix,
)

from src.dashboard.components import (
    render_allocation_table,
    render_portfolio_summary,
)

from src.dashboard.plots import (
    format_metric_cards,
    plot_correlation_heatmap,
    plot_dendrogram,
    plot_drawdowns,
    plot_equity_curve,
    plot_weight_bar,
    plot_weight_pie,
)

from src.data_pipeline import (
    DataPreprocessor,
    YahooFinanceProvider,
)

from src.optimization import (
    EqualWeightAllocator,
    HRPAllocator,
    InverseVolatilityAllocator,
)


# ============================================================
# HELPERS
# ============================================================


def get_allocator(strategy_name: str):

    allocators = {
        "Equal Weight": EqualWeightAllocator(),
        "Inverse Volatility": InverseVolatilityAllocator(),
        "HRP": HRPAllocator(),
    }

    return allocators[strategy_name]


# ============================================================
# PAGE
# ============================================================


st.set_page_config(
    page_title="Adaptive Portfolio Risk Analytics",
    layout="wide",
)

st.title("Adaptive Portfolio Risk Analytics")

# ============================================================
# SIDEBAR
# ============================================================


st.sidebar.header("Portfolio Inputs")

symbols_input = st.sidebar.text_input(
    "Tickers",
    "HDFCBANK.NS,TCS.NS,GOLDBEES.NS",
)

start_date = st.sidebar.date_input(
    "Start Date",
    date(2020, 1, 1),
)

end_date = st.sidebar.date_input(
    "End Date",
    date.today(),
)

strategy = st.sidebar.selectbox(
    "Strategy",
    [
        "Equal Weight",
        "Inverse Volatility",
        "HRP",
    ],
)

run_button = st.sidebar.button(
    "Run Portfolio Analysis"
)

# ============================================================
# EXECUTION
# ============================================================


if run_button:

    try:

        symbols = [
            s.strip()
            for s in symbols_input.split(",")
            if s.strip()
        ]

        provider = YahooFinanceProvider()

        market_data = provider.get_market_data(
            symbols=symbols,
            start_date=str(start_date),
            end_date=str(end_date),
        )

        prices_df = DataPreprocessor.handle_missing_values(
            market_data.prices_df
        )

        returns_df = (
            DataPreprocessor
            .build_returns_risk_outputs(
                prices_df
            )
            .returns_df
        )

        # ----------------------------------------------------
        # Covariance & Correlation
        # ----------------------------------------------------

        covariance_matrix_df = (
            compute_covariance_matrix(
                returns_df
            )
        )

        correlation_matrix_df = (
            compute_correlation_matrix(
                returns_df
            )
        )

        distance_matrix_df = (
            compute_distance_matrix(
                correlation_matrix_df
            )
        )

        linkage_matrix = (
            compute_linkage_matrix(
                distance_matrix_df
            )
        )
        # ----------------------------------------------------
        # Portfolio Allocation
        # ----------------------------------------------------

        allocator = get_allocator(
            strategy
        )

        backtester = RollingBacktester(
            allocator=allocator,
            train_window=252,
            rebalance_frequency="M",
        )

        backtest_results = backtester.run(
            returns_df
        )

        weights = (
            backtest_results["weights_history"]
            .iloc[-1]
        )


        portfolio_returns = (
            backtest_results[
                "portfolio_returns"
            ]
        )

        portfolio_value = (
            backtest_results[
                "portfolio_values"
            ]
        )
        # ----------------------------------------------------
        # Analytics
        # ----------------------------------------------------

        metrics = (
            PerformanceAnalytics
            .summary_table(
                portfolio_returns
            )
        )

        formatted_metrics = (
            format_metric_cards(
                metrics
            )
        )

        # ----------------------------------------------------
        # KPI CARDS
        # ----------------------------------------------------

        st.header("Portfolio Metrics")

        render_portfolio_summary(
            formatted_metrics
        )

        # ----------------------------------------------------
        # WEIGHTS
        # ----------------------------------------------------

        st.header("Portfolio Allocation")

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                plot_weight_bar(weights),
                use_container_width=True,
            )

        with col2:
            st.plotly_chart(
                plot_weight_pie(weights),
                use_container_width=True,
            )

        render_allocation_table(
            weights
        )

        # ----------------------------------------------------
        # PERFORMANCE
        # ----------------------------------------------------

        st.header("Performance")

        st.plotly_chart(
            plot_equity_curve(
                portfolio_value
            ),
            use_container_width=True,
        )

        drawdown_series = (
            RiskAnalytics
            .drawdown_series(
                portfolio_returns
            )
        )

        st.plotly_chart(
            plot_drawdowns(
                drawdown_series
            ),
            use_container_width=True,
        )

        # ----------------------------------------------------
        # CORRELATION
        # ----------------------------------------------------

        st.header("Correlation Structure")

        st.plotly_chart(
            plot_correlation_heatmap(
                correlation_matrix_df
            ),
            use_container_width=True,
        )

        # ----------------------------------------------------
        # DENDROGRAM
        # ----------------------------------------------------

        st.header("Hierarchical Clustering")

        dendrogram_fig = plot_dendrogram(
            linkage_matrix,
            labels=list(
                returns_df.columns
            ),
        )

        st.pyplot(
            dendrogram_fig
        )

    except Exception as exc:

        st.error(
            f"Error: {exc}"
        )

else:

    st.info(
        "Configure inputs and click 'Run Portfolio Analysis'."
    )