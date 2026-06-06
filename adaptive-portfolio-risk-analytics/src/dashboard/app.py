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
    compare_risk_contributions,
    risk_contribution_table,
)
from src.backtesting import RollingBacktester
from src.backtesting.transaction_costs import TransactionCostModel
from src.benchmarks import (
    BenchmarkFactory,
    build_performance_comparison_table,
    compute_relative_performance,
    run_strategy_comparison,
)

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
    plot_drawdown_curves,
    plot_drawdowns,
    plot_cost_adjusted_comparison,
    plot_equity_curve,
    plot_final_value_comparison,
    plot_metric_comparison,
    plot_performance_curves,
    plot_rebalance_events,
    plot_hrp_herc_risk_comparison,
    plot_relative_performance,
    plot_risk_contribution_bar,
    plot_transaction_costs,
    plot_turnover_series,
    plot_weight_bar,
    plot_weight_vs_risk_contribution,
    plot_weight_pie,
)

from src.data_pipeline import (
    DataPreprocessor,
    YahooFinanceProvider,
)

from src.optimization import (
    EqualWeightAllocator,
    HERCAllocator,
    HRPAllocator,
    InverseVolatilityAllocator,
)


# ============================================================
# HELPERS
# ============================================================


def get_allocator(strategy_name: str):
    return BenchmarkFactory.get_allocator(strategy_name)


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
        "HERC",
    ],
)

comparison_strategies = st.sidebar.multiselect(
    "Benchmark Comparison Strategies",
    [
        "Equal Weight",
        "Inverse Volatility",
        "HRP",
        "HERC",
    ],
    default=[
        "Equal Weight",
        "Inverse Volatility",
        "HRP",
        "HERC",
    ],
)

benchmark_strategy = st.sidebar.selectbox(
    "Benchmark Strategy",
    [
        "Equal Weight",
        "Inverse Volatility",
        "HRP",
        "HERC",
    ],
    index=0,
)

rebalance_mode = st.sidebar.selectbox(
    "Rebalance Mode",
    [
        "calendar",
        "threshold",
        "calendar_or_threshold",
    ],
    index=0,
)

threshold = st.sidebar.slider(
    "Threshold",
    min_value=0.01,
    max_value=0.20,
    value=0.05,
    step=0.01,
)

base_bps = st.sidebar.number_input(
    "Base Cost (bps)",
    min_value=0.0,
    max_value=100.0,
    value=10.0,
    step=1.0,
)

slippage_bps = st.sidebar.number_input(
    "Slippage (bps)",
    min_value=0.0,
    max_value=100.0,
    value=5.0,
    step=1.0,
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

        prices_df, data_quality_summary = DataPreprocessor.handle_missing_values(
            market_data.prices_df
        )

        with st.expander("Data Quality Report"):
            quality_col1, quality_col2, quality_col3 = st.columns(3)

            quality_col1.metric(
                "Assets requested",
                data_quality_summary.total_assets_requested,
            )
            quality_col2.metric(
                "Assets retained",
                data_quality_summary.assets_retained,
            )
            quality_col3.metric(
                "Assets dropped",
                data_quality_summary.assets_dropped,
            )

            missing_col1, missing_col2 = st.columns(2)
            missing_col1.metric(
                "Missing observations before cleaning",
                data_quality_summary.missing_before,
            )
            missing_col2.metric(
                "Missing observations after cleaning",
                data_quality_summary.missing_after,
            )

            st.write(
                "Cleaning method:",
                data_quality_summary.cleaning_method,
            )

            if data_quality_summary.dropped_asset_names:
                st.write(
                    "Dropped assets:",
                    ", ".join(data_quality_summary.dropped_asset_names),
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
        transaction_cost_model = TransactionCostModel(
            base_bps=base_bps,
            slippage_bps=slippage_bps,
        )

        backtester = RollingBacktester(
            allocator=allocator,
            train_window=252,
            rebalance_frequency="M",
            rebalance_mode=rebalance_mode,
            threshold=threshold,
            transaction_cost_model=transaction_cost_model,
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
        gross_portfolio_value = (
            backtest_results[
                "gross_portfolio_values"
            ]
        )
        rebalance_log_df = backtest_results["rebalance_log"]
        turnover_summary = backtest_results["turnover_summary"]
        rebalance_summary = backtest_results["rebalance_summary"]
        cost_drag_summary = backtest_results["cost_drag_summary"]
        turnover_series = (
            pd.Series(
                rebalance_log_df["turnover"].values,
                index=pd.to_datetime(rebalance_log_df["rebalance_date"]),
                name="turnover",
            )
            if not rebalance_log_df.empty
            else pd.Series(dtype=float, name="turnover")
        )

        risk_contribution_df = risk_contribution_table(
            weights,
            covariance_matrix_df,
        )

        hrp_herc_risk_comparison_df = None
        if strategy in {"HRP", "HERC"}:
            hrp_weights = (
                HRPAllocator()
                .fit(
                    returns_df,
                    cov_matrix=covariance_matrix_df,
                    linkage_matrix=linkage_matrix,
                )
                .get_weights()
            )
            herc_weights = (
                HERCAllocator()
                .fit(
                    returns_df,
                    cov_matrix=covariance_matrix_df,
                    linkage_matrix=linkage_matrix,
                )
                .get_weights()
            )
            hrp_herc_risk_comparison_df = compare_risk_contributions(
                hrp_weights,
                herc_weights,
                covariance_matrix_df,
            )

        benchmark_strategy_names = list(
            dict.fromkeys(
                comparison_strategies + [benchmark_strategy]
            )
        )
        strategy_results = run_strategy_comparison(
            returns_df,
            strategy_names=benchmark_strategy_names,
            covariance_method="sample",
            train_window=252,
            rebalance_frequency="M",
            initial_capital=1_000_000.0,
            rebalance_mode=rebalance_mode,
            threshold=threshold,
            transaction_cost_model=transaction_cost_model,
        )
        performance_comparison_df = build_performance_comparison_table(strategy_results)
        relative_performance_df = compute_relative_performance(
            performance_comparison_df,
            benchmark_name=benchmark_strategy,
        )
        growth_curves = {
            strategy_name: result["portfolio_values"]
            for strategy_name, result in strategy_results.items()
        }
        drawdown_curves = {
            strategy_name: result["drawdown"]
            for strategy_name, result in strategy_results.items()
        }
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
        # RISK CONTRIBUTION
        # ----------------------------------------------------

        st.header("Risk Contribution Analysis")

        risk_col1, risk_col2 = st.columns(2)

        with risk_col1:
            st.plotly_chart(
                plot_risk_contribution_bar(risk_contribution_df),
                use_container_width=True,
            )

        with risk_col2:
            st.plotly_chart(
                plot_weight_vs_risk_contribution(risk_contribution_df),
                use_container_width=True,
            )

        st.dataframe(
            risk_contribution_df,
            use_container_width=True,
        )

        if hrp_herc_risk_comparison_df is not None:
            st.subheader("HRP vs HERC Risk Contribution Comparison")

            st.plotly_chart(
                plot_hrp_herc_risk_comparison(hrp_herc_risk_comparison_df),
                use_container_width=True,
            )
            st.dataframe(
                hrp_herc_risk_comparison_df,
                use_container_width=True,
            )

        # ----------------------------------------------------
        # BENCHMARK COMPARISON
        # ----------------------------------------------------

        st.header("Benchmark Comparison")

        comparison_metric = st.selectbox(
            "Comparison Metric",
            [
                "cagr",
                "sharpe",
                "sortino",
                "volatility",
                "max_drawdown",
                "calmar",
            ],
            index=1,
        )
        relative_metric = st.selectbox(
            "Relative Metric",
            [
                "excess_cagr",
                "excess_sharpe",
                "drawdown_difference",
                "volatility_difference",
                "final_value_difference",
            ],
            index=0,
        )

        st.subheader("Performance Comparison Table")
        st.dataframe(
            performance_comparison_df,
            use_container_width=True,
        )

        benchmark_col1, benchmark_col2 = st.columns(2)

        with benchmark_col1:
            st.plotly_chart(
                plot_performance_curves(growth_curves),
                use_container_width=True,
            )

        with benchmark_col2:
            st.plotly_chart(
                plot_drawdown_curves(drawdown_curves),
                use_container_width=True,
            )

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.plotly_chart(
                plot_metric_comparison(
                    performance_comparison_df,
                    comparison_metric,
                ),
                use_container_width=True,
            )

        with metric_col2:
            st.plotly_chart(
                plot_final_value_comparison(performance_comparison_df),
                use_container_width=True,
            )

        st.subheader(f"Relative Performance vs {benchmark_strategy}")

        st.plotly_chart(
            plot_relative_performance(
                relative_performance_df,
                relative_metric,
            ),
            use_container_width=True,
        )

        st.dataframe(
            relative_performance_df,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # TRADING ACTIVITY & COSTS
        # ----------------------------------------------------

        st.header("Trading Activity & Costs")

        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        summary_col1.metric("Total Turnover", f"{turnover_summary['total_turnover']:.2f}")
        summary_col2.metric("Average Turnover", f"{turnover_summary['average_turnover']:.2f}")
        summary_col3.metric(
            "Total Transaction Cost",
            f"{rebalance_summary['total_transaction_cost']:.2f}",
        )
        summary_col4.metric(
            "Number of Rebalances",
            str(rebalance_summary["total_rebalances"]),
        )

        cost_col1, cost_col2 = st.columns(2)
        cost_col1.metric(
            "Gross Final Value",
            f"{cost_drag_summary['gross_final_value']:.2f}",
        )
        cost_col2.metric(
            "Cost Drag",
            f"{cost_drag_summary['cost_drag']:.2f} ({cost_drag_summary['cost_drag_pct']:.2%})",
        )

        trade_chart_col1, trade_chart_col2 = st.columns(2)

        with trade_chart_col1:
            if not turnover_series.empty:
                st.plotly_chart(
                    plot_turnover_series(turnover_series),
                    use_container_width=True,
                )
            else:
                st.info("No turnover events recorded.")

        with trade_chart_col2:
            if not rebalance_log_df.empty:
                st.plotly_chart(
                    plot_transaction_costs(rebalance_log_df),
                    use_container_width=True,
                )
            else:
                st.info("No transaction costs recorded.")

        trade_chart_col3, trade_chart_col4 = st.columns(2)

        with trade_chart_col3:
            st.plotly_chart(
                plot_rebalance_events(portfolio_value, rebalance_log_df),
                use_container_width=True,
            )

        with trade_chart_col4:
            st.plotly_chart(
                plot_cost_adjusted_comparison(
                    gross_portfolio_value,
                    portfolio_value,
                ),
                use_container_width=True,
            )

        st.subheader("Rebalance Log")
        st.dataframe(
            rebalance_log_df,
            use_container_width=True,
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
