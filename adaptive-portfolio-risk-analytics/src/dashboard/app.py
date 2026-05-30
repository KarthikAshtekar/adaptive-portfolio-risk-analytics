"""Streamlit dashboard for Phase 1 portfolio optimization and risk analytics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import streamlit as st

from src.backtesting import RollingBacktester
from src.clustering import HierarchicalClusterer
from src.covariance import SampleCovarianceEstimator
from src.dashboard.plots import (
    plot_correlation_heatmap,
    plot_dendrogram,
    plot_drawdown_curves,
    plot_performance_curves,
    plot_weight_bar,
)
from src.data_pipeline import DataPreprocessor, YFinanceIngester
from src.optimization import (
    EqualWeightAllocator,
    HRPAllocator,
    InverseVolatilityAllocator,
    MeanVarianceAllocator,
)


@dataclass
class DashboardInputs:
    symbols: list[str]
    start_date: str
    end_date: str
    strategies: list[str]
    rebalance_frequency: str
    train_window: int


def _strategy_factory(strategy_name: str):
    mapping = {
        "Equal Weight": EqualWeightAllocator,
        "Mean Variance": MeanVarianceAllocator,
        "Inverse Volatility": InverseVolatilityAllocator,
        "HRP": HRPAllocator,
    }
    if strategy_name not in mapping:
        raise ValueError(f"unsupported strategy: {strategy_name}")
    return mapping[strategy_name]()


@st.cache_data(show_spinner=False)
def _download_prices(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    ingester = YFinanceIngester()
    return ingester.fetch(symbols, start_date=start_date, end_date=end_date)


def _run_backtests(returns: pd.DataFrame, inputs: DashboardInputs) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for strategy in inputs.strategies:
        allocator = _strategy_factory(strategy)
        bt = RollingBacktester(
            allocator=allocator,
            train_window=inputs.train_window,
            rebalance_frequency=inputs.rebalance_frequency,
        )
        results[strategy] = bt.run(returns)
    return results


def _parse_symbols(raw_symbols: str) -> list[str]:
    symbols = [s.strip().upper() for s in raw_symbols.split(",") if s.strip()]
    if len(symbols) < 2:
        raise ValueError("please provide at least 2 assets")
    return symbols


def main() -> None:
    st.set_page_config(page_title="Portfolio Optimization Dashboard", layout="wide")
    st.title("Phase 1 Portfolio Optimization and Risk Analytics")

    with st.sidebar:
        st.header("Inputs")
        symbol_text = st.text_input("Asset universe (comma-separated)", "SPY,QQQ,TLT,GLD,IEF")

        col_a, col_b = st.columns(2)
        with col_a:
            start = st.date_input("Start date", value=date(2018, 1, 1))
        with col_b:
            end = st.date_input("End date", value=date.today())

        strategies = st.multiselect(
            "Strategies",
            ["Equal Weight", "Mean Variance", "Inverse Volatility", "HRP"],
            default=["Equal Weight", "Mean Variance", "Inverse Volatility", "HRP"],
        )

        rebalance = st.selectbox("Rebalance frequency", ["D", "W", "M", "Q"], index=2)
        train_window = st.slider(
            "Training window (days)",
            min_value=63,
            max_value=504,
            value=252,
            step=21,
        )

        run = st.button("Run Analysis", type="primary")

    if not run:
        st.info("Set your inputs and click Run Analysis.")
        return

    try:
        symbols = _parse_symbols(symbol_text)
        if not strategies:
            st.warning("Select at least one strategy.")
            return

        inputs = DashboardInputs(
            symbols=symbols,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            strategies=strategies,
            rebalance_frequency=rebalance,
            train_window=train_window,
        )

        prices = _download_prices(inputs.symbols, inputs.start_date, inputs.end_date)
        prices = DataPreprocessor.handle_missing_values(prices, method="forward_fill")
        returns = DataPreprocessor.calculate_returns(prices, method="simple")

        if len(returns) <= inputs.train_window + 1:
            st.error("Not enough return history for selected training window.")
            return

        cov_estimator = SampleCovarianceEstimator().fit(returns)
        cov_matrix = cov_estimator.get_covariance()
        corr = returns.corr()

        clusterer = HierarchicalClusterer(linkage_method="single").fit(returns)
        linkage_matrix = clusterer.linkage_matrix

        backtest_results = _run_backtests(returns, inputs)

        # Metrics table
        summary_rows = []
        for name, result in backtest_results.items():
            perf = result["performance_metrics"]
            summary_rows.append(
                {
                    "Strategy": name,
                    "Sharpe": perf["sharpe"],
                    "Sortino": perf["sortino"],
                    "CAGR": perf["cagr"],
                    "Volatility": perf["volatility"],
                    "Max Drawdown": perf["max_drawdown"],
                    "Final Value": perf["final_value"],
                }
            )
        summary_df = pd.DataFrame(summary_rows).set_index("Strategy")

        st.subheader("Strategy Metrics")
        st.dataframe(summary_df.style.format("{:.4f}"), use_container_width=True)

        curves = {k: v["portfolio_values"] for k, v in backtest_results.items()}
        drawdowns = {k: v["drawdown"] for k, v in backtest_results.items()}

        st.subheader("Performance Curve")
        st.plotly_chart(plot_performance_curves(curves), use_container_width=True)

        st.subheader("Drawdown Curve")
        st.plotly_chart(plot_drawdown_curves(drawdowns), use_container_width=True)

        # Latest weights by strategy
        st.subheader("Latest Portfolio Weights")
        weight_cols = st.columns(len(backtest_results))
        for (name, result), col in zip(backtest_results.items(), weight_cols):
            with col:
                st.markdown(f"**{name}**")
                weights_history = result["weights_history"]
                if weights_history.empty:
                    st.warning("No rebalance points generated.")
                else:
                    latest = weights_history.iloc[-1]
                    st.plotly_chart(
                        plot_weight_bar(latest, title=f"{name} Weights"),
                        use_container_width=True,
                    )

        vis_col1, vis_col2 = st.columns(2)
        with vis_col1:
            st.subheader("Correlation Heatmap")
            st.plotly_chart(plot_correlation_heatmap(corr), use_container_width=True)
        with vis_col2:
            st.subheader("Dendrogram")
            if linkage_matrix is not None:
                st.plotly_chart(
                    plot_dendrogram(linkage_matrix, labels=list(returns.columns)),
                    use_container_width=True,
                )

        st.caption(
            f"Covariance matrix shape: {cov_matrix.shape} | Returns rows: {len(returns)}"
        )

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")


if __name__ == "__main__":
    main()
