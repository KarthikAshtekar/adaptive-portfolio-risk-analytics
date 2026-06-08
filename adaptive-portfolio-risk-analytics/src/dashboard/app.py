from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.analytics import (
    PerformanceAnalytics,
    RiskAnalytics,
    compare_risk_contributions,
    risk_contribution_table,
)
from src.backtesting import RollingBacktester, VolatilityTargetingConfig, apply_volatility_targeting
from src.backtesting.transaction_costs import TransactionCostModel
from src.benchmarks import (
    BenchmarkFactory,
    build_performance_comparison_table,
    compute_relative_performance,
    run_strategy_comparison,
)
from src.clustering import compute_linkage_matrix
from src.covariance import CovarianceFactory, compute_correlation_matrix, compute_distance_matrix
from src.dashboard.components import render_allocation_table, render_portfolio_summary
from src.dashboard.plots import (
    format_metric_cards,
    plot_base_vs_vol_targeted_growth,
    plot_correlation_heatmap,
    plot_cost_adjusted_comparison,
    plot_defensive_allocation,
    plot_dendrogram,
    plot_drawdown_curves,
    plot_drawdowns,
    plot_equity_curve,
    plot_experiment_metric_by_parameter,
    plot_exposure_series,
    plot_final_value_comparison,
    plot_hrp_herc_risk_comparison,
    plot_metric_comparison,
    plot_performance_curves,
    plot_rebalance_events,
    plot_realized_vs_target_vol,
    plot_regime_series,
    plot_relative_performance,
    plot_risk_contribution_bar,
    plot_sensitivity_heatmap,
    plot_top_experiments,
    plot_transaction_costs,
    plot_turnover_series,
    plot_weight_bar,
    plot_weight_pie,
    plot_weight_vs_risk_contribution,
)
from src.data_pipeline import DataPreprocessor, YahooFinanceProvider, get_defensive_asset_returns
from src.experiments import (
    ExperimentConfig,
    build_experiment_summary_table,
    build_top_n_table,
    compute_parameter_sensitivity,
    run_experiment_grid,
)
from src.optimization import HERCAllocator, HRPAllocator


PORTFOLIO_RESULT_KEY = "dashboard_portfolio_results"
SENSITIVITY_RESULT_KEY = "dashboard_sensitivity_results"
UI_MESSAGE_KEY = "dashboard_ui_message"

INDIAN_ASSET_UNIVERSE = {
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "SBIN.NS": "State Bank of India",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "AXISBANK.NS": "Axis Bank",
    "TCS.NS": "TCS",
    "INFY.NS": "Infosys",
    "WIPRO.NS": "Wipro",
    "HCLTECH.NS": "HCL Technologies",
    "TECHM.NS": "Tech Mahindra",
    "RELIANCE.NS": "Reliance Industries",
    "ONGC.NS": "ONGC",
    "NTPC.NS": "NTPC",
    "POWERGRID.NS": "Power Grid",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS": "ITC",
    "NESTLEIND.NS": "Nestle India",
    "TATACONSUM.NS": "Tata Consumer",
    "SUNPHARMA.NS": "Sun Pharma",
    "DRREDDY.NS": "Dr Reddy's",
    "CIPLA.NS": "Cipla",
    "TATAMOTORS.NS": "Tata Motors",
    "MARUTI.NS": "Maruti Suzuki",
    "M&M.NS": "Mahindra & Mahindra",
    "LT.NS": "Larsen & Toubro",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "ASIANPAINT.NS": "Asian Paints",
    "BHARTIARTL.NS": "Bharti Airtel",
    "GOLDBEES.NS": "GoldBeES",
    "SILVERBEES.NS": "SilverBeES",
}

DEFAULT_UNIVERSES = {
    "Core Diversified": [
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "TCS.NS",
        "INFY.NS",
        "RELIANCE.NS",
        "HINDUNILVR.NS",
        "ITC.NS",
        "SUNPHARMA.NS",
        "LT.NS",
        "BHARTIARTL.NS",
        "GOLDBEES.NS",
    ],
    "Banks + IT + Gold": [
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "KOTAKBANK.NS",
        "TCS.NS",
        "INFY.NS",
        "WIPRO.NS",
        "HCLTECH.NS",
        "GOLDBEES.NS",
    ],
    "Full Research Universe": list(INDIAN_ASSET_UNIVERSE.keys()),
}

COVARIANCE_METHOD_OPTIONS = [
    "sample",
    "ledoit_wolf",
    "ewma",
    "ewma_ledoit_wolf",
]

SENSITIVITY_OBJECTIVE_MAP = {
    "CAGR": "cagr",
    "Sharpe": "sharpe",
    "Sortino": "sortino",
    "Calmar": "calmar",
    "Max Drawdown": "max_drawdown",
    "Final Value": "final_value",
}


def ticker_label(ticker: str) -> str:
    return f"{ticker} — {INDIAN_ASSET_UNIVERSE[ticker]}"


def initialize_session_state() -> None:
    default_labels = [ticker_label(ticker) for ticker in DEFAULT_UNIVERSES["Core Diversified"]]
    defaults = {
        "ui_universe_preset": "Core Diversified",
        "ui_select_all_assets": True,
        "ui_selected_assets": default_labels,
        "_last_preset": "Core Diversified",
        "_last_select_all": True,
        "ui_manual_ticker_override": "",
        PORTFOLIO_RESULT_KEY: None,
        SENSITIVITY_RESULT_KEY: None,
        UI_MESSAGE_KEY: None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_allocator(strategy_name: str, covariance_method: str = "sample"):
    return BenchmarkFactory.get_allocator(strategy_name, covariance_method=covariance_method)


def parse_ticker_override(raw_value: str) -> list[str]:
    tickers = [ticker.strip().upper() for ticker in raw_value.split(",") if ticker.strip()]
    return list(dict.fromkeys(tickers))


def parse_float_list(raw_values: str) -> list[float]:
    try:
        values = [float(raw_value.strip()) for raw_value in raw_values.split(",") if raw_value.strip()]
    except ValueError as exc:
        raise ValueError("Sensitivity Thresholds must be comma-separated decimals like 0.03,0.05,0.10") from exc
    if not values:
        raise ValueError("At least one threshold value is required")
    return values


def update_selected_assets_for_preset(
    preset: str,
    select_all: bool,
) -> list[str]:
    all_labels = [ticker_label(ticker) for ticker in INDIAN_ASSET_UNIVERSE]
    current_selection = list(st.session_state.get("ui_selected_assets", []))
    if preset == "Custom":
        if select_all:
            return all_labels
        filtered = [label for label in current_selection if label in all_labels]
        return filtered or default_labels_for_preset("Core Diversified")

    preset_labels = default_labels_for_preset(preset)
    if select_all:
        return preset_labels
    filtered = [label for label in current_selection if label in preset_labels]
    return filtered or preset_labels


def default_labels_for_preset(preset: str) -> list[str]:
    return [ticker_label(ticker) for ticker in DEFAULT_UNIVERSES[preset]]


def selected_tickers_from_labels(labels: list[str]) -> list[str]:
    tickers = []
    for label in labels:
        ticker = label.split(" — ", 1)[0].strip().upper()
        if ticker in INDIAN_ASSET_UNIVERSE:
            tickers.append(ticker)
    return list(dict.fromkeys(tickers))


def show_message(level: str, message: str) -> None:
    if level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    else:
        st.info(message)


def validate_common_inputs(
    selected_tickers: list[str],
    start_dt: date,
    end_dt: date,
    exposure_floor: float,
    exposure_cap: float,
    defensive_sleeve: str,
) -> str | None:
    if len(selected_tickers) < 2:
        return "Select at least 2 risky assets."
    if start_dt >= end_dt:
        return "Start date must be before end date."
    if exposure_floor > exposure_cap:
        return "Exposure Floor must be less than or equal to Exposure Cap."
    if defensive_sleeve != "Synthetic Risk-Free" and defensive_sleeve in selected_tickers:
        return "Defensive sleeve must remain separate from the risky asset universe."
    return None


def validate_sensitivity_inputs(
    threshold_values: list[float],
    max_runs: int,
) -> str | None:
    if max_runs <= 0:
        return "Sensitivity Max Runs must be positive."
    if any(value <= 0 or value >= 1 for value in threshold_values):
        return "Sensitivity thresholds must be between 0 and 1."
    return None


def load_market_context(
    selected_tickers: list[str],
    start_dt: date,
    end_dt: date,
) -> dict[str, object]:
    provider = YahooFinanceProvider()
    market_data = provider.get_market_data(
        symbols=selected_tickers,
        start_date=str(start_dt),
        end_date=str(end_dt),
    )
    prices_df, data_quality_summary = DataPreprocessor.handle_missing_values(market_data.prices_df)
    returns_df = DataPreprocessor.build_returns_risk_outputs(prices_df).returns_df
    return {
        "prices_df": prices_df,
        "returns_df": returns_df,
        "data_quality_summary": data_quality_summary,
    }


def build_portfolio_results(
    *,
    selected_tickers: list[str],
    start_dt: date,
    end_dt: date,
    strategy: str,
    comparison_strategies: list[str],
    benchmark_strategy: str,
    covariance_method: str,
    rebalance_mode: str,
    threshold: float,
    base_bps: float,
    slippage_bps: float,
    enable_vol_targeting: bool,
    defensive_sleeve: str,
    synthetic_annual_rate: float,
    vol_target_mode: str,
    base_target_vol: float,
    realized_vol_window: int,
    regime_lookback_window: int,
    exposure_floor: float,
    exposure_cap: float,
    no_trade_band: float,
    initial_capital: float,
) -> dict[str, object]:
    market_context = load_market_context(selected_tickers, start_dt, end_dt)
    returns_df = market_context["returns_df"]

    covariance_matrix_df = CovarianceFactory.compute(returns_df, method=covariance_method)
    correlation_matrix_df = compute_correlation_matrix(returns_df)
    distance_matrix_df = compute_distance_matrix(correlation_matrix_df)
    linkage_matrix = compute_linkage_matrix(distance_matrix_df)

    allocator = get_allocator(strategy, covariance_method)
    transaction_cost_model = TransactionCostModel(base_bps=base_bps, slippage_bps=slippage_bps)
    backtester = RollingBacktester(
        allocator=allocator,
        train_window=252,
        rebalance_frequency="M",
        initial_capital=initial_capital,
        rebalance_mode=rebalance_mode,
        threshold=threshold,
        transaction_cost_model=transaction_cost_model,
    )
    backtest_results = backtester.run(returns_df)

    weights = backtest_results["weights_history"].iloc[-1]
    portfolio_returns = backtest_results["portfolio_returns"]
    portfolio_value = backtest_results["portfolio_values"]
    gross_portfolio_value = backtest_results["gross_portfolio_values"]
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

    risk_contribution_df = risk_contribution_table(weights, covariance_matrix_df)

    hrp_herc_risk_comparison_df = None
    if strategy in {"HRP", "HERC"}:
        hrp_weights = (
            HRPAllocator(covariance_method=covariance_method)
            .fit(returns_df, cov_matrix=covariance_matrix_df, linkage_matrix=linkage_matrix)
            .get_weights()
        )
        herc_weights = (
            HERCAllocator(covariance_method=covariance_method)
            .fit(returns_df, cov_matrix=covariance_matrix_df, linkage_matrix=linkage_matrix)
            .get_weights()
        )
        hrp_herc_risk_comparison_df = compare_risk_contributions(
            hrp_weights,
            herc_weights,
            covariance_matrix_df,
        )

    benchmark_strategy_names = list(dict.fromkeys(comparison_strategies + [benchmark_strategy]))
    strategy_results = run_strategy_comparison(
        returns_df,
        strategy_names=benchmark_strategy_names,
        covariance_method=covariance_method,
        train_window=252,
        rebalance_frequency="M",
        initial_capital=initial_capital,
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
        strategy_name: result["portfolio_values"] for strategy_name, result in strategy_results.items()
    }
    drawdown_curves = {
        strategy_name: result["drawdown"] for strategy_name, result in strategy_results.items()
    }

    vol_target_results = None
    defensive_metadata = None
    if enable_vol_targeting:
        preferred_ticker = None if defensive_sleeve == "Synthetic Risk-Free" else defensive_sleeve
        fallback_tickers = []
        if defensive_sleeve == "LIQUIDBEES.NS":
            fallback_tickers = ["LIQUIDETF.NS"]
        elif defensive_sleeve == "LIQUIDETF.NS":
            fallback_tickers = ["LIQUIDBEES.NS"]

        # Defensive sleeve stays outside the risky universe by design.
        defensive_returns, defensive_metadata = get_defensive_asset_returns(
            start_date=returns_df.index.min(),
            end_date=returns_df.index.max(),
            preferred_ticker=preferred_ticker,
            fallback_tickers=fallback_tickers,
            synthetic_annual_rate=synthetic_annual_rate,
        )
        vol_target_config = VolatilityTargetingConfig(
            realized_vol_window=realized_vol_window,
            regime_lookback_window=regime_lookback_window,
            base_target_vol=base_target_vol,
            calm_target_vol=0.12,
            normal_target_vol=base_target_vol,
            stress_target_vol=0.06,
            crisis_target_vol=0.03,
            exposure_floor=exposure_floor,
            exposure_cap=exposure_cap,
            no_trade_band=no_trade_band,
        )
        if vol_target_mode == "Fixed":
            vol_target_config = replace(
                vol_target_config,
                calm_target_vol=base_target_vol,
                normal_target_vol=base_target_vol,
                stress_target_vol=base_target_vol,
                crisis_target_vol=base_target_vol,
            )
        vol_target_results = apply_volatility_targeting(
            risky_returns=portfolio_returns,
            defensive_returns=defensive_returns,
            config=vol_target_config,
        )

    metrics = PerformanceAnalytics.summary_table(portfolio_returns)

    return {
        "selected_tickers": selected_tickers,
        "market_context": market_context,
        "covariance_method": covariance_method,
        "covariance_matrix_df": covariance_matrix_df,
        "correlation_matrix_df": correlation_matrix_df,
        "linkage_matrix": linkage_matrix,
        "weights": weights,
        "backtest_results": backtest_results,
        "portfolio_returns": portfolio_returns,
        "portfolio_value": portfolio_value,
        "gross_portfolio_value": gross_portfolio_value,
        "turnover_series": turnover_series,
        "turnover_summary": turnover_summary,
        "rebalance_summary": rebalance_summary,
        "cost_drag_summary": cost_drag_summary,
        "rebalance_log_df": rebalance_log_df,
        "risk_contribution_df": risk_contribution_df,
        "hrp_herc_risk_comparison_df": hrp_herc_risk_comparison_df,
        "performance_comparison_df": performance_comparison_df,
        "relative_performance_df": relative_performance_df,
        "growth_curves": growth_curves,
        "drawdown_curves": drawdown_curves,
        "metrics": metrics,
        "formatted_metrics": format_metric_cards(metrics),
        "vol_target_results": vol_target_results,
        "defensive_metadata": defensive_metadata,
    }


def build_sensitivity_results(
    *,
    selected_tickers: list[str],
    start_dt: date,
    end_dt: date,
    base_bps: float,
    slippage_bps: float,
    enable_vol_targeting: bool,
    defensive_sleeve: str,
    synthetic_annual_rate: float,
    base_target_vol: float,
    strategies: list[str],
    covariance_methods: list[str],
    rebalance_modes: list[str],
    thresholds_text: str,
    objective_label: str,
    max_runs: int,
    initial_capital: float,
) -> dict[str, object]:
    threshold_values = parse_float_list(thresholds_text)
    validation_error = validate_sensitivity_inputs(threshold_values, max_runs)
    if validation_error is not None:
        raise ValueError(validation_error)

    market_context = load_market_context(selected_tickers, start_dt, end_dt)
    returns_df = market_context["returns_df"]

    defensive_input = None
    if enable_vol_targeting:
        preferred_ticker = None if defensive_sleeve == "Synthetic Risk-Free" else defensive_sleeve
        fallback_tickers = []
        if defensive_sleeve == "LIQUIDBEES.NS":
            fallback_tickers = ["LIQUIDETF.NS"]
        elif defensive_sleeve == "LIQUIDETF.NS":
            fallback_tickers = ["LIQUIDBEES.NS"]
        defensive_returns, _ = get_defensive_asset_returns(
            start_date=returns_df.index.min(),
            end_date=returns_df.index.max(),
            preferred_ticker=preferred_ticker,
            fallback_tickers=fallback_tickers,
            synthetic_annual_rate=synthetic_annual_rate,
        )
        key = defensive_sleeve if defensive_sleeve != "Synthetic Risk-Free" else "Synthetic Risk-Free"
        defensive_input = {key: defensive_returns}

    # Sensitivity study is separated from one-off portfolio analysis so it can run independently.
    experiment_config = ExperimentConfig(
        experiment_name="dashboard_sensitivity_study",
        strategies=strategies or ["HRP", "HERC"],
        covariance_methods=covariance_methods or ["sample"],
        rebalance_modes=rebalance_modes or ["calendar"],
        thresholds=threshold_values,
        transaction_cost_bps=[base_bps],
        slippage_bps=[slippage_bps],
        enable_vol_targeting=[False, True] if enable_vol_targeting else [False],
        target_vols=[base_target_vol],
        defensive_assets=[defensive_sleeve],
        start_date=str(start_dt),
        end_date=str(end_dt),
        train_window=252,
        initial_capital=initial_capital,
    )
    objective_metric = SENSITIVITY_OBJECTIVE_MAP[objective_label]
    experiment_results_df = run_experiment_grid(
        returns_df=returns_df,
        config=experiment_config,
        defensive_returns=defensive_input,
        max_runs=max_runs,
    )

    return {
        "experiment_results_df": experiment_results_df,
        "objective_metric": objective_metric,
        "experiment_summary_df": build_experiment_summary_table(experiment_results_df),
        "top_experiments_df": build_top_n_table(
            experiment_results_df,
            metric=objective_metric,
            n=10,
        ),
        "parameter_sensitivity_df": compute_parameter_sensitivity(
            experiment_results_df,
            metric=objective_metric,
        ),
    }


def render_data_quality_report(data_quality_summary) -> None:
    with st.expander("Data Quality Report"):
        quality_col1, quality_col2, quality_col3 = st.columns(3)
        quality_col1.metric("Assets requested", data_quality_summary.total_assets_requested)
        quality_col2.metric("Assets retained", data_quality_summary.assets_retained)
        quality_col3.metric("Assets dropped", data_quality_summary.assets_dropped)

        missing_col1, missing_col2 = st.columns(2)
        missing_col1.metric(
            "Missing observations before cleaning",
            data_quality_summary.missing_before,
        )
        missing_col2.metric(
            "Missing observations after cleaning",
            data_quality_summary.missing_after,
        )

        st.write("Cleaning method:", data_quality_summary.cleaning_method)
        if data_quality_summary.dropped_asset_names:
            st.write("Dropped assets:", ", ".join(data_quality_summary.dropped_asset_names))


def render_dashboard_tabs(
    portfolio_payload: dict[str, object] | None,
    sensitivity_payload: dict[str, object] | None,
) -> None:
    tabs = st.tabs(
        [
            "Portfolio Overview",
            "Backtest Results",
            "Risk & Allocation",
            "Trading Activity",
            "Volatility Targeting",
            "Experiment Sensitivity",
        ]
    )

    if portfolio_payload is None:
        with tabs[0]:
            st.info("Run Portfolio Analysis to populate the dashboard.")
        with tabs[1]:
            st.info("Backtest results will appear here after a portfolio run.")
        with tabs[2]:
            st.info("Risk and allocation outputs will appear here after a portfolio run.")
        with tabs[3]:
            st.info("Trading activity diagnostics will appear here after a portfolio run.")
        with tabs[4]:
            st.info("Enable Volatility Targeting and run the portfolio analysis to see overlay results.")
        with tabs[5]:
            if sensitivity_payload is None:
                st.info("Run Sensitivity Study to populate experiment outputs.")
            else:
                render_sensitivity_content(sensitivity_payload)
        return

    data_quality_summary = portfolio_payload["market_context"]["data_quality_summary"]
    weights = portfolio_payload["weights"]
    portfolio_value = portfolio_payload["portfolio_value"]
    portfolio_returns = portfolio_payload["portfolio_returns"]
    performance_comparison_df = portfolio_payload["performance_comparison_df"]
    relative_performance_df = portfolio_payload["relative_performance_df"]
    drawdown_curves = portfolio_payload["drawdown_curves"]
    growth_curves = portfolio_payload["growth_curves"]
    risk_contribution_df = portfolio_payload["risk_contribution_df"]
    hrp_herc_risk_comparison_df = portfolio_payload["hrp_herc_risk_comparison_df"]
    correlation_matrix_df = portfolio_payload["correlation_matrix_df"]
    linkage_matrix = portfolio_payload["linkage_matrix"]
    turnover_summary = portfolio_payload["turnover_summary"]
    rebalance_summary = portfolio_payload["rebalance_summary"]
    cost_drag_summary = portfolio_payload["cost_drag_summary"]
    turnover_series = portfolio_payload["turnover_series"]
    rebalance_log_df = portfolio_payload["rebalance_log_df"]
    gross_portfolio_value = portfolio_payload["gross_portfolio_value"]
    vol_target_results = portfolio_payload["vol_target_results"]
    defensive_metadata = portfolio_payload["defensive_metadata"]

    with tabs[0]:
        st.header("Portfolio Overview")
        render_data_quality_report(data_quality_summary)
        render_portfolio_summary(portfolio_payload["formatted_metrics"])

        overview_col1, overview_col2 = st.columns(2)
        with overview_col1:
            st.plotly_chart(plot_weight_bar(weights), use_container_width=True)
        with overview_col2:
            st.plotly_chart(plot_weight_pie(weights), use_container_width=True)

        render_allocation_table(weights)

    with tabs[1]:
        st.header("Backtest Results")
        comparison_metric = st.selectbox(
            "Comparison Metric",
            ["cagr", "sharpe", "sortino", "volatility", "max_drawdown", "calmar"],
            index=1,
            key="tab_comparison_metric",
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
            key="tab_relative_metric",
        )

        st.plotly_chart(plot_equity_curve(portfolio_value), use_container_width=True)
        drawdown_series = RiskAnalytics.drawdown_series(portfolio_returns)
        st.plotly_chart(plot_drawdowns(drawdown_series), use_container_width=True)

        benchmark_col1, benchmark_col2 = st.columns(2)
        with benchmark_col1:
            st.plotly_chart(plot_performance_curves(growth_curves), use_container_width=True)
        with benchmark_col2:
            st.plotly_chart(plot_drawdown_curves(drawdown_curves), use_container_width=True)

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.plotly_chart(
                plot_metric_comparison(performance_comparison_df, comparison_metric),
                use_container_width=True,
            )
        with metric_col2:
            st.plotly_chart(
                plot_final_value_comparison(performance_comparison_df),
                use_container_width=True,
            )

        st.plotly_chart(
            plot_relative_performance(relative_performance_df, relative_metric),
            use_container_width=True,
        )
        st.dataframe(performance_comparison_df, use_container_width=True)
        st.dataframe(relative_performance_df, use_container_width=True)

    with tabs[2]:
        st.header("Risk & Allocation")
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

        st.dataframe(risk_contribution_df, use_container_width=True)

        if hrp_herc_risk_comparison_df is not None:
            st.subheader("HRP vs HERC Risk Contribution Comparison")
            st.plotly_chart(
                plot_hrp_herc_risk_comparison(hrp_herc_risk_comparison_df),
                use_container_width=True,
            )
            st.dataframe(hrp_herc_risk_comparison_df, use_container_width=True)

        st.plotly_chart(
            plot_correlation_heatmap(correlation_matrix_df),
            use_container_width=True,
        )
        st.pyplot(plot_dendrogram(linkage_matrix, labels=list(weights.index)))

    with tabs[3]:
        st.header("Trading Activity")

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
        cost_col1.metric("Gross Final Value", f"{cost_drag_summary['gross_final_value']:.2f}")
        cost_col2.metric(
            "Cost Drag",
            f"{cost_drag_summary['cost_drag']:.2f} ({cost_drag_summary['cost_drag_pct']:.2%})",
        )

        trading_col1, trading_col2 = st.columns(2)
        with trading_col1:
            if not turnover_series.empty:
                st.plotly_chart(plot_turnover_series(turnover_series), use_container_width=True)
            else:
                st.info("No turnover events recorded.")
        with trading_col2:
            if not rebalance_log_df.empty:
                st.plotly_chart(plot_transaction_costs(rebalance_log_df), use_container_width=True)
            else:
                st.info("No transaction costs recorded.")

        trading_col3, trading_col4 = st.columns(2)
        with trading_col3:
            st.plotly_chart(
                plot_rebalance_events(portfolio_value, rebalance_log_df),
                use_container_width=True,
            )
        with trading_col4:
            st.plotly_chart(
                plot_cost_adjusted_comparison(gross_portfolio_value, portfolio_value),
                use_container_width=True,
            )

        st.subheader("Rebalance Log")
        st.dataframe(rebalance_log_df, use_container_width=True)

    with tabs[4]:
        st.header("Volatility Targeting")
        if vol_target_results is None or defensive_metadata is None:
            st.info("Enable Volatility Targeting and run the portfolio analysis to view overlay outputs.")
        else:
            targeted_returns = vol_target_results["targeted_returns"]
            exposure_series = vol_target_results["exposure_series"]
            realized_volatility = vol_target_results["realized_volatility"]
            regime_series = vol_target_results["regime_series"]
            target_volatility = vol_target_results["target_volatility"]
            overlay_summary = vol_target_results["summary"]
            diagnostics_df = vol_target_results["diagnostics_df"]

            base_growth = (1.0 + portfolio_returns).cumprod()
            targeted_growth = (1.0 + targeted_returns).cumprod()
            base_drawdown = RiskAnalytics.drawdown_series(portfolio_returns)
            targeted_drawdown = RiskAnalytics.drawdown_series(targeted_returns)

            base_metrics = PerformanceAnalytics.summary_table(portfolio_returns)
            targeted_metrics = PerformanceAnalytics.summary_table(targeted_returns)
            comparison_table = pd.DataFrame(
                {
                    "Base Strategy": {**base_metrics, "final_growth": float(base_growth.iloc[-1])},
                    "Volatility Targeted": {
                        **targeted_metrics,
                        "final_growth": float(targeted_growth.iloc[-1]),
                    },
                }
            )

            vt_col1, vt_col2, vt_col3, vt_col4 = st.columns(4)
            vt_col1.metric("Average Exposure", f"{overlay_summary['average_exposure']:.2f}")
            vt_col2.metric("Min Exposure", f"{overlay_summary['min_exposure']:.2f}")
            vt_col3.metric("Max Exposure", f"{overlay_summary['max_exposure']:.2f}")
            vt_col4.metric(
                "Final Growth Delta",
                f"{overlay_summary['final_targeted_growth'] - overlay_summary['final_base_growth']:.2f}",
            )

            vt_chart_col1, vt_chart_col2 = st.columns(2)
            with vt_chart_col1:
                st.plotly_chart(
                    plot_base_vs_vol_targeted_growth(base_growth, targeted_growth),
                    use_container_width=True,
                )
            with vt_chart_col2:
                st.plotly_chart(plot_exposure_series(exposure_series), use_container_width=True)

            vt_chart_col3, vt_chart_col4 = st.columns(2)
            with vt_chart_col3:
                st.plotly_chart(
                    plot_realized_vs_target_vol(realized_volatility, target_volatility),
                    use_container_width=True,
                )
            with vt_chart_col4:
                st.plotly_chart(
                    plot_defensive_allocation(exposure_series),
                    use_container_width=True,
                )

            vt_chart_col5, vt_chart_col6 = st.columns(2)
            with vt_chart_col5:
                st.plotly_chart(plot_regime_series(regime_series), use_container_width=True)
            with vt_chart_col6:
                st.plotly_chart(
                    plot_drawdown_curves(
                        {
                            "Base Strategy": base_drawdown,
                            "Volatility Targeted": targeted_drawdown,
                        }
                    ),
                    use_container_width=True,
                )

            st.subheader("Volatility Targeting Summary")
            st.dataframe(comparison_table, use_container_width=True)

            regime_distribution_df = pd.DataFrame(
                {
                    "Regime": ["Calm", "Normal", "Stress", "Crisis"],
                    "Percent of Time": [
                        overlay_summary["percent_time_calm"],
                        overlay_summary["percent_time_normal"],
                        overlay_summary["percent_time_stress"],
                        overlay_summary["percent_time_crisis"],
                    ],
                }
            )
            st.dataframe(regime_distribution_df, use_container_width=True)

            st.subheader("Defensive Sleeve Metadata")
            st.dataframe(
                pd.DataFrame(
                    {"Field": list(defensive_metadata.keys()), "Value": list(defensive_metadata.values())}
                ),
                use_container_width=True,
            )

            st.subheader("Overlay Diagnostics")
            st.dataframe(diagnostics_df.tail(100), use_container_width=True)

    with tabs[5]:
        if sensitivity_payload is None:
            st.info("Run Sensitivity Study to populate experiment outputs.")
        else:
            render_sensitivity_content(sensitivity_payload)


def render_sensitivity_content(sensitivity_payload: dict[str, object]) -> None:
    st.header("Experiment Sensitivity")

    experiment_results_df = sensitivity_payload["experiment_results_df"]
    objective_metric = sensitivity_payload["objective_metric"]
    experiment_summary_df = sensitivity_payload["experiment_summary_df"]
    top_experiments_df = sensitivity_payload["top_experiments_df"]
    parameter_sensitivity_df = sensitivity_payload["parameter_sensitivity_df"]

    st.subheader("Experiment Results")
    st.dataframe(experiment_summary_df, use_container_width=True)

    sensitivity_col1, sensitivity_col2 = st.columns(2)
    with sensitivity_col1:
        st.plotly_chart(
            plot_top_experiments(experiment_results_df, objective_metric),
            use_container_width=True,
        )
    with sensitivity_col2:
        st.plotly_chart(
            plot_sensitivity_heatmap(
                experiment_results_df,
                x_param="covariance_method",
                y_param="strategy",
                metric=objective_metric,
            ),
            use_container_width=True,
        )

    sensitivity_col3, sensitivity_col4 = st.columns(2)
    with sensitivity_col3:
        st.plotly_chart(
            plot_experiment_metric_by_parameter(
                experiment_results_df,
                parameter="covariance_method",
                metric=objective_metric,
            ),
            use_container_width=True,
        )
    with sensitivity_col4:
        st.plotly_chart(
            plot_experiment_metric_by_parameter(
                experiment_results_df,
                parameter="rebalance_mode",
                metric=objective_metric,
            ),
            use_container_width=True,
        )

    st.subheader("Top 10 Configurations")
    st.dataframe(top_experiments_df, use_container_width=True)

    st.subheader("Parameter Sensitivity")
    st.dataframe(parameter_sensitivity_df, use_container_width=True)


initialize_session_state()

st.set_page_config(page_title="Adaptive Portfolio Risk Analytics", layout="wide")
st.title("Adaptive Portfolio Risk Analytics")

all_asset_labels = [ticker_label(ticker) for ticker in INDIAN_ASSET_UNIVERSE]

st.sidebar.header("Portfolio Inputs")

with st.sidebar.expander("Basic Portfolio Setup", expanded=True):
    preset = st.selectbox(
        "Universe Preset",
        ["Core Diversified", "Banks + IT + Gold", "Full Research Universe", "Custom"],
        key="ui_universe_preset",
    )
    select_all = st.checkbox(
        "Select All Assets In Preset",
        key="ui_select_all_assets",
    )

    if (
        preset != st.session_state.get("_last_preset")
        or select_all != st.session_state.get("_last_select_all")
    ):
        st.session_state["ui_selected_assets"] = update_selected_assets_for_preset(preset, select_all)
        st.session_state["_last_preset"] = preset
        st.session_state["_last_select_all"] = select_all

    asset_options = all_asset_labels if preset == "Custom" else default_labels_for_preset(preset)
    st.multiselect(
        "Selected Assets",
        options=asset_options,
        key="ui_selected_assets",
        help="Search by ticker or company name. Manual override below can replace this selection.",
    )

    with st.expander("Manual Ticker Override", expanded=False):
        st.text_input(
            "Manual Ticker Override",
            key="ui_manual_ticker_override",
            help="Optional comma-separated tickers. If provided, these tickers replace the multiselect selection.",
        )

    start_date = st.date_input("Start Date", date(2020, 1, 1), key="ui_start_date")
    end_date = st.date_input("End Date", date.today(), key="ui_end_date")
    initial_capital = st.number_input(
        "Initial Capital",
        min_value=100_000.0,
        max_value=100_000_000.0,
        value=1_000_000.0,
        step=100_000.0,
        key="ui_initial_capital",
    )

with st.sidebar.expander("Strategy & Backtest Settings", expanded=False):
    strategy = st.selectbox(
        "Strategy",
        ["Equal Weight", "Inverse Volatility", "HRP", "HERC"],
        help="Primary portfolio construction method for the main analysis.",
        key="ui_strategy",
    )
    comparison_strategies = st.multiselect(
        "Benchmark Comparison Strategies",
        ["Equal Weight", "Inverse Volatility", "HRP", "HERC"],
        default=["Equal Weight", "Inverse Volatility", "HRP", "HERC"],
        help="Strategies included in the benchmark comparison table and charts.",
        key="ui_comparison_strategies",
    )
    benchmark_strategy = st.selectbox(
        "Benchmark Strategy",
        ["Equal Weight", "Inverse Volatility", "HRP", "HERC"],
        index=0,
        help="Reference strategy used for relative performance metrics.",
        key="ui_benchmark_strategy",
    )
    covariance_method = st.selectbox(
        "Covariance Method",
        COVARIANCE_METHOD_OPTIONS,
        index=0,
        help="Covariance estimator used by HRP, HERC, and research comparisons.",
        key="ui_covariance_method",
    )
    rebalance_mode = st.selectbox(
        "Rebalance Mode",
        ["calendar", "threshold", "calendar_or_threshold"],
        index=0,
        help="Calendar rebalances on schedule. Threshold modes wait for weight drift triggers.",
        key="ui_rebalance_mode",
    )
    if rebalance_mode in {"threshold", "calendar_or_threshold"}:
        threshold = st.slider(
            "Threshold",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            help="Maximum absolute weight drift required before a threshold rebalance triggers.",
            key="ui_threshold",
        )
    else:
        threshold = 0.05
    base_bps = st.number_input(
        "Base Cost (bps)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        help="Base proportional trading cost applied to turnover.",
        key="ui_base_bps",
    )
    slippage_bps = st.number_input(
        "Slippage (bps)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=1.0,
        help="Additional slippage cost layered on top of the base trading cost.",
        key="ui_slippage_bps",
    )
    run_portfolio_button = st.button("Run Portfolio Analysis", key="ui_run_portfolio")

with st.sidebar.expander("Advanced Risk Controls", expanded=False):
    # Advanced controls are hidden by default to keep first-run usage focused on the base workflow.
    enable_vol_targeting = st.checkbox(
        "Enable Volatility Targeting",
        value=False,
        key="ui_enable_vol_targeting",
    )
    if enable_vol_targeting:
        defensive_sleeve = st.selectbox(
            "Defensive Sleeve",
            ["LIQUIDBEES.NS", "LIQUIDETF.NS", "Synthetic Risk-Free"],
            key="ui_defensive_sleeve",
        )
        synthetic_annual_rate = st.number_input(
            "Synthetic Annual Rate",
            min_value=0.0,
            max_value=0.20,
            value=0.04,
            step=0.01,
            format="%.2f",
            key="ui_synthetic_annual_rate",
        )
        vol_target_mode = st.selectbox(
            "Vol Target Mode",
            ["Adaptive", "Fixed"],
            index=0,
            key="ui_vol_target_mode",
        )
        base_target_vol = st.slider(
            "Base Target Vol",
            min_value=0.03,
            max_value=0.20,
            value=0.10,
            step=0.01,
            key="ui_base_target_vol",
        )
        realized_vol_window = st.slider(
            "Realized Vol Window",
            min_value=21,
            max_value=126,
            value=63,
            step=1,
            key="ui_realized_vol_window",
        )
        regime_lookback_window = st.slider(
            "Regime Lookback Window",
            min_value=126,
            max_value=504,
            value=252,
            step=21,
            key="ui_regime_lookback_window",
        )
        exposure_floor = st.slider(
            "Exposure Floor",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            key="ui_exposure_floor",
        )
        exposure_cap = st.slider(
            "Exposure Cap",
            min_value=0.25,
            max_value=1.0,
            value=1.0,
            step=0.05,
            key="ui_exposure_cap",
        )
        no_trade_band = st.slider(
            "No-Trade Band",
            min_value=0.00,
            max_value=0.20,
            value=0.05,
            step=0.01,
            key="ui_no_trade_band",
        )
    else:
        defensive_sleeve = "Synthetic Risk-Free"
        synthetic_annual_rate = 0.04
        vol_target_mode = "Adaptive"
        base_target_vol = 0.10
        realized_vol_window = 63
        regime_lookback_window = 252
        exposure_floor = 0.25
        exposure_cap = 1.0
        no_trade_band = 0.05

with st.sidebar.expander("Experiment Sensitivity", expanded=False):
    sensitivity_strategies = st.multiselect(
        "Sensitivity Strategies",
        ["Equal Weight", "Inverse Volatility", "HRP", "HERC"],
        default=["HRP", "HERC"],
        key="ui_sensitivity_strategies",
    )
    sensitivity_covariance_methods = st.multiselect(
        "Sensitivity Covariance Methods",
        COVARIANCE_METHOD_OPTIONS,
        default=["sample", "ledoit_wolf", "ewma_ledoit_wolf"],
        key="ui_sensitivity_covariance_methods",
    )
    sensitivity_rebalance_modes = st.multiselect(
        "Sensitivity Rebalance Modes",
        ["calendar", "threshold"],
        default=["calendar", "threshold"],
        key="ui_sensitivity_rebalance_modes",
    )
    sensitivity_thresholds = st.text_input(
        "Sensitivity Thresholds",
        "0.03,0.05,0.10",
        help="Comma-separated thresholds such as 0.03,0.05,0.10.",
        key="ui_sensitivity_thresholds",
    )
    sensitivity_objective_label = st.selectbox(
        "Sensitivity Objective",
        list(SENSITIVITY_OBJECTIVE_MAP.keys()),
        index=3,
        key="ui_sensitivity_objective",
    )
    sensitivity_max_runs = st.number_input(
        "Sensitivity Max Runs",
        min_value=1,
        max_value=500,
        value=24,
        step=1,
        key="ui_sensitivity_max_runs",
    )
    run_sensitivity_button = st.button("Run Sensitivity Study", key="ui_run_sensitivity")


manual_override = st.session_state["ui_manual_ticker_override"]
selected_tickers = (
    parse_ticker_override(manual_override)
    if manual_override.strip()
    else selected_tickers_from_labels(st.session_state["ui_selected_assets"])
)

if run_portfolio_button:
    validation_error = validate_common_inputs(
        selected_tickers=selected_tickers,
        start_dt=start_date,
        end_dt=end_date,
        exposure_floor=exposure_floor,
        exposure_cap=exposure_cap,
        defensive_sleeve=defensive_sleeve,
    )
    if validation_error is not None:
        st.session_state[UI_MESSAGE_KEY] = ("warning", validation_error)
    else:
        try:
            portfolio_payload = build_portfolio_results(
                selected_tickers=selected_tickers,
                start_dt=start_date,
                end_dt=end_date,
                strategy=strategy,
                comparison_strategies=comparison_strategies,
                benchmark_strategy=benchmark_strategy,
                covariance_method=covariance_method,
                rebalance_mode=rebalance_mode,
                threshold=threshold,
                base_bps=base_bps,
                slippage_bps=slippage_bps,
                enable_vol_targeting=enable_vol_targeting,
                defensive_sleeve=defensive_sleeve,
                synthetic_annual_rate=synthetic_annual_rate,
                vol_target_mode=vol_target_mode,
                base_target_vol=base_target_vol,
                realized_vol_window=realized_vol_window,
                regime_lookback_window=regime_lookback_window,
                exposure_floor=exposure_floor,
                exposure_cap=exposure_cap,
                no_trade_band=no_trade_band,
                initial_capital=initial_capital,
            )
            st.session_state[PORTFOLIO_RESULT_KEY] = portfolio_payload
            st.session_state[UI_MESSAGE_KEY] = ("info", "Portfolio analysis updated.")
        except Exception as exc:
            st.session_state[UI_MESSAGE_KEY] = ("error", f"Portfolio analysis failed: {exc}")

if run_sensitivity_button:
    validation_error = validate_common_inputs(
        selected_tickers=selected_tickers,
        start_dt=start_date,
        end_dt=end_date,
        exposure_floor=exposure_floor,
        exposure_cap=exposure_cap,
        defensive_sleeve=defensive_sleeve,
    )
    if validation_error is not None:
        st.session_state[UI_MESSAGE_KEY] = ("warning", validation_error)
    else:
        try:
            sensitivity_payload = build_sensitivity_results(
                selected_tickers=selected_tickers,
                start_dt=start_date,
                end_dt=end_date,
                base_bps=base_bps,
                slippage_bps=slippage_bps,
                enable_vol_targeting=enable_vol_targeting,
                defensive_sleeve=defensive_sleeve,
                synthetic_annual_rate=synthetic_annual_rate,
                base_target_vol=base_target_vol,
                strategies=sensitivity_strategies,
                covariance_methods=sensitivity_covariance_methods,
                rebalance_modes=sensitivity_rebalance_modes,
                thresholds_text=sensitivity_thresholds,
                objective_label=sensitivity_objective_label,
                max_runs=int(sensitivity_max_runs),
                initial_capital=initial_capital,
            )
            st.session_state[SENSITIVITY_RESULT_KEY] = sensitivity_payload
            st.session_state[UI_MESSAGE_KEY] = ("info", "Sensitivity study completed.")
        except ValueError as exc:
            st.session_state[UI_MESSAGE_KEY] = ("warning", str(exc))
        except Exception as exc:
            st.session_state[UI_MESSAGE_KEY] = ("error", f"Sensitivity study failed: {exc}")


if st.session_state.get(UI_MESSAGE_KEY) is not None:
    message_level, message_text = st.session_state[UI_MESSAGE_KEY]
    show_message(message_level, message_text)

render_dashboard_tabs(
    st.session_state.get(PORTFOLIO_RESULT_KEY),
    st.session_state.get(SENSITIVITY_RESULT_KEY),
)
