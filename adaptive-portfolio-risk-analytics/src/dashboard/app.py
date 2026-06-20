from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.analytics import (
    PerformanceAnalytics,
    RiskAnalytics,
    calculate_correlation_stress,
    calculate_active_risk_metrics,
    calculate_historical_es,
    calculate_historical_var,
    calculate_historical_stress_performance,
    calculate_hypothetical_stress_table,
    calculate_liquidity_diagnostics,
    calculate_stress_period_benchmark_comparison,
    calculate_var_exceptions,
    compare_risk_contributions,
    find_worst_periods,
    risk_contribution_table,
    summarize_liquidity_diagnostics,
)
from src.adaptive import (
    defensive_source_from_label,
    format_defensive_source,
    get_defensive_returns,
    get_policy_preset,
    run_regime_adaptive_backtest,
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
from src.dashboard.modes import (
    DASHBOARD_MODES,
    DEFAULT_DASHBOARD_MODE,
    DEFAULT_MANAGER_ADAPTIVE_OVERLAY,
    DEFAULT_RESEARCH_OBJECTIVE,
    DEVELOPER_VIEW,
    MANAGER_VIEW,
    MANAGER_PROFILE_OBJECTIVES,
    MODE_NOTES,
    RESEARCH_OBJECTIVES,
    RESEARCH_VIEW,
    RULE_BASED_ROBUSTNESS_REFERENCE,
    adaptive_overlay_name,
    classify_recommended_use,
    net_metric_label,
    objective_metric,
    research_objective_label,
)
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
    plot_hmm_state_probabilities,
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
from src.data_pipeline import DataPreprocessor, YahooFinanceProvider
from src.experiments import (
    AdaptiveExperimentConfig,
    ExperimentConfig,
    build_adaptive_attribution,
    build_adaptive_stress_comparison,
    build_experiment_summary_table,
    build_top_n_table,
    compare_adaptive_vs_fixed,
    compute_parameter_sensitivity,
    run_adaptive_experiment_grid,
    run_experiment_grid,
)
from src.optimization import HERCAllocator, HRPAllocator
from src.regime import (
    DEFAULT_HMM_FEATURE_COLUMNS,
    HMM_AVAILABLE,
    calculate_regime_features,
    calculate_regime_performance,
    calculate_regime_state_table,
    calculate_regime_transitions,
    calculate_strategy_regime_summary,
    classify_rule_based_regime,
    compare_regime_methods,
    fit_hmm_full_sample,
    fit_hmm_walk_forward,
    lag_regime_labels,
    map_hmm_states_to_regimes,
    select_best_strategy_by_regime,
)
from src.validation import run_cpcv_validation
from src.selection import (
    COST_ASSUMPTIONS,
    COST_ASSUMPTION_NAMES,
    PROFILE_NAMES,
    StrategyRecommendation,
    build_strategy_playbook,
    load_selection_artifacts,
    select_strategy_for_profile,
)

PORTFOLIO_RESULT_KEY = "dashboard_portfolio_results"
SENSITIVITY_RESULT_KEY = "dashboard_sensitivity_results"
ROBUSTNESS_RESULT_KEY = "dashboard_robustness_results"
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
    **RESEARCH_OBJECTIVES,
}

LOWER_IS_BETTER_METRICS = {"volatility"}


def ticker_label(ticker: str) -> str:
    company_name = INDIAN_ASSET_UNIVERSE.get(ticker, "Custom ticker")
    return f"{ticker} — {company_name}"


def initialize_session_state() -> None:
    default_labels = [ticker_label(ticker) for ticker in DEFAULT_UNIVERSES["Core Diversified"]]
    defaults = {
        "ui_universe_preset": "Core Diversified",
        "ui_select_all_assets": True,
        "ui_selected_assets": default_labels,
        "_last_preset": "Core Diversified",
        "_last_select_all": True,
        "ui_ticker_to_add": "",
        "ui_added_tickers": [],
        "ui_dashboard_mode": DEFAULT_DASHBOARD_MODE,
        "ui_research_objective": DEFAULT_RESEARCH_OBJECTIVE,
        "ui_manager_universe_preset": "Core Diversified",
        "ui_manager_investor_profile": "Balanced",
        "ui_manager_cost_assumption": "Moderate",
        "ui_manager_start_date": date(2020, 1, 1),
        "ui_manager_end_date": date.today(),
        "ui_manager_initial_capital": 1_000_000.0,
        "ui_manager_custom_base_bps": 10.0,
        "ui_manager_custom_slippage_bps": 5.0,
        "ui_strategy": "HERC",
        "ui_enable_regime_analytics": True,
        "ui_regime_method": DEFAULT_MANAGER_ADAPTIVE_OVERLAY["regime_method"],
        "ui_enable_adaptive_strategy": True,
        "ui_adaptive_regime_source": DEFAULT_MANAGER_ADAPTIVE_OVERLAY["regime_source"],
        "ui_adaptive_policy_preset": DEFAULT_MANAGER_ADAPTIVE_OVERLAY["policy_preset"],
        "ui_covariance_method": "sample",
        "ui_rebalance_mode": "calendar",
        "ui_threshold": 0.05,
        "ui_defensive_sleeve": "Synthetic 4% annualized",
        "ui_synthetic_annual_rate": 0.04,
        PORTFOLIO_RESULT_KEY: None,
        SENSITIVITY_RESULT_KEY: None,
        ROBUSTNESS_RESULT_KEY: None,
        UI_MESSAGE_KEY: None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_allocator(strategy_name: str, covariance_method: str = "sample"):
    return BenchmarkFactory.get_allocator(strategy_name, covariance_method=covariance_method)


def parse_ticker_entries(raw_value: str) -> list[str]:
    normalized_value = raw_value.replace(";", ",").replace("\n", ",")
    tickers = [ticker.strip().upper() for ticker in normalized_value.split(",") if ticker.strip()]
    return list(dict.fromkeys(tickers))


def parse_ticker_override(raw_value: str) -> list[str]:
    return parse_ticker_entries(raw_value)


def parse_float_list(raw_values: str) -> list[float]:
    try:
        values = [
            float(raw_value.strip()) for raw_value in raw_values.split(",") if raw_value.strip()
        ]
    except ValueError as exc:
        raise ValueError(
            "Sensitivity Thresholds must be comma-separated decimals like 0.03,0.05,0.10"
        ) from exc
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


def merge_portfolio_tickers(selected_labels: list[str], added_tickers: list[str]) -> list[str]:
    selected_tickers = selected_tickers_from_labels(selected_labels)
    cleaned_added_tickers = parse_ticker_entries(",".join(added_tickers))
    return list(dict.fromkeys(selected_tickers + cleaned_added_tickers))


def add_tickers_to_portfolio() -> None:
    requested_tickers = parse_ticker_entries(st.session_state.get("ui_ticker_to_add", ""))
    if not requested_tickers:
        st.session_state[UI_MESSAGE_KEY] = ("warning", "Enter at least one ticker to add.")
        return

    selected_tickers = selected_tickers_from_labels(st.session_state.get("ui_selected_assets", []))
    current_added_tickers = list(st.session_state.get("ui_added_tickers", []))
    current_portfolio_tickers = set(selected_tickers + current_added_tickers)
    new_tickers = [
        ticker for ticker in requested_tickers if ticker not in current_portfolio_tickers
    ]

    if not new_tickers:
        st.session_state[UI_MESSAGE_KEY] = ("info", "Ticker is already in the portfolio.")
        st.session_state["ui_ticker_to_add"] = ""
        return

    st.session_state["ui_added_tickers"] = list(dict.fromkeys(current_added_tickers + new_tickers))
    st.session_state["ui_ticker_to_add"] = ""
    st.session_state[UI_MESSAGE_KEY] = ("info", f"Added ticker(s): {', '.join(new_tickers)}.")


def clear_added_tickers() -> None:
    if st.session_state.get("ui_added_tickers"):
        st.session_state["ui_added_tickers"] = []
        st.session_state[UI_MESSAGE_KEY] = ("info", "Added tickers cleared.")


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
    defensive_source, defensive_ticker = defensive_source_from_label(defensive_sleeve)
    if defensive_source == "ticker" and defensive_ticker in selected_tickers:
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


def build_active_risk_metrics_table(
    strategy_results: dict[str, dict],
    performance_comparison_df: pd.DataFrame,
    benchmark_name: str,
) -> pd.DataFrame:
    """Build active-risk metrics for every strategy against the selected benchmark."""
    if not strategy_results or performance_comparison_df.empty:
        return pd.DataFrame()

    benchmark_display_name = BenchmarkFactory.normalize_strategy_name(benchmark_name)
    if benchmark_display_name not in strategy_results:
        return pd.DataFrame()

    benchmark_result = strategy_results[benchmark_display_name]
    benchmark_cagr = _lookup_strategy_metric(
        performance_comparison_df,
        benchmark_display_name,
        "cagr",
    )
    rows = []

    for strategy_name, result in strategy_results.items():
        strategy_cagr = _lookup_strategy_metric(
            performance_comparison_df,
            strategy_name,
            "cagr",
        )
        weights = result.get("weights_history")
        if weights is None or getattr(weights, "empty", True):
            weights = result.get("latest_weights")

        metrics = calculate_active_risk_metrics(
            strategy_returns=result["portfolio_returns"],
            benchmark_returns=benchmark_result["portfolio_returns"],
            strategy_values=result.get("portfolio_values"),
            weights=weights,
            strategy_cagr=strategy_cagr,
            benchmark_cagr=benchmark_cagr,
        )
        rows.append(
            {
                "strategy": strategy_name,
                "benchmark": benchmark_display_name,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def _lookup_strategy_metric(
    performance_comparison_df: pd.DataFrame,
    strategy_name: str,
    metric: str,
) -> float | None:
    if (
        strategy_name not in performance_comparison_df.index
        or metric not in performance_comparison_df
    ):
        return None
    value = performance_comparison_df.loc[strategy_name, metric]
    return float(value) if pd.notna(value) else None


def format_active_risk_metrics_table(active_risk_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Return a readable presentation table for active-risk metrics."""
    if active_risk_metrics_df.empty:
        return pd.DataFrame()

    display_df = pd.DataFrame(
        {
            "Strategy": active_risk_metrics_df["strategy"],
            "Benchmark": active_risk_metrics_df["benchmark"],
            "Simple Alpha": active_risk_metrics_df["simple_alpha"].map(format_percent),
            "Jensen's Alpha": active_risk_metrics_df["jensen_alpha_annualized"].map(format_percent),
            "Beta vs Benchmark": active_risk_metrics_df["beta"].map(format_decimal),
            "Tracking Error": active_risk_metrics_df["tracking_error"].map(format_percent),
            "Information Ratio": active_risk_metrics_df["information_ratio"].map(format_decimal),
            "Hit Ratio": active_risk_metrics_df["hit_ratio"].map(format_percent),
            "Max DD Duration": active_risk_metrics_df["max_drawdown_duration"].map(format_days),
            "Current DD Duration": active_risk_metrics_df["current_drawdown_duration"].map(
                format_days
            ),
            "HHI": active_risk_metrics_df["hhi"].map(lambda value: format_decimal(value, digits=3)),
            "Effective N": active_risk_metrics_df["effective_n"].map(
                lambda value: format_decimal(value, digits=2)
            ),
        }
    )
    return display_df


def render_key_takeaways(
    performance_comparison_df: pd.DataFrame,
    strategy_results: dict[str, dict],
    active_risk_metrics_df: pd.DataFrame,
    *,
    selected_objective_label: str,
    selected_objective_metric: str,
    volatility_targeting_enabled: bool,
) -> None:
    """Render compact presentation cards for the most important results."""
    st.subheader("Key Takeaways")
    st.info(
        "The experiment ranking is single-objective. The dashboard ranks strategies by the "
        "selected objective only. Other metrics are diagnostics unless they affect the selected "
        "objective through performance."
    )

    selected_strategy, selected_value = _best_strategy_by_metric(
        performance_comparison_df,
        selected_objective_metric,
        lower_is_better=selected_objective_metric in LOWER_IS_BETTER_METRICS,
    )
    drawdown_strategy, drawdown_value = _best_strategy_by_metric(
        performance_comparison_df,
        "max_drawdown",
    )
    volatility_strategy, volatility_value = _best_strategy_by_metric(
        performance_comparison_df,
        "volatility",
        lower_is_better=True,
    )
    sharpe_strategy, sharpe_value = _best_strategy_by_metric(performance_comparison_df, "sharpe")
    calmar_strategy, calmar_value = _best_strategy_by_metric(performance_comparison_df, "calmar")
    final_value_strategy, final_value = _best_strategy_by_metric(
        performance_comparison_df,
        "final_value",
    )
    turnover_strategy, turnover_value = _best_backtest_metric(strategy_results, "average_turnover")
    cost_strategy, cost_value = _best_backtest_metric(strategy_results, "total_transaction_cost")

    cards = [
        (
            "Best by selected objective",
            f"{selected_strategy} ({selected_objective_label}: "
            f"{format_metric_for_card(selected_objective_metric, selected_value)})",
        ),
        ("Best drawdown control", f"{drawdown_strategy} ({format_percent(drawdown_value)})"),
        ("Lowest volatility", f"{volatility_strategy} ({format_percent(volatility_value)})"),
        ("Highest Net Sharpe", f"{sharpe_strategy} ({format_decimal(sharpe_value)})"),
        ("Highest Net Calmar", f"{calmar_strategy} ({format_decimal(calmar_value)})"),
        (
            "Highest Net Final Value",
            f"{final_value_strategy} ({format_currency(final_value)})",
        ),
        ("Lowest turnover", f"{turnover_strategy} ({format_decimal(turnover_value)})"),
        ("Lowest transaction cost", f"{cost_strategy} ({format_currency(cost_value)})"),
        ("Volatility targeting", "Enabled" if volatility_targeting_enabled else "Disabled"),
    ]
    _render_metric_cards(cards)
    render_interpretation_badges(
        performance_comparison_df,
        strategy_results,
        active_risk_metrics_df,
    )


def render_risk_tracking_explainer() -> None:
    with st.expander("What Risks Are We Tracking?"):
        st.write(
            "Systematic risk is market-wide risk that diversification cannot fully remove. "
            "Unsystematic risk is asset-specific risk that diversification can reduce."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    ("Market risk", "Volatility, VaR, ES/CVaR, drawdown"),
                    ("Systematic risk", "Beta versus benchmark, market drawdown sensitivity"),
                    (
                        "Unsystematic risk",
                        "Diversification through covariance/correlation/clustering",
                    ),
                    ("Correlation risk", "Correlation matrix, distance matrix, dendrogram"),
                    ("Concentration risk", "HHI, Effective N, risk contribution"),
                    ("Benchmark active risk", "Alpha, tracking error, information ratio"),
                    (
                        "Liquidity trading risk",
                        "Turnover, transaction cost, slippage, ADTV, participation rate",
                    ),
                    ("Model risk", "Covariance estimator sensitivity, parameter sensitivity"),
                    ("Regime / volatility-state risk", "Volatility targeting overlay"),
                    ("Tail risk", "VaR, ES/CVaR, stress testing"),
                ],
                columns=["FRM Risk Concept", "Project Mapping"],
            ),
            use_container_width=True,
        )


def render_active_risk_metric_help() -> None:
    with st.expander("Active Risk Metric Guide"):
        st.markdown("""
- Simple Alpha: strategy CAGR minus benchmark CAGR.
- Jensen's Alpha: CAPM-style annualized alpha after accounting for benchmark beta.
- Beta vs Benchmark: sensitivity of strategy returns to benchmark returns.
- Tracking Error: annualized volatility of daily active returns.
- Information Ratio: annualized active return divided by tracking error.
- Hit Ratio: share of days the strategy beats the benchmark.
- Max DD Duration: longest time spent below a prior portfolio peak.
- HHI and Effective N: allocation concentration and implied number of equally weighted assets.
""")


def render_interpretation_badges(
    performance_comparison_df: pd.DataFrame,
    strategy_results: dict[str, dict],
    active_risk_metrics_df: pd.DataFrame,
) -> None:
    messages = []

    if "max_drawdown" in performance_comparison_df:
        high_drawdown = performance_comparison_df[
            pd.to_numeric(performance_comparison_df["max_drawdown"], errors="coerce") < -0.25
        ]
        if not high_drawdown.empty:
            messages.append(
                "High drawdown risk: "
                + ", ".join(map(str, high_drawdown.index))
                + " breached -25% max drawdown."
            )

    turnover_series = _backtest_metric_series(strategy_results, "average_turnover")
    high_turnover = _high_relative_values(turnover_series)
    if not high_turnover.empty:
        messages.append(
            "High turnover: transaction costs may erode performance for "
            + ", ".join(high_turnover.index.astype(str))
            + "."
        )

    if not active_risk_metrics_df.empty:
        concentration_rows = active_risk_metrics_df.dropna(subset=["effective_n"])
        asset_count = _infer_asset_count_from_strategy_results(strategy_results)
        concentrated = pd.DataFrame()
        if asset_count > 0:
            concentrated = concentration_rows[concentration_rows["effective_n"] < 0.4 * asset_count]
        if not concentrated.empty:
            messages.append(
                "Concentrated allocation: "
                + ", ".join(concentrated["strategy"].astype(str))
                + " has a low effective number of assets."
            )

        tracking_error = pd.Series(
            pd.to_numeric(active_risk_metrics_df["tracking_error"], errors="coerce").values,
            index=active_risk_metrics_df["strategy"].astype(str),
            dtype=float,
        ).dropna()
        high_tracking_error = _high_relative_values(tracking_error)
        if not high_tracking_error.empty:
            messages.append(
                "High active risk versus benchmark: "
                + ", ".join(high_tracking_error.index.astype(str))
                + " has elevated tracking error versus peers."
            )

    if messages:
        st.subheader("Interpretation Badges")
        for message in messages:
            st.warning(message)


def _best_strategy_by_metric(
    performance_comparison_df: pd.DataFrame,
    metric: str,
    *,
    lower_is_better: bool = False,
) -> tuple[str, float]:
    if metric not in performance_comparison_df:
        return "n/a", float("nan")
    series = pd.to_numeric(performance_comparison_df[metric], errors="coerce").dropna()
    if series.empty:
        return "n/a", float("nan")
    index = series.idxmin() if lower_is_better else series.idxmax()
    return str(index), float(series.loc[index])


def _best_backtest_metric(strategy_results: dict[str, dict], metric: str) -> tuple[str, float]:
    series = _backtest_metric_series(strategy_results, metric)
    if series.empty:
        return "n/a", float("nan")
    index = series.idxmin()
    return str(index), float(series.loc[index])


def _backtest_metric_series(strategy_results: dict[str, dict], metric: str) -> pd.Series:
    values = {}
    for strategy_name, result in strategy_results.items():
        metrics = result.get("performance_metrics", {})
        value = metrics.get(metric)
        if value is not None and pd.notna(value):
            values[strategy_name] = float(value)
    return pd.Series(values, dtype=float).dropna()


def _high_relative_values(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").dropna()
    values = values[values > 0.0]
    if len(values) < 2 or np.isclose(float(values.max()), float(values.min())):
        return pd.Series(dtype=float)
    threshold = float(values.quantile(0.75))
    return values[values > threshold]


def _infer_asset_count_from_strategy_results(strategy_results: dict[str, dict]) -> int:
    for result in strategy_results.values():
        weights = result.get("latest_weights")
        if isinstance(weights, pd.Series) and not weights.empty:
            return len(weights)
        weights_history = result.get("weights_history")
        if isinstance(weights_history, pd.DataFrame) and not weights_history.empty:
            return weights_history.shape[1]
    return 0


def _render_metric_cards(cards: list[tuple[str, str]]) -> None:
    for start in range(0, len(cards), 3):
        row = cards[start : start + 3]
        columns = st.columns(len(row))
        for column, (label, value) in zip(columns, row):
            column.metric(label, value)


def format_metric_for_card(metric: str, value: float) -> str:
    if metric in {"cagr", "volatility", "max_drawdown"}:
        return format_percent(value)
    if metric == "final_value":
        return format_currency(value)
    return format_decimal(value)


def dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a dashboard table for audit-friendly download."""
    return frame.to_csv(index=True).encode("utf-8")


def render_dataframe_download(
    label: str,
    frame: pd.DataFrame,
    file_name: str,
    *,
    key: str,
) -> None:
    """Render a CSV download without forcing a large raw table inline."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        st.caption(f"{label}: no data available.")
        return
    st.download_button(
        label,
        data=dataframe_csv_bytes(frame),
        file_name=file_name,
        mime="text/csv",
        key=key,
    )


def format_net_performance_table(performance_df: pd.DataFrame) -> pd.DataFrame:
    """Return a comparison table whose return-derived columns are explicitly net."""
    if not isinstance(performance_df, pd.DataFrame) or performance_df.empty:
        return pd.DataFrame()
    display = performance_df.copy()
    display.index.name = "Strategy"
    return display.rename(
        columns={column: net_metric_label(column) for column in display.columns}
    )


def _performance_row_from_result(
    result: dict[str, object],
) -> dict[str, float]:
    metrics = PerformanceAnalytics.summary_table(result["portfolio_returns"])
    performance = result.get("performance_metrics", {})
    return {
        **metrics,
        "final_value": float(result["portfolio_values"].iloc[-1]),
        "total_turnover": float(performance.get("total_turnover", np.nan)),
        "total_transaction_cost": float(
            performance.get("total_transaction_cost", np.nan)
        ),
        "number_of_rebalances": float(
            performance.get("number_of_rebalances", np.nan)
        ),
    }


def build_manager_decision_table(
    portfolio_payload: dict[str, object],
) -> tuple[pd.DataFrame, str, str, str]:
    """Build the manager's fixed/adaptive/benchmark comparison on a net basis."""
    fixed_table = portfolio_payload["performance_comparison_df"]
    strategy_results = portfolio_payload.get("strategy_results", {})
    adaptive_results = portfolio_payload.get("adaptive_results")
    benchmark_name = BenchmarkFactory.normalize_strategy_name(
        portfolio_payload.get("benchmark_strategy", "Equal Weight")
    )

    fixed_growth_name, _ = _best_strategy_by_metric(fixed_table, "cagr")
    selected_names = [fixed_growth_name]
    if benchmark_name not in selected_names:
        selected_names.append(benchmark_name)

    rows: dict[str, dict[str, float]] = {}
    for strategy_name in selected_names:
        if strategy_name in strategy_results:
            rows[strategy_name] = _performance_row_from_result(
                strategy_results[strategy_name]
            )

    adaptive_name = adaptive_overlay_name(
        str(portfolio_payload.get("adaptive_regime_source", "")),
        str(portfolio_payload.get("adaptive_policy_preset", "Conservative")),
    )
    if adaptive_results is not None:
        rows[adaptive_name] = _performance_row_from_result(adaptive_results)

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Strategy"
    return table, fixed_growth_name, adaptive_name, benchmark_name


def _format_manager_decision_table(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    display = table[
        [
            "cagr",
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown",
            "final_value",
            "total_turnover",
            "total_transaction_cost",
            "number_of_rebalances",
        ]
    ].copy()
    for column in ["cagr", "max_drawdown"]:
        display[column] = display[column].map(format_percent)
    for column in ["sharpe", "sortino", "calmar", "total_turnover"]:
        display[column] = display[column].map(format_decimal)
    for column in ["final_value", "total_transaction_cost"]:
        display[column] = display[column].map(format_currency)
    display["number_of_rebalances"] = display["number_of_rebalances"].map(
        lambda value: "n/a" if not _is_finite(value) else str(int(value))
    )
    return display.rename(
        columns={column: net_metric_label(column) for column in display.columns}
    )


def format_percent(value: float) -> str:
    if not _is_finite(value):
        return "n/a"
    return f"{float(value):.2%}"


def format_decimal(value: float, digits: int = 2) -> str:
    if not _is_finite(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def format_currency(value: float) -> str:
    if not _is_finite(value):
        return "n/a"
    return f"{float(value):,.0f}"


def format_days(value: float) -> str:
    if not _is_finite(value):
        return "n/a"
    return f"{int(round(float(value)))} days"


def build_var_es_metrics_table(
    strategy_results: dict[str, dict],
    *,
    confidence_level: float,
    holding_period_days: int,
    value_basis: str,
    initial_capital: float,
) -> pd.DataFrame:
    """Build VaR/ES and exception diagnostics for strategy return streams."""
    rows = []
    for strategy_name, result in strategy_results.items():
        portfolio_value = (
            float(initial_capital)
            if value_basis == "Initial capital"
            else float(result["portfolio_values"].iloc[-1])
        )
        returns = result["portfolio_returns"]
        var_result = calculate_historical_var(
            returns,
            confidence_level=confidence_level,
            holding_period_days=holding_period_days,
            portfolio_value=portfolio_value,
        )
        es_result = calculate_historical_es(
            returns,
            confidence_level=confidence_level,
            holding_period_days=holding_period_days,
            portfolio_value=portfolio_value,
        )
        exception_result = calculate_var_exceptions(
            returns,
            confidence_level=confidence_level,
            rolling_window=252 if len(returns) > 252 else None,
        )
        rows.append(
            {
                "strategy": strategy_name,
                "confidence_level": confidence_level,
                "holding_period_days": holding_period_days,
                "historical_var": var_result["var_return"],
                "historical_var_amount": var_result["var_amount"],
                "historical_es": es_result["es_return"],
                "historical_es_amount": es_result["es_amount"],
                "actual_exceptions": exception_result["actual_exceptions"],
                "expected_exceptions": exception_result["expected_exceptions"],
                "exception_ratio": exception_result["exception_ratio"],
                "exception_rate": exception_result["exception_rate"],
                "expected_exception_rate": exception_result["expected_exception_rate"],
                "n_observations": exception_result["n_observations"],
                "exception_interpretation": interpret_exception_ratio(
                    exception_result["exception_ratio"]
                ),
            }
        )
    return pd.DataFrame(rows)


def format_var_es_metrics_table(var_es_df: pd.DataFrame) -> pd.DataFrame:
    if var_es_df.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "Strategy": var_es_df["strategy"],
            "Confidence": var_es_df["confidence_level"].map(format_percent),
            "Holding Period": var_es_df["holding_period_days"].map(
                lambda value: f"{int(value)} days"
            ),
            "Historical VaR": var_es_df["historical_var"].map(format_percent),
            "VaR Amount": var_es_df["historical_var_amount"].map(format_currency),
            "Historical ES/CVaR": var_es_df["historical_es"].map(format_percent),
            "ES/CVaR Amount": var_es_df["historical_es_amount"].map(format_currency),
        }
    )


def format_var_exception_table(var_es_df: pd.DataFrame) -> pd.DataFrame:
    if var_es_df.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "Strategy": var_es_df["strategy"],
            "Actual Exceptions": var_es_df["actual_exceptions"].map(
                lambda value: "n/a" if pd.isna(value) else f"{int(value)}"
            ),
            "Expected Exceptions": var_es_df["expected_exceptions"].map(
                lambda value: format_decimal(value, digits=1)
            ),
            "Exception Ratio": var_es_df["exception_ratio"].map(format_decimal),
            "Exception Rate": var_es_df["exception_rate"].map(format_percent),
            "Expected Rate": var_es_df["expected_exception_rate"].map(format_percent),
            "Observations": var_es_df["n_observations"].map(
                lambda value: "n/a" if pd.isna(value) else f"{int(value)}"
            ),
            "Interpretation": var_es_df["exception_interpretation"],
        }
    )


def interpret_exception_ratio(exception_ratio: float) -> str:
    if not _is_finite(exception_ratio):
        return "Insufficient data"
    ratio = float(exception_ratio)
    if ratio > 1.5:
        return "VaR may be underestimating risk"
    if ratio < 0.5:
        return "VaR may be conservative"
    return "VaR exceptions broadly consistent"


def build_stress_period_benchmark_table(
    strategy_results: dict[str, dict],
    benchmark_name: str,
) -> pd.DataFrame:
    """Build stress-period strategy-vs-benchmark comparison rows."""
    benchmark_display_name = BenchmarkFactory.normalize_strategy_name(benchmark_name)
    if benchmark_display_name not in strategy_results:
        return pd.DataFrame()
    benchmark_returns = strategy_results[benchmark_display_name]["portfolio_returns"]
    rows = []
    for strategy_name, result in strategy_results.items():
        comparison_df = calculate_stress_period_benchmark_comparison(
            result["portfolio_returns"],
            benchmark_returns,
        )
        if comparison_df.empty:
            continue
        comparison_df.insert(0, "strategy", strategy_name)
        comparison_df.insert(1, "benchmark", benchmark_display_name)
        rows.append(comparison_df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def format_stress_period_benchmark_table(stress_df: pd.DataFrame) -> pd.DataFrame:
    if stress_df.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "Strategy": stress_df["strategy"],
            "Benchmark": stress_df["benchmark"],
            "Stress Period": stress_df["stress_period"],
            "Net Strategy Return": stress_df["strategy_stress_return"].map(format_percent),
            "Net Benchmark Return": stress_df["benchmark_stress_return"].map(format_percent),
            "Net Excess Stress Return": stress_df["excess_stress_return"].map(format_percent),
            "Strategy Max DD": stress_df["strategy_max_drawdown"].map(format_percent),
            "Benchmark Max DD": stress_df["benchmark_max_drawdown"].map(format_percent),
            "Drawdown Reduction": stress_df["drawdown_reduction"].map(format_percent),
        }
    )


def build_historical_stress_table(strategy_results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for strategy_name, result in strategy_results.items():
        stress_df = calculate_historical_stress_performance(
            result["portfolio_returns"],
            strategy_values=result.get("portfolio_values"),
        )
        if stress_df.empty:
            continue
        stress_df.insert(0, "strategy", strategy_name)
        rows.append(stress_df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def format_historical_stress_table(stress_df: pd.DataFrame) -> pd.DataFrame:
    if stress_df.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "Strategy": stress_df["strategy"],
            "Stress Period": stress_df["stress_period"],
            "Net Period Return": stress_df["period_return"].map(format_percent),
            "Max Drawdown": stress_df["max_drawdown"].map(format_percent),
            "Volatility": stress_df["volatility"].map(format_percent),
            "VaR 95": stress_df["var_95"].map(format_percent),
            "ES/CVaR 95": stress_df["es_95"].map(format_percent),
            "Max DD Duration": stress_df["max_drawdown_duration"].map(format_days),
            "Observations": stress_df["n_observations"],
            "Status": stress_df["status"],
        }
    )


def build_hypothetical_stress_dashboard_table(
    strategy_results: dict[str, dict],
    benchmark_name: str,
) -> pd.DataFrame:
    weights_by_strategy = {
        strategy_name: _latest_weights_from_result(result)
        for strategy_name, result in strategy_results.items()
    }
    weights_by_strategy = {
        strategy_name: weights
        for strategy_name, weights in weights_by_strategy.items()
        if isinstance(weights, pd.Series) and not weights.empty
    }
    benchmark_display_name = BenchmarkFactory.normalize_strategy_name(benchmark_name)
    return calculate_hypothetical_stress_table(weights_by_strategy, benchmark_display_name)


def format_hypothetical_stress_table(stress_df: pd.DataFrame) -> pd.DataFrame:
    if stress_df.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "Scenario": stress_df["scenario"],
            "Strategy": stress_df["strategy"],
            "Net Strategy Stress Return": stress_df["strategy_stress_return"].map(format_percent),
            "Net Benchmark Stress Return": stress_df["benchmark_stress_return"].map(format_percent),
            "Net Difference vs Benchmark": stress_df["difference_vs_benchmark"].map(format_percent),
            "Worst Strategy": stress_df["worst_strategy_under_scenario"],
            "Most Defensive Strategy": stress_df["most_defensive_strategy_under_scenario"],
        }
    )


def build_correlation_stress_table(
    strategy_results: dict[str, dict],
    returns_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for strategy_name, result in strategy_results.items():
        metrics = calculate_correlation_stress(_latest_weights_from_result(result), returns_df)
        rows.append({"strategy": strategy_name, **metrics})
    return pd.DataFrame(rows)


def format_correlation_stress_table(correlation_stress_df: pd.DataFrame) -> pd.DataFrame:
    if correlation_stress_df.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "Strategy": correlation_stress_df["strategy"],
            "Normal Volatility": correlation_stress_df["normal_volatility"].map(format_percent),
            "Correlation-Stressed Volatility": correlation_stress_df[
                "correlation_stressed_volatility"
            ].map(format_percent),
            "Volatility Increase": correlation_stress_df["volatility_increase"].map(format_percent),
            "Assumed Correlation": correlation_stress_df["stressed_correlation"].map(
                lambda value: format_decimal(value, digits=2)
            ),
        }
    )


def format_worst_period_table(worst_periods_df: pd.DataFrame) -> pd.DataFrame:
    if worst_periods_df.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "Window": worst_periods_df["window_days"].map(lambda value: f"{int(value)} days"),
            "Start": worst_periods_df["start_date"],
            "End": worst_periods_df["end_date"],
            "Net Period Return": worst_periods_df["period_return"].map(format_percent),
            "Max Drawdown": worst_periods_df["max_drawdown"].map(format_percent),
            "Observations": worst_periods_df["n_observations"],
        }
    )


def build_liquidity_dashboard_table(
    prices_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    weights_history: pd.DataFrame,
    portfolio_value: float,
) -> pd.DataFrame:
    target_weights, current_weights = _latest_rebalance_weight_pair(weights_history)
    return calculate_liquidity_diagnostics(
        prices=prices_df,
        volumes=volume_df,
        target_weights=target_weights,
        current_weights=current_weights,
        portfolio_value=portfolio_value,
        lookback_days=60,
    )


def format_liquidity_table(liquidity_df: pd.DataFrame) -> pd.DataFrame:
    if liquidity_df.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "Asset": liquidity_df["asset"],
            "Latest Price": liquidity_df["latest_price"].map(
                lambda value: format_decimal(value, 2)
            ),
            "Avg Daily Volume": liquidity_df["average_daily_volume"].map(
                lambda value: format_currency(value)
            ),
            "ADTV": liquidity_df["average_daily_traded_value"].map(format_currency),
            "Trade Value": liquidity_df["estimated_trade_value"].map(format_currency),
            "Participation Rate": liquidity_df["participation_rate"].map(format_percent),
            "Warning": liquidity_df["liquidity_warning"],
        }
    )


def _latest_weights_from_result(result: dict) -> pd.Series:
    latest_weights = result.get("latest_weights")
    if isinstance(latest_weights, pd.Series) and not latest_weights.empty:
        return latest_weights.astype(float)
    weights_history = result.get("weights_history")
    if isinstance(weights_history, pd.DataFrame) and not weights_history.empty:
        return weights_history.iloc[-1].astype(float)
    return pd.Series(dtype=float)


def _latest_rebalance_weight_pair(weights_history: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if not isinstance(weights_history, pd.DataFrame) or weights_history.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    target_weights = weights_history.iloc[-1].astype(float)
    if len(weights_history) >= 2:
        current_weights = weights_history.iloc[-2].astype(float)
    else:
        current_weights = pd.Series(0.0, index=target_weights.index, dtype=float)
    return target_weights, current_weights


def _is_finite(value: float) -> bool:
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


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
    volume_df = market_data.volume_df.reindex(columns=prices_df.columns).sort_index()
    return {
        "prices_df": prices_df,
        "volume_df": volume_df,
        "returns_df": returns_df,
        "data_quality_summary": data_quality_summary,
    }


def _find_optional_asset_returns(
    returns_df: pd.DataFrame,
    asset_keyword: str,
) -> pd.Series | None:
    matching_columns = [
        column for column in returns_df.columns if asset_keyword.upper() in str(column).upper()
    ]
    return returns_df[matching_columns[0]] if matching_columns else None


def build_market_regime_results(
    *,
    returns_df: pd.DataFrame,
    strategy_results: dict[str, dict],
    benchmark_strategy: str,
    lookback_vol: int,
    lookback_trend: int,
    lookback_corr: int,
    crisis_drawdown: float,
    stress_drawdown: float,
    use_lagged_decision_regime: bool,
    objective_metric: str,
    regime_method: str = "Rule-based",
    hmm_n_states: int = 4,
    hmm_min_train_size: int = 504,
    hmm_refit_frequency: int = 21,
    hmm_covariance_type: str = "diag",
    hmm_decision_lag: int = 1,
    hmm_feature_columns: list[str] | None = None,
) -> dict[str, object]:
    """Build rule-based or experimental HMM Phase 3B regime diagnostics."""
    benchmark_name = BenchmarkFactory.normalize_strategy_name(benchmark_strategy)
    benchmark_result = strategy_results.get(benchmark_name, {})
    benchmark_returns = benchmark_result.get("portfolio_returns")

    features = calculate_regime_features(
        returns=returns_df,
        benchmark_returns=benchmark_returns,
        gold_returns=_find_optional_asset_returns(returns_df, "GOLD"),
        silver_returns=_find_optional_asset_returns(returns_df, "SILVER"),
        lookback_vol=lookback_vol,
        lookback_trend=lookback_trend,
        lookback_corr=lookback_corr,
    )
    rule_based_regimes = classify_rule_based_regime(
        features,
        crisis_drawdown=crisis_drawdown,
        stress_drawdown=stress_drawdown,
    )
    rule_based_decision_regimes = lag_regime_labels(rule_based_regimes, lag=1)
    observed_regimes = rule_based_regimes
    decision_regimes = rule_based_decision_regimes
    analytics_regimes = decision_regimes if use_lagged_decision_regime else observed_regimes
    hmm_result: dict[str, object] | None = None
    hmm_error: str | None = None
    hmm_warning: str | None = None
    state_summary = pd.DataFrame()
    state_probabilities = pd.DataFrame()
    hmm_diagnostics = pd.DataFrame()
    method_comparison = None

    if regime_method != "Rule-based":
        if not HMM_AVAILABLE:
            hmm_error = "HMM regime detection requires the optional dependency `hmmlearn`."
        else:
            try:
                if regime_method == "HMM full-sample historical":
                    fitted = fit_hmm_full_sample(
                        features,
                        n_states=hmm_n_states,
                        feature_columns=hmm_feature_columns,
                        covariance_type=hmm_covariance_type,
                    )
                    mapped = map_hmm_states_to_regimes(
                        fitted["states"],
                        features,
                        n_states=hmm_n_states,
                    )
                    observed_regimes = mapped["regimes"].reindex(features.index).fillna("Unknown")
                    decision_regimes = lag_regime_labels(
                        observed_regimes,
                        lag=hmm_decision_lag,
                    )
                    analytics_regimes = observed_regimes
                    state_summary = mapped["state_summary"]
                    state_probabilities = fitted["state_probabilities"]
                    hmm_result = fitted
                    hmm_warning = "Historical visualization only; not trading-safe."
                elif regime_method == "HMM walk-forward experimental":
                    fitted = fit_hmm_walk_forward(
                        features,
                        n_states=hmm_n_states,
                        feature_columns=hmm_feature_columns,
                        min_train_size=hmm_min_train_size,
                        refit_frequency=hmm_refit_frequency,
                        covariance_type=hmm_covariance_type,
                        decision_lag=hmm_decision_lag,
                    )
                    observed_regimes = fitted["regimes"].reindex(features.index).fillna("Unknown")
                    decision_regimes = (
                        fitted["decision_regimes"].reindex(features.index).fillna("Unknown")
                    )
                    analytics_regimes = decision_regimes
                    state_summary = fitted["state_summary"]
                    state_probabilities = fitted["state_probabilities"]
                    hmm_diagnostics = fitted["diagnostics"]
                    hmm_result = fitted
                    hmm_warning = (
                        "Time-series-safe experimental regime inference using " "past data only."
                    )
                else:
                    raise ValueError(f"unsupported regime method '{regime_method}'")
            except Exception as exc:
                hmm_error = str(exc)
                observed_regimes = rule_based_regimes
                decision_regimes = rule_based_decision_regimes
                analytics_regimes = (
                    decision_regimes if use_lagged_decision_regime else observed_regimes
                )

    strategy_returns_dict = {
        strategy_name: result["portfolio_returns"]
        for strategy_name, result in strategy_results.items()
        if isinstance(result.get("portfolio_returns"), pd.Series)
    }
    regime_summary = calculate_strategy_regime_summary(
        strategy_returns_dict,
        analytics_regimes,
        benchmark_returns=benchmark_returns,
        objective=objective_metric,
    )
    observed_regime_summary = calculate_strategy_regime_summary({}, observed_regimes)
    transitions = calculate_regime_transitions(observed_regimes)
    state_table = calculate_regime_state_table(features, observed_regimes)
    if regime_method != "Rule-based" and hmm_error is None:
        method_comparison = compare_regime_methods(
            rule_based_regimes,
            observed_regimes,
        )
        comparison_labels = pd.concat(
            [
                rule_based_regimes.rename("rule_based_regime"),
                observed_regimes.rename("hmm_regime"),
            ],
            axis=1,
        )
        comparison_labels.insert(0, "date", comparison_labels.index)
        state_table = state_table.merge(
            comparison_labels.reset_index(drop=True),
            on="date",
            how="left",
        )

    latest_features = features.iloc[-1].to_dict() if not features.empty else {}
    current_observed_regime = (
        str(observed_regimes.iloc[-1]) if not observed_regimes.empty else "Unknown"
    )
    current_decision_regime = (
        str(decision_regimes.iloc[-1]) if not decision_regimes.empty else "Unknown"
    )

    return {
        "method": regime_method,
        "hmm_available": HMM_AVAILABLE,
        "hmm_error": hmm_error,
        "hmm_warning": hmm_warning,
        "hmm_result": hmm_result,
        "hmm_state_summary": state_summary,
        "hmm_state_probabilities": state_probabilities,
        "hmm_diagnostics": hmm_diagnostics,
        "method_comparison": method_comparison,
        "rule_based_regimes": rule_based_regimes,
        "features": features,
        "observed_regimes": observed_regimes,
        "decision_regimes": decision_regimes,
        "analytics_regimes": analytics_regimes,
        "use_lagged_decision_regime": use_lagged_decision_regime,
        "state_table": state_table,
        "performance": regime_summary["performance"],
        "regime_distribution": observed_regime_summary["regime_distribution"],
        "transitions": transitions,
        "latest_features": latest_features,
        "current_observed_regime": current_observed_regime,
        "current_decision_regime": current_decision_regime,
        "objective_metric": objective_metric,
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
    enable_regime_analytics: bool = False,
    regime_vol_lookback: int = 63,
    regime_trend_lookback: int = 126,
    regime_corr_lookback: int = 63,
    regime_crisis_drawdown: float = -0.15,
    regime_stress_drawdown: float = -0.08,
    use_lagged_decision_regime: bool = True,
    regime_objective_metric: str = "calmar",
    regime_method: str = "Rule-based",
    hmm_n_states: int = 4,
    hmm_min_train_size: int = 504,
    hmm_refit_frequency: int = 21,
    hmm_covariance_type: str = "diag",
    hmm_decision_lag: int = 1,
    hmm_feature_columns: list[str] | None = None,
    enable_adaptive_strategy: bool = False,
    adaptive_regime_source: str = "Rule-based lagged decision regime",
    adaptive_policy_preset: str = "Balanced default",
    adaptive_training_window: int = 252,
    adaptive_rebalance_frequency: str = "M",
    adaptive_show_policy_table: bool = True,
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
    active_risk_metrics_df = build_active_risk_metrics_table(
        strategy_results,
        performance_comparison_df,
        benchmark_strategy,
    )
    growth_curves = {
        strategy_name: result["portfolio_values"]
        for strategy_name, result in strategy_results.items()
    }
    drawdown_curves = {
        strategy_name: result["drawdown"] for strategy_name, result in strategy_results.items()
    }

    vol_target_results = None
    defensive_metadata = None
    defensive_returns = None
    resolved_defensive_result = None
    defensive_source, defensive_ticker = defensive_source_from_label(
        defensive_sleeve
    )
    if enable_vol_targeting:
        resolved_defensive_result = get_defensive_returns(
            index=returns_df.index,
            source=defensive_source,
            annual_rate=synthetic_annual_rate,
            defensive_ticker=defensive_ticker,
            fallback="synthetic",
        )
        defensive_returns = resolved_defensive_result.returns
        defensive_metadata = resolved_defensive_result.metadata
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
    market_regime_results = None
    if enable_regime_analytics or enable_adaptive_strategy:
        market_regime_results = build_market_regime_results(
            returns_df=returns_df,
            strategy_results=strategy_results,
            benchmark_strategy=benchmark_strategy,
            lookback_vol=regime_vol_lookback,
            lookback_trend=regime_trend_lookback,
            lookback_corr=regime_corr_lookback,
            crisis_drawdown=regime_crisis_drawdown,
            stress_drawdown=regime_stress_drawdown,
            use_lagged_decision_regime=use_lagged_decision_regime,
            objective_metric=regime_objective_metric,
            regime_method=regime_method,
            hmm_n_states=hmm_n_states,
            hmm_min_train_size=hmm_min_train_size,
            hmm_refit_frequency=hmm_refit_frequency,
            hmm_covariance_type=hmm_covariance_type,
            hmm_decision_lag=hmm_decision_lag,
            hmm_feature_columns=hmm_feature_columns,
        )

    adaptive_results = None
    adaptive_comparison_df = pd.DataFrame()
    adaptive_active_risk_df = pd.DataFrame()
    adaptive_regime_performance_df = pd.DataFrame()
    if enable_adaptive_strategy:
        adaptive_regime_payload = market_regime_results
        if adaptive_regime_source == "HMM walk-forward decision regime":
            if not HMM_AVAILABLE:
                raise ValueError(
                    "HMM adaptive allocation requires the optional dependency `hmmlearn`."
                )
            if (
                adaptive_regime_payload is None
                or adaptive_regime_payload.get("method") != "HMM walk-forward experimental"
            ):
                adaptive_regime_payload = build_market_regime_results(
                    returns_df=returns_df,
                    strategy_results=strategy_results,
                    benchmark_strategy=benchmark_strategy,
                    lookback_vol=regime_vol_lookback,
                    lookback_trend=regime_trend_lookback,
                    lookback_corr=regime_corr_lookback,
                    crisis_drawdown=regime_crisis_drawdown,
                    stress_drawdown=regime_stress_drawdown,
                    use_lagged_decision_regime=True,
                    objective_metric=regime_objective_metric,
                    regime_method="HMM walk-forward experimental",
                    hmm_n_states=hmm_n_states,
                    hmm_min_train_size=hmm_min_train_size,
                    hmm_refit_frequency=hmm_refit_frequency,
                    hmm_covariance_type=hmm_covariance_type,
                    hmm_decision_lag=hmm_decision_lag,
                    hmm_feature_columns=hmm_feature_columns,
                )
            if adaptive_regime_payload.get("hmm_error"):
                raise ValueError(str(adaptive_regime_payload["hmm_error"]))
            adaptive_regimes = adaptive_regime_payload["decision_regimes"]
            adaptive_use_lagged = False
            adaptive_method_name = "HMM walk-forward decision regimes"
        else:
            if adaptive_regime_payload is None:
                raise ValueError("rule-based regime features are unavailable")
            adaptive_regimes = adaptive_regime_payload["rule_based_regimes"]
            adaptive_use_lagged = True
            adaptive_method_name = "Rule-based observed regimes, lagged internally"

        adaptive_results = run_regime_adaptive_backtest(
            returns=returns_df,
            regimes=adaptive_regimes,
            defensive_returns=(
                None
                if (
                    defensive_source == "provided_series"
                    and defensive_metadata
                    and defensive_metadata.get("defensive_fallback_used")
                )
                else (resolved_defensive_result or defensive_returns)
            ),
            initial_value=initial_capital,
            training_window=adaptive_training_window,
            rebalance_frequency=adaptive_rebalance_frequency,
            transaction_cost_bps=base_bps,
            slippage_bps=slippage_bps,
            policy_map=get_policy_preset(adaptive_policy_preset),
            regime_method_name=adaptive_method_name,
            use_lagged_regimes=adaptive_use_lagged,
            defensive_source=defensive_source,
            defensive_annual_rate=synthetic_annual_rate,
            defensive_ticker=defensive_ticker,
            defensive_fallback="synthetic",
        )
        defensive_metadata = adaptive_results["defensive_metadata"]
        combined_results = dict(strategy_results)
        combined_results["Regime-Adaptive"] = adaptive_results
        adaptive_comparison_df = build_performance_comparison_table(combined_results)
        adaptive_comparison_df["total_turnover"] = [
            float(
                combined_results[strategy_name]
                .get("performance_metrics", {})
                .get("total_turnover", np.nan)
            )
            for strategy_name in adaptive_comparison_df.index
        ]
        adaptive_comparison_df["total_transaction_cost"] = [
            float(
                combined_results[strategy_name]
                .get("performance_metrics", {})
                .get("total_transaction_cost", np.nan)
            )
            for strategy_name in adaptive_comparison_df.index
        ]
        adaptive_active_risk_df = build_active_risk_metrics_table(
            combined_results,
            adaptive_comparison_df,
            benchmark_strategy,
        )
        benchmark_display_name = BenchmarkFactory.normalize_strategy_name(benchmark_strategy)
        benchmark_returns = strategy_results[benchmark_display_name]["portfolio_returns"]
        adaptive_regime_performance_df = calculate_regime_performance(
            adaptive_results["portfolio_returns"],
            adaptive_results["applied_regimes"],
            benchmark_returns=benchmark_returns,
        )

    return {
        "selected_tickers": selected_tickers,
        "initial_capital": initial_capital,
        "benchmark_strategy": benchmark_strategy,
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
        "active_risk_metrics_df": active_risk_metrics_df,
        "strategy_results": strategy_results,
        "growth_curves": growth_curves,
        "drawdown_curves": drawdown_curves,
        "metrics": metrics,
        "formatted_metrics": format_metric_cards(metrics),
        "vol_target_results": vol_target_results,
        "defensive_metadata": defensive_metadata,
        "market_regime_results": market_regime_results,
        "adaptive_results": adaptive_results,
        "adaptive_comparison_df": adaptive_comparison_df,
        "adaptive_active_risk_df": adaptive_active_risk_df,
        "adaptive_regime_performance_df": adaptive_regime_performance_df,
        "adaptive_regime_source": adaptive_regime_source,
        "adaptive_policy_preset": adaptive_policy_preset,
        "adaptive_strategy_label": adaptive_overlay_name(
            adaptive_regime_source,
            adaptive_policy_preset,
        ),
        "adaptive_show_policy_table": adaptive_show_policy_table,
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
    include_adaptive_strategies: bool = False,
    adaptive_regime_sources: list[str] | None = None,
    adaptive_policy_presets: list[str] | None = None,
    max_adaptive_configs: int = 6,
    adaptive_training_window: int = 252,
    adaptive_rebalance_frequency: str = "M",
    hmm_n_states: int = 4,
    hmm_min_train_size: int = 504,
    hmm_refit_frequency: int = 21,
    hmm_covariance_type: str = "diag",
    hmm_decision_lag: int = 1,
) -> dict[str, object]:
    threshold_values = parse_float_list(thresholds_text)
    validation_error = validate_sensitivity_inputs(threshold_values, max_runs)
    if validation_error is not None:
        raise ValueError(validation_error)

    market_context = load_market_context(selected_tickers, start_dt, end_dt)
    returns_df = market_context["returns_df"]

    defensive_input = None

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
        defensive_annual_rate=synthetic_annual_rate,
        defensive_fallback="synthetic",
    )
    objective_metric = SENSITIVITY_OBJECTIVE_MAP[objective_label]
    experiment_results_df = run_experiment_grid(
        returns_df=returns_df,
        config=experiment_config,
        defensive_returns=defensive_input,
        max_runs=max_runs,
    )
    adaptive_backtests: dict[str, dict[str, object]] = {}
    adaptive_warnings: list[str] = []
    if include_adaptive_strategies:
        adaptive_config = AdaptiveExperimentConfig(
            experiment_name="dashboard_adaptive_sensitivity",
            regime_sources=adaptive_regime_sources or ["rule_based_lagged"],
            policy_presets=adaptive_policy_presets
            or [
                "conservative",
                "balanced",
                "aggressive",
            ],
            training_windows=[int(adaptive_training_window)],
            defensive_assets=[defensive_sleeve],
            transaction_cost_bps=[float(base_bps)],
            slippage_bps=[float(slippage_bps)],
            rebalance_frequencies=[adaptive_rebalance_frequency],
            hmm_n_states=int(hmm_n_states),
            hmm_min_train_size=int(hmm_min_train_size),
            hmm_refit_frequency=int(hmm_refit_frequency),
            hmm_covariance_type=hmm_covariance_type,
            hmm_decision_lag=max(1, int(hmm_decision_lag)),
            initial_capital=float(initial_capital),
            defensive_annual_rate=float(synthetic_annual_rate),
            defensive_fallback="synthetic",
        )
        adaptive_grid_result = run_adaptive_experiment_grid(
            returns_df=returns_df,
            config=adaptive_config,
            defensive_returns=defensive_input,
            max_adaptive_configs=int(max_adaptive_configs),
        )
        adaptive_results_df = adaptive_grid_result["results"]
        adaptive_backtests = adaptive_grid_result["backtests"]
        adaptive_warnings = adaptive_grid_result["warnings"]
        experiment_results_df = pd.concat(
            [experiment_results_df, adaptive_results_df],
            ignore_index=True,
            sort=False,
        )

    return {
        "experiment_results_df": experiment_results_df,
        "objective_metric": objective_metric,
        "returns_df": returns_df,
        "defensive_returns": defensive_input,
        "train_window": experiment_config.train_window,
        "initial_capital": initial_capital,
        "include_adaptive_strategies": bool(include_adaptive_strategies),
        "adaptive_backtests": adaptive_backtests,
        "adaptive_warnings": adaptive_warnings,
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


def build_robustness_results(
    sensitivity_payload: dict[str, object],
    *,
    n_blocks: int,
    n_test_blocks: int,
    embargo_pct: float,
    purge_window: int,
    max_configs: int,
    objective_metric: str | None = None,
    include_adaptive_in_cpcv: bool = False,
    max_adaptive_configs: int = 2,
) -> dict[str, object]:
    """Run CPCV validation over the top single-objective sensitivity configurations."""
    selected_objective = objective_metric or sensitivity_payload.get("objective_metric") or "calmar"
    experiment_results = sensitivity_payload["experiment_results_df"]
    strategy_types = (
        experiment_results["strategy_type"].fillna("fixed")
        if "strategy_type" in experiment_results.columns
        else pd.Series("fixed", index=experiment_results.index)
    )
    fixed_results = experiment_results.loc[~strategy_types.eq("regime_adaptive")]
    top_configs = build_top_n_table(
        fixed_results,
        metric=str(selected_objective),
        n=max_configs,
    )
    if include_adaptive_in_cpcv:
        adaptive_results = experiment_results.loc[strategy_types.eq("regime_adaptive")]
        if "status" in adaptive_results.columns:
            adaptive_results = adaptive_results.loc[adaptive_results["status"].eq("success")]
        if not adaptive_results.empty:
            top_adaptive = build_top_n_table(
                adaptive_results,
                metric=str(selected_objective),
                n=int(max_adaptive_configs),
            )
            top_configs = pd.concat(
                [top_configs, top_adaptive],
                ignore_index=True,
                sort=False,
            )
    default_train_window = int(sensitivity_payload.get("train_window", 252))
    default_initial_capital = float(sensitivity_payload.get("initial_capital", 1_000_000.0))
    if "train_window" in top_configs.columns:
        top_configs["train_window"] = top_configs["train_window"].fillna(default_train_window)
    else:
        top_configs["train_window"] = default_train_window
    if "initial_capital" in top_configs.columns:
        top_configs["initial_capital"] = top_configs["initial_capital"].fillna(
            default_initial_capital
        )
    else:
        top_configs["initial_capital"] = default_initial_capital

    robustness_results = run_cpcv_validation(
        returns=sensitivity_payload["returns_df"],
        experiment_configs=top_configs,
        n_blocks=n_blocks,
        n_test_blocks=n_test_blocks,
        embargo_pct=embargo_pct,
        purge_window=purge_window,
        objective=str(selected_objective),
        max_configs=max_configs,
        max_adaptive_configs=(int(max_adaptive_configs) if include_adaptive_in_cpcv else None),
        defensive_returns=sensitivity_payload.get("defensive_returns"),
    )
    robustness_results["objective_metric"] = str(selected_objective)
    robustness_results["objective_label"] = research_objective_label(
        str(selected_objective)
    )
    robustness_results["include_adaptive_in_cpcv"] = bool(include_adaptive_in_cpcv)
    return robustness_results


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
    robustness_payload: dict[str, object] | None,
    selected_objective_label: str = DEFAULT_RESEARCH_OBJECTIVE,
) -> None:
    st.info(MODE_NOTES[RESEARCH_VIEW])
    st.caption(f"Current research objective: {selected_objective_label}")
    tabs = st.tabs(
        [
            "Portfolio Overview",
            "Backtest Results",
            "Risk & Allocation",
            "Phase 3B — Market Regimes",
            "Phase 3C — Adaptive Allocation",
            "Phase 3D — Adaptive Evaluation",
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
            st.info(
                "Enable regime analytics and run the portfolio analysis to view market regimes."
            )
        with tabs[4]:
            st.info(
                "Enable the adaptive regime strategy and run the portfolio analysis "
                "to view Phase 3C results."
            )
        with tabs[5]:
            render_adaptive_evaluation_content(
                sensitivity_payload=sensitivity_payload,
                robustness_payload=robustness_payload,
                portfolio_payload=None,
                selected_objective_label=selected_objective_label,
            )
        with tabs[6]:
            st.info("Trading activity diagnostics will appear here after a portfolio run.")
        with tabs[7]:
            st.info(
                "Enable Volatility Targeting and run the portfolio analysis to see overlay results."
            )
        with tabs[8]:
            if sensitivity_payload is None:
                st.info("Run Sensitivity Study to populate experiment outputs.")
                render_robustness_content(robustness_payload)
            else:
                render_sensitivity_content(sensitivity_payload, robustness_payload)
        return

    data_quality_summary = portfolio_payload["market_context"]["data_quality_summary"]
    prices_df = portfolio_payload["market_context"].get("prices_df", pd.DataFrame())
    volume_df = portfolio_payload["market_context"].get("volume_df", pd.DataFrame())
    returns_df = portfolio_payload["market_context"].get("returns_df", pd.DataFrame())
    initial_capital = float(portfolio_payload.get("initial_capital", 1_000_000.0))
    benchmark_strategy = portfolio_payload.get("benchmark_strategy", "Equal Weight")
    weights = portfolio_payload["weights"]
    portfolio_value = portfolio_payload["portfolio_value"]
    portfolio_returns = portfolio_payload["portfolio_returns"]
    performance_comparison_df = portfolio_payload["performance_comparison_df"]
    relative_performance_df = portfolio_payload["relative_performance_df"]
    active_risk_metrics_df = portfolio_payload.get("active_risk_metrics_df", pd.DataFrame())
    strategy_results = portfolio_payload.get("strategy_results", {})
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
    backtest_results = portfolio_payload["backtest_results"]
    vol_target_results = portfolio_payload["vol_target_results"]
    defensive_metadata = portfolio_payload["defensive_metadata"]
    market_regime_results = portfolio_payload.get("market_regime_results")
    adaptive_results = portfolio_payload.get("adaptive_results")
    adaptive_comparison_df = portfolio_payload.get(
        "adaptive_comparison_df",
        pd.DataFrame(),
    )
    adaptive_active_risk_df = portfolio_payload.get(
        "adaptive_active_risk_df",
        pd.DataFrame(),
    )
    adaptive_regime_performance_df = portfolio_payload.get(
        "adaptive_regime_performance_df",
        pd.DataFrame(),
    )
    selected_objective_metric = SENSITIVITY_OBJECTIVE_MAP.get(
        selected_objective_label,
        "calmar",
    )

    with tabs[0]:
        st.header("Portfolio Overview")
        render_data_quality_report(data_quality_summary)
        render_portfolio_summary(portfolio_payload["formatted_metrics"])
        render_risk_tracking_explainer()

        render_key_takeaways(
            performance_comparison_df,
            strategy_results,
            active_risk_metrics_df,
            selected_objective_label=selected_objective_label,
            selected_objective_metric=selected_objective_metric,
            volatility_targeting_enabled=vol_target_results is not None,
        )

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

        st.subheader("Benchmark & Active Risk Metrics")
        st.caption(
            "These metrics compare each strategy against the selected benchmark using aligned daily returns."
        )
        if active_risk_metrics_df.empty:
            st.info("Active-risk metrics are unavailable for this run.")
        else:
            st.dataframe(
                format_active_risk_metrics_table(active_risk_metrics_df),
                use_container_width=True,
            )
            render_active_risk_metric_help()

        st.subheader("Stress Period Benchmark Comparison")
        stress_benchmark_df = build_stress_period_benchmark_table(
            strategy_results,
            benchmark_strategy,
        )
        if stress_benchmark_df.empty:
            st.info("Stress-period benchmark comparison is unavailable for this date range.")
        else:
            st.caption(
                "Positive drawdown reduction means the strategy had a smaller drawdown than the benchmark."
            )
            st.dataframe(
                format_stress_period_benchmark_table(stress_benchmark_df),
                use_container_width=True,
            )

        with st.expander("Stress Testing & Scenario Analysis", expanded=False):
            st.caption(
                "Stress tests are diagnostics only. They do not change the single-objective strategy ranking."
            )
            historical_stress_df = build_historical_stress_table(strategy_results)
            st.subheader("Historical Stress Windows")
            if historical_stress_df.empty:
                st.info("No historical stress-window observations are available for this run.")
            else:
                st.dataframe(
                    format_historical_stress_table(historical_stress_df),
                    use_container_width=True,
                )

            st.subheader("Worst Rolling Periods")
            worst_periods_df = find_worst_periods(portfolio_returns)
            st.dataframe(format_worst_period_table(worst_periods_df), use_container_width=True)

            st.subheader("Hypothetical Shock Scenarios")
            hypothetical_stress_df = build_hypothetical_stress_dashboard_table(
                strategy_results,
                benchmark_strategy,
            )
            if hypothetical_stress_df.empty:
                st.info(
                    "Hypothetical stress results are unavailable because strategy weights are missing."
                )
            else:
                st.dataframe(
                    format_hypothetical_stress_table(hypothetical_stress_df),
                    use_container_width=True,
                )

            st.subheader("Correlation Shock Scenario")
            st.caption("Diagnostic scenario: all risky asset correlations rise to 0.8.")
            correlation_stress_df = build_correlation_stress_table(strategy_results, returns_df)
            st.dataframe(
                format_correlation_stress_table(correlation_stress_df),
                use_container_width=True,
            )

        st.dataframe(
            format_net_performance_table(performance_comparison_df),
            use_container_width=True,
        )
        st.dataframe(
            format_net_performance_table(relative_performance_df),
            use_container_width=True,
        )

    with tabs[2]:
        st.header("Risk & Allocation")
        st.subheader("VaR / Expected Shortfall")
        st.caption(
            "VaR answers: How bad can things get under normal historical conditions? "
            "Expected Shortfall answers: If losses exceed VaR, what is the average loss in the tail?"
        )
        var_control_col1, var_control_col2, var_control_col3 = st.columns(3)
        with var_control_col1:
            var_confidence_label = st.selectbox(
                "VaR / ES Confidence Level",
                ["95%", "97.5%", "99%"],
                index=0,
                key="tab_var_confidence",
                help="Confidence level used for left-tail historical VaR and ES/CVaR.",
            )
        with var_control_col2:
            holding_period_days = st.selectbox(
                "Holding Period",
                [1, 5, 10, 21],
                index=0,
                format_func=lambda value: f"{value} day" if value == 1 else f"{value} days",
                key="tab_var_holding_period",
                help="Holding-period scaling uses square-root-of-time.",
            )
        with var_control_col3:
            value_basis = st.selectbox(
                "Portfolio Value Basis",
                ["Latest portfolio value", "Initial capital"],
                index=0,
                key="tab_var_value_basis",
                help="Basis used to convert VaR/ES percentages into currency amounts.",
            )
        confidence_level = {"95%": 0.95, "97.5%": 0.975, "99%": 0.99}[var_confidence_label]
        var_es_df = build_var_es_metrics_table(
            strategy_results,
            confidence_level=confidence_level,
            holding_period_days=int(holding_period_days),
            value_basis=value_basis,
            initial_capital=initial_capital,
        )
        st.dataframe(format_var_es_metrics_table(var_es_df), use_container_width=True)
        st.caption(
            "Historical VaR is based on observed past returns. It is not a guarantee and may "
            "underestimate losses during unseen stress events."
        )

        st.subheader("VaR Backtesting / Exceptions")
        st.caption(
            "This is a practical diagnostic, not full regulatory VaR validation. Rolling VaR uses "
            "lagged thresholds when enough observations are available."
        )
        st.dataframe(format_var_exception_table(var_es_df), use_container_width=True)

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
        render_market_regime_content(
            market_regime_results,
            selected_objective_label=selected_objective_label,
            selected_objective_metric=selected_objective_metric,
        )

    with tabs[4]:
        render_adaptive_allocation_content(
            adaptive_results=adaptive_results,
            strategy_results=strategy_results,
            comparison_df=adaptive_comparison_df,
            active_risk_df=adaptive_active_risk_df,
            regime_performance_df=adaptive_regime_performance_df,
            selected_objective_label=selected_objective_label,
            selected_objective_metric=selected_objective_metric,
            show_policy_table=bool(portfolio_payload.get("adaptive_show_policy_table", True)),
        )

    with tabs[5]:
        render_adaptive_evaluation_content(
            sensitivity_payload=sensitivity_payload,
            robustness_payload=robustness_payload,
            portfolio_payload=portfolio_payload,
            selected_objective_label=selected_objective_label,
        )

    with tabs[6]:
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

        cost_col1, cost_col2, cost_col3 = st.columns(3)
        cost_col1.metric(
            "Gross Final Value",
            f"{cost_drag_summary['gross_final_value']:.2f}",
        )
        cost_col2.metric(
            "Net Final Value",
            f"{cost_drag_summary['net_final_value']:.2f}",
        )
        cost_col3.metric(
            "Cost Drag",
            f"{cost_drag_summary['cost_drag']:.2f} ({cost_drag_summary['cost_drag_pct']:.2%})",
        )

        st.subheader("Liquidity Diagnostics")
        st.caption(
            "Turnover tells us how much we trade. Liquidity diagnostics tell us whether those "
            "trades are large relative to the asset's normal market activity."
        )
        if volume_df.empty:
            st.info("Volume data is unavailable for liquidity diagnostics in this run.")
        else:
            liquidity_df = build_liquidity_dashboard_table(
                prices_df=prices_df,
                volume_df=volume_df,
                weights_history=backtest_results["weights_history"],
                portfolio_value=float(portfolio_value.iloc[-1]),
            )
            if liquidity_df.empty:
                st.info("Volume data is unavailable for liquidity diagnostics in this run.")
            else:
                liquidity_summary = summarize_liquidity_diagnostics(liquidity_df)
                liq_col1, liq_col2, liq_col3 = st.columns(3)
                liq_col1.metric(
                    "Average Participation",
                    format_percent(liquidity_summary["average_participation_rate"]),
                )
                liq_col2.metric(
                    "Max Participation",
                    format_percent(liquidity_summary["max_participation_rate"]),
                )
                liq_col3.metric("Minimum ADTV", format_currency(liquidity_summary["min_adtv"]))
                if liquidity_summary["num_high_risk_assets"] > 0:
                    st.warning("Large trades may create market impact for some assets.")
                least_liquid = liquidity_df.sort_values(
                    "average_daily_traded_value",
                    ascending=True,
                ).head(10)
                highest_participation = liquidity_df.sort_values(
                    "participation_rate",
                    ascending=False,
                ).head(10)
                liq_table_col1, liq_table_col2 = st.columns(2)
                with liq_table_col1:
                    st.write("Least liquid assets by ADTV")
                    st.dataframe(format_liquidity_table(least_liquid), use_container_width=True)
                with liq_table_col2:
                    st.write("Highest participation-rate assets")
                    st.dataframe(
                        format_liquidity_table(highest_participation),
                        use_container_width=True,
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

    with tabs[7]:
        st.header("Volatility Targeting")
        if vol_target_results is None or defensive_metadata is None:
            st.info(
                "Enable Volatility Targeting and run the portfolio analysis to view overlay outputs."
            )
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
                    {
                        "Field": list(defensive_metadata.keys()),
                        "Value": list(defensive_metadata.values()),
                    }
                ),
                use_container_width=True,
            )

            st.subheader("Overlay Diagnostics")
            st.dataframe(diagnostics_df.tail(100), use_container_width=True)
            st.info(
                "The existing volatility targeting overlay adjusts risky exposure using "
                "lagged realized volatility. Phase 3B extends this idea into a broader "
                "regime analytics layer using volatility, drawdown, trend, momentum, "
                "and correlation features."
            )

    with tabs[8]:
        if sensitivity_payload is None:
            st.info("Run Sensitivity Study to populate experiment outputs.")
            render_robustness_content(robustness_payload)
        else:
            render_sensitivity_content(sensitivity_payload, robustness_payload)


def render_market_regime_content(
    regime_payload: dict[str, object] | None,
    *,
    selected_objective_label: str,
    selected_objective_metric: str,
    show_debug: bool = False,
) -> None:
    st.header("Phase 3B — Market Regime Detection")
    st.info(
        "The rule-based method is the explainable baseline. Phase 3B.2 adds "
        "experimental Gaussian HMM regimes using the same volatility, drawdown, "
        "trend, momentum, and correlation features. Phase 3C uses lagged decision "
        "regimes for adaptive allocator and risk-control switching."
    )
    if regime_payload is None:
        st.info("Enable regime analytics and run the portfolio analysis to populate this section.")
        return

    regime_method = str(regime_payload.get("method", "Rule-based"))
    st.write(f"Current regime method: {regime_method}")
    if regime_payload.get("hmm_error"):
        st.warning(str(regime_payload["hmm_error"]))
        st.caption(
            "Showing the rule-based baseline because the selected HMM method is unavailable."
        )
    elif regime_payload.get("hmm_warning"):
        if regime_method == "HMM full-sample historical":
            st.warning(str(regime_payload["hmm_warning"]))
        else:
            st.info(str(regime_payload["hmm_warning"]))

    latest = regime_payload.get("latest_features", {})
    transitions = regime_payload["transitions"]
    current_duration = transitions.get("current_regime_duration", 0)
    is_hmm = regime_method != "Rule-based" and not regime_payload.get("hmm_error")
    is_walk_forward = regime_method == "HMM walk-forward experimental"

    regime_col1, regime_col2, regime_col3 = st.columns(3)
    regime_col1.metric(
        "Current HMM regime" if is_hmm else "Current observed regime",
        regime_payload.get("current_observed_regime", "Unknown"),
    )
    regime_col2.metric(
        "Current HMM decision regime" if is_walk_forward else "Current decision regime",
        (
            regime_payload.get("current_decision_regime", "Unknown")
            if is_walk_forward or not is_hmm
            else "Historical only"
        ),
    )
    regime_col3.metric("Current regime duration", f"{current_duration} days")

    feature_col1, feature_col2, feature_col3, feature_col4 = st.columns(4)
    feature_col1.metric(
        "Latest rolling volatility",
        format_percent(latest.get("rolling_volatility")),
    )
    feature_col2.metric(
        "Latest drawdown",
        format_percent(latest.get("rolling_drawdown")),
    )
    feature_col3.metric(
        "Latest trend",
        format_percent(latest.get("trend_126d")),
    )
    feature_col4.metric(
        "Latest average correlation",
        format_decimal(latest.get("average_correlation")),
    )

    timeline_col1, timeline_col2 = st.columns(2)
    with timeline_col1:
        st.plotly_chart(
            plot_regime_series(
                regime_payload["observed_regimes"],
                title="Observed Market Regime Timeline",
            ),
            use_container_width=True,
        )
    with timeline_col2:
        comparison_timeline = regime_payload["decision_regimes"]
        comparison_title = "Decision Regime Timeline (Lagged One Day)"
        if regime_method == "HMM full-sample historical":
            comparison_timeline = regime_payload["rule_based_regimes"]
            comparison_title = "Rule-Based Baseline Timeline"
        st.plotly_chart(
            plot_regime_series(
                comparison_timeline,
                title=comparison_title,
            ),
            use_container_width=True,
        )

    state_summary = regime_payload.get("hmm_state_summary", pd.DataFrame())
    state_probabilities = regime_payload.get(
        "hmm_state_probabilities",
        pd.DataFrame(),
    )
    if is_hmm:
        st.subheader("HMM State Mapping")
        st.dataframe(state_summary, use_container_width=True)
        hmm_result = regime_payload.get("hmm_result") or {}
        used_columns = hmm_result.get("used_columns", [])
        if used_columns:
            st.caption(f"HMM features: {', '.join(used_columns)}")
        if regime_method == "HMM full-sample historical":
            st.caption(
                "Converged: "
                f"{hmm_result.get('converged', False)} | "
                f"Log likelihood: {format_decimal(hmm_result.get('log_likelihood'))}"
            )
        if show_debug and not state_probabilities.empty:
            st.subheader("HMM State Probabilities")
            st.plotly_chart(
                plot_hmm_state_probabilities(state_probabilities),
                use_container_width=True,
            )
            st.dataframe(
                state_probabilities.tail(100),
                use_container_width=True,
            )
        hmm_diagnostics = regime_payload.get("hmm_diagnostics", pd.DataFrame())
        if show_debug and is_walk_forward and not hmm_diagnostics.empty:
            st.subheader("HMM Walk-Forward Refit Diagnostics")
            st.dataframe(hmm_diagnostics, use_container_width=True)

    if show_debug:
        st.subheader("Regime State Table")
        st.dataframe(regime_payload["state_table"].tail(500), use_container_width=True)

    st.subheader("Regime Distribution")
    distribution = regime_payload["regime_distribution"].copy()
    preferred_order = [
        "Risk-On",
        "Calm",
        "Normal",
        "Stress",
        "Risk-Off",
        "Crisis",
        "Unknown",
    ]
    if not distribution.empty:
        distribution["_order"] = distribution["regime"].map(
            {regime: position for position, regime in enumerate(preferred_order)}
        )
        distribution = (
            distribution.sort_values(
                ["_order", "regime"],
                na_position="last",
            )
            .drop(columns="_order")
            .reset_index(drop=True)
        )
    st.dataframe(distribution, use_container_width=True)

    method_comparison = regime_payload.get("method_comparison")
    if is_hmm and method_comparison is not None:
        st.subheader("Rule-Based vs HMM Comparison")
        st.metric(
            "Agreement rate",
            format_percent(method_comparison["agreement_rate"]),
        )
        comparison_col1, comparison_col2 = st.columns(2)
        with comparison_col1:
            st.write("Regime crosstab")
            st.dataframe(
                method_comparison["crosstab"],
                use_container_width=True,
            )
        with comparison_col2:
            st.write("Regime counts by method")
            st.dataframe(
                method_comparison["regime_counts_by_method"],
                use_container_width=True,
            )
        if show_debug:
            st.write("Timeline comparison")
            st.dataframe(
                method_comparison["comparison_table"].tail(500),
                use_container_width=True,
            )
            with st.expander("Dates of disagreement", expanded=False):
                st.dataframe(
                    method_comparison["dates_of_disagreement"],
                    use_container_width=True,
                )

    st.subheader("Strategy Performance by Regime")
    performance = regime_payload["performance"]
    st.dataframe(
        format_net_performance_table(performance),
        use_container_width=True,
    )

    selection = select_best_strategy_by_regime(
        performance,
        objective=selected_objective_metric,
    )
    active_objective = selection["objective"] or selected_objective_metric
    active_objective_label = research_objective_label(active_objective)
    st.write(f"Current regime-performance objective: {selected_objective_label}")
    if selection["fallback_used"]:
        st.caption(
            f"{selected_objective_label} is unavailable in the regime summary; "
            f"best-strategy selection uses {active_objective_label}."
        )
    st.subheader("Best Strategy by Regime")
    st.dataframe(selection["table"], use_container_width=True)

    st.subheader("Regime Transition Diagnostics")
    transition_col1, transition_col2 = st.columns(2)
    with transition_col1:
        st.write("Transition count matrix")
        st.dataframe(
            transitions["transition_count_matrix"],
            use_container_width=True,
        )
    with transition_col2:
        st.write("Transition probability matrix")
        st.dataframe(
            transitions["transition_probability_matrix"],
            use_container_width=True,
        )
    st.write("Average duration by regime")
    st.dataframe(transitions["average_duration"], use_container_width=True)
    st.caption(
        f"Current observed regime: {transitions.get('current_regime', 'Unknown')} | "
        f"Current duration: {current_duration} days"
    )


def render_adaptive_allocation_content(
    *,
    adaptive_results: dict[str, object] | None,
    strategy_results: dict[str, dict],
    comparison_df: pd.DataFrame,
    active_risk_df: pd.DataFrame,
    regime_performance_df: pd.DataFrame,
    selected_objective_label: str,
    selected_objective_metric: str,
    show_policy_table: bool,
    show_debug: bool = False,
) -> None:
    st.header("Phase 3C — Regime-Aware Adaptive Allocation")
    st.warning("Uses lagged regimes only.")
    st.warning("Full-sample HMM is not allowed for adaptive backtests.")
    st.warning("This is research validation, not live trading advice.")

    if adaptive_results is None:
        st.info(
            "Enable the adaptive regime strategy and run the portfolio analysis "
            "to populate this section."
        )
        return

    metrics = adaptive_results["performance_metrics"]
    defensive_metadata = adaptive_results.get("defensive_metadata", {})
    st.caption(
        f"Defensive sleeve: {format_defensive_source(defensive_metadata)}"
    )
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Adaptive Net CAGR", format_percent(metrics.get("cagr")))
    metric_col2.metric("Adaptive Net Sharpe", format_decimal(metrics.get("sharpe")))
    metric_col3.metric("Adaptive Net Calmar", format_decimal(metrics.get("calmar")))
    metric_col4.metric(
        "Adaptive Max Drawdown",
        format_percent(metrics.get("max_drawdown")),
    )

    if show_policy_table:
        st.subheader("Regime Policy Table")
        st.dataframe(adaptive_results["policy_table"], use_container_width=True)

    growth_curves = {
        strategy_name: result["portfolio_values"]
        for strategy_name, result in strategy_results.items()
    }
    growth_curves["Regime-Adaptive"] = adaptive_results["portfolio_values"]
    drawdown_curves = {
        strategy_name: result["drawdown"] for strategy_name, result in strategy_results.items()
    }
    drawdown_curves["Regime-Adaptive"] = adaptive_results["drawdown"]

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(
            plot_performance_curves(growth_curves),
            use_container_width=True,
        )
    with chart_col2:
        st.plotly_chart(
            plot_drawdown_curves(drawdown_curves),
            use_container_width=True,
        )

    st.subheader("Adaptive Strategy Comparison")
    st.write(f"Current comparison objective: {selected_objective_label}")
    ranked_comparison = comparison_df.copy()
    if not ranked_comparison.empty and selected_objective_metric in ranked_comparison.columns:
        objective_values = pd.to_numeric(
            ranked_comparison[selected_objective_metric],
            errors="coerce",
        )
        ranked_comparison["objective_rank"] = objective_values.rank(
            method="min",
            ascending=selected_objective_metric in LOWER_IS_BETTER_METRICS,
        )
        ranked_comparison = ranked_comparison.sort_values(
            "objective_rank",
            kind="mergesort",
        )
    st.dataframe(
        format_net_performance_table(ranked_comparison),
        use_container_width=True,
    )

    if not active_risk_df.empty:
        st.subheader("Benchmark-Relative Adaptive Diagnostics")
        st.dataframe(
            active_risk_df[active_risk_df["strategy"].astype(str).eq("Regime-Adaptive")],
            use_container_width=True,
        )

    st.subheader("Adaptive Performance by Decision Regime")
    st.dataframe(regime_performance_df, use_container_width=True)

    if show_debug:
        st.subheader("Adaptive Decision Diagnostics")
        st.dataframe(
            adaptive_results["diagnostics"].tail(500),
            use_container_width=True,
        )

        st.subheader("Adaptive Weight History")
        st.dataframe(adaptive_results["weights"].tail(200), use_container_width=True)


def render_adaptive_evaluation_content(
    *,
    sensitivity_payload: dict[str, object] | None,
    robustness_payload: dict[str, object] | None,
    portfolio_payload: dict[str, object] | None,
    selected_objective_label: str,
) -> None:
    """Render Phase 3D sensitivity, attribution, stress, and CPCV evaluation."""
    st.header("Phase 3D — Adaptive Strategy Evaluation")
    st.warning(
        "Adaptive strategy evaluation uses lagged decision regimes. Full-sample "
        "HMM is excluded from trading-safe adaptive experiments."
    )
    if sensitivity_payload is None:
        st.info(
            "Enable adaptive strategies in the sensitivity controls and run the "
            "sensitivity study to populate Phase 3D."
        )
        return

    results = sensitivity_payload["experiment_results_df"]
    strategy_types = (
        results["strategy_type"].fillna("fixed")
        if "strategy_type" in results.columns
        else pd.Series("fixed", index=results.index)
    )
    adaptive_results = results.loc[strategy_types.eq("regime_adaptive")].copy()
    fixed_results = results.loc[~strategy_types.eq("regime_adaptive")].copy()
    successful_adaptive = adaptive_results.loc[
        adaptive_results.get(
            "status",
            pd.Series("success", index=adaptive_results.index),
        ).eq("success")
    ].copy()

    for warning in sensitivity_payload.get("adaptive_warnings", []):
        st.warning(str(warning))
    if adaptive_results.empty:
        st.info("This sensitivity run did not include adaptive regime strategies.")
        return
    if successful_adaptive.empty:
        st.warning("No adaptive configuration completed successfully.")
        render_dataframe_download(
            "Download Full Sensitivity Table",
            adaptive_results,
            "adaptive_sensitivity_results.csv",
            key="download_failed_adaptive_sensitivity",
        )
        return

    objective_metric = sensitivity_payload.get(
        "objective_metric",
        SENSITIVITY_OBJECTIVE_MAP.get(selected_objective_label, "calmar"),
    )
    successful_adaptive = successful_adaptive.sort_values(
        objective_metric,
        ascending=objective_metric in LOWER_IS_BETTER_METRICS,
        kind="mergesort",
    )
    best_row = successful_adaptive.iloc[0]

    hero1, hero2, hero3, hero4 = st.columns(4)
    hero1.metric("Best Adaptive Config", str(best_row["strategy"]))
    hero2.metric(
        f"Selected Objective ({selected_objective_label})",
        format_decimal(best_row.get(objective_metric)),
    )
    hero3.metric(
        "Average Risky Exposure",
        format_percent(best_row.get("average_risky_exposure")),
    )
    hero4.metric(
        "Policy Switches",
        str(int(best_row.get("number_of_policy_switches", 0))),
    )

    st.subheader("Adaptive Strategy Sensitivity Results")
    st.dataframe(
        format_net_performance_table(adaptive_results.head(20)),
        use_container_width=True,
    )
    render_dataframe_download(
        "Download Full Sensitivity Table",
        adaptive_results,
        "adaptive_sensitivity_results.csv",
        key="download_adaptive_sensitivity",
    )

    comparison = compare_adaptive_vs_fixed(
        successful_adaptive,
        fixed_results,
        objective=str(objective_metric),
    )
    st.subheader("Adaptive vs Fixed Strategy Comparison")
    st.info(str(comparison["interpretation"]))
    st.dataframe(
        format_net_performance_table(pd.DataFrame([comparison])),
        use_container_width=True,
    )

    config_id = str(best_row.get("config_id"))
    adaptive_backtest = sensitivity_payload.get("adaptive_backtests", {}).get(config_id)
    if adaptive_backtest is not None:
        benchmark_returns = None
        if portfolio_payload is not None:
            benchmark_name = BenchmarkFactory.normalize_strategy_name(
                portfolio_payload.get("benchmark_strategy", "Equal Weight")
            )
            benchmark_result = portfolio_payload.get("strategy_results", {}).get(
                benchmark_name,
                {},
            )
            benchmark_returns = benchmark_result.get("portfolio_returns")
        attribution = build_adaptive_attribution(
            adaptive_backtest,
            benchmark_returns=benchmark_returns,
        )
        st.subheader("Adaptive Strategy Attribution")
        exposure_history = attribution["exposure_history"]
        if not exposure_history.empty:
            exposure_series = exposure_history.set_index("date")["risky_exposure"]
            regime_series = exposure_history.set_index("date")["regime"]
            attribution_col1, attribution_col2 = st.columns(2)
            with attribution_col1:
                st.plotly_chart(
                    plot_exposure_series(exposure_series),
                    use_container_width=True,
                )
            with attribution_col2:
                st.plotly_chart(
                    plot_defensive_allocation(exposure_series),
                    use_container_width=True,
                )
            st.plotly_chart(
                plot_regime_series(
                    regime_series,
                    title="Adaptive Decision Regime Timeline",
                ),
                use_container_width=True,
            )
            regime_exposure = (
                exposure_history.groupby("regime", dropna=False)
                .agg(
                    average_risky_exposure=("risky_exposure", "mean"),
                    minimum_risky_exposure=("risky_exposure", "min"),
                    average_defensive_weight=("defensive_weight", "mean"),
                    maximum_defensive_weight=("defensive_weight", "max"),
                    number_of_days=("regime", "size"),
                )
                .reset_index()
            )
            st.write("Exposure by regime")
            st.dataframe(regime_exposure, use_container_width=True)

        attribution_col3, attribution_col4 = st.columns(2)
        with attribution_col3:
            st.write("Regime distribution")
            st.dataframe(
                attribution["regime_distribution"],
                use_container_width=True,
            )
            st.write("Allocator usage")
            st.dataframe(
                attribution["allocator_usage"],
                use_container_width=True,
            )
        with attribution_col4:
            st.write("Policy usage")
            st.dataframe(attribution["policy_usage"], use_container_width=True)
            st.write("Covariance method usage")
            st.dataframe(
                attribution["covariance_usage"],
                use_container_width=True,
            )

        st.write("Policy switches over time")
        st.dataframe(attribution["policy_switches"], use_container_width=True)
        st.write("Performance contribution by regime")
        st.dataframe(
            attribution["regime_performance"],
            use_container_width=True,
        )

        if portfolio_payload is not None:
            stress_table = build_adaptive_stress_comparison(
                {
                    **adaptive_backtest,
                    "strategy": best_row["strategy"],
                },
                portfolio_payload.get("strategy_results", {}),
                BenchmarkFactory.normalize_strategy_name(
                    portfolio_payload.get(
                        "benchmark_strategy",
                        "Equal Weight",
                    )
                ),
                objective=str(objective_metric),
            )
            st.subheader("Adaptive Stress-Period Performance")
            if stress_table.empty:
                st.info(
                    "Stress-period comparison requires a completed portfolio "
                    "analysis with fixed-strategy results."
                )
            else:
                st.dataframe(
                    format_net_performance_table(stress_table),
                    use_container_width=True,
                )

    st.subheader("Adaptive CPCV Robustness")
    if robustness_payload is None or not robustness_payload.get("include_adaptive_in_cpcv", False):
        st.info(
            "Enable adaptive strategies in CPCV and run robustness validation "
            "to compare adaptive fold stability."
        )
    else:
        ranking = robustness_payload.get("robustness_ranking", pd.DataFrame()).copy()
        if not ranking.empty:
            ranking.insert(0, "overall_rank", np.arange(1, len(ranking) + 1))
        adaptive_ranking = (
            ranking.loc[
                ranking.get(
                    "strategy_type",
                    pd.Series("fixed", index=ranking.index),
                ).eq("regime_adaptive")
            ]
            if not ranking.empty
            else pd.DataFrame()
        )
        if adaptive_ranking.empty:
            st.warning(
                "Adaptive CPCV was enabled, but no adaptive configuration "
                "produced a complete robustness ranking."
            )
        else:
            st.write(
                f"Current adaptive robustness objective: "
                f"{robustness_payload.get('objective_label', selected_objective_label)}"
            )
            display_columns = [
                column
                for column in [
                    "overall_rank",
                    "strategy",
                    "regime_source",
                    "policy_preset",
                    "objective_median",
                    "objective_worst",
                    "stability_score",
                    "robustness_score",
                ]
                if column in adaptive_ranking.columns
            ]
            st.dataframe(
                adaptive_ranking[display_columns],
                use_container_width=True,
            )


def render_sensitivity_content(
    sensitivity_payload: dict[str, object],
    robustness_payload: dict[str, object] | None = None,
) -> None:
    st.header("Experiment Sensitivity")

    experiment_results_df = sensitivity_payload["experiment_results_df"]
    objective_metric = sensitivity_payload["objective_metric"]
    experiment_summary_df = sensitivity_payload["experiment_summary_df"]
    top_experiments_df = sensitivity_payload["top_experiments_df"]
    parameter_sensitivity_df = sensitivity_payload["parameter_sensitivity_df"]
    objective_label = research_objective_label(objective_metric)

    st.info(
        "Sensitivity analysis is descriptive, not a weighted decision model. It shows how the "
        "selected objective changes across strategy, covariance method, rebalance mode, "
        "transaction cost, slippage, volatility targeting, and other parameters."
    )
    st.metric("Current ranking objective", objective_label)

    st.subheader("Experiment Results")
    st.dataframe(
        format_net_performance_table(experiment_summary_df),
        use_container_width=True,
    )
    render_dataframe_download(
        "Download Full Sensitivity Table",
        experiment_results_df,
        "full_sensitivity_results.csv",
        key="download_full_sensitivity",
    )

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
    st.dataframe(
        format_net_performance_table(top_experiments_df),
        use_container_width=True,
    )

    st.subheader("Parameter Sensitivity")
    st.dataframe(parameter_sensitivity_df, use_container_width=True)
    render_robustness_content(robustness_payload, show_raw=False)


def render_robustness_content(
    robustness_payload: dict[str, object] | None,
    *,
    show_raw: bool = False,
) -> None:
    with st.expander(
        "Phase 3A — Robustness Validation / CPCV-Style Validation",
        expanded=robustness_payload is not None,
    ):
        st.info(
            "This is a time-series-safe robustness validation layer. It tests whether "
            "strategy configurations perform consistently across different historical "
            "partitions. Purge and embargo reduce leakage around test periods."
        )
        st.warning("This is CPCV-style research validation, not a guarantee of future performance.")

        if robustness_payload is None:
            st.caption(
                "Enable robustness validation in the sidebar and run it after a sensitivity study."
            )
            return

        split_diagnostics = robustness_payload["split_diagnostics"]
        fold_results = robustness_payload["fold_results"]
        summary_table = robustness_payload["summary_table"]
        robustness_ranking = robustness_payload["robustness_ranking"]
        objective_label = robustness_payload.get(
            "objective_label",
            DEFAULT_RESEARCH_OBJECTIVE,
        )

        split_count = len(split_diagnostics)
        average_train = (
            float(split_diagnostics["n_train"].mean())
            if split_count and "n_train" in split_diagnostics
            else 0.0
        )
        average_test = (
            float(split_diagnostics["n_test"].mean())
            if split_count and "n_test" in split_diagnostics
            else 0.0
        )

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Number of splits", split_count)
        metric_col2.metric("Average train size", f"{average_train:,.1f}")
        metric_col3.metric("Average test size", f"{average_test:,.1f}")
        st.write(f"Current robustness objective: {objective_label}")

        if show_raw:
            st.subheader("Split Diagnostics")
            st.dataframe(split_diagnostics, use_container_width=True)
            st.subheader("Fold-level Metrics")
            st.dataframe(fold_results, use_container_width=True)
        else:
            download_col1, download_col2 = st.columns(2)
            with download_col1:
                render_dataframe_download(
                    "Download Full CPCV Split Table",
                    split_diagnostics,
                    "cpcv_split_diagnostics.csv",
                    key="download_cpcv_splits",
                )
            with download_col2:
                render_dataframe_download(
                    "Download Full CPCV Fold Table",
                    fold_results,
                    "cpcv_fold_results.csv",
                    key="download_cpcv_folds",
                )
        st.subheader("Robustness Summary")
        st.dataframe(summary_table, use_container_width=True)
        st.subheader("Robustness Ranking")
        st.dataframe(robustness_ranking, use_container_width=True)


def build_selection_candidate_metrics(
    portfolio_payload: dict[str, object],
    *,
    base_bps: float,
    slippage_bps: float,
) -> pd.DataFrame:
    """Translate current dashboard results into the selection engine schema."""

    rows: list[dict[str, object]] = []
    n_observations = len(portfolio_payload.get("portfolio_returns", pd.Series(dtype=float)))
    for strategy_name, result in portfolio_payload.get("strategy_results", {}).items():
        rows.append(
            {
                "strategy": str(strategy_name),
                "strategy_type": "fixed",
                "return_basis": "net",
                "n_observations": n_observations,
                "total_cost_bps": float(base_bps) + float(slippage_bps),
                **_performance_row_from_result(result),
            }
        )

    adaptive_results = portfolio_payload.get("adaptive_results")
    if adaptive_results is not None:
        defensive_metadata = adaptive_results.get("defensive_metadata", {})
        rows.append(
            {
                "strategy": portfolio_payload.get(
                    "adaptive_strategy_label",
                    adaptive_overlay_name(
                        str(portfolio_payload.get("adaptive_regime_source", "")),
                        str(portfolio_payload.get("adaptive_policy_preset", "")),
                    ),
                ),
                "strategy_type": "regime_adaptive",
                "regime_source": (
                    "hmm_walk_forward"
                    if "hmm" in str(portfolio_payload.get("adaptive_regime_source", "")).lower()
                    else "rule_based_lagged"
                ),
                "policy_preset": str(
                    portfolio_payload.get("adaptive_policy_preset", "Conservative")
                ).lower(),
                "return_basis": "net",
                "n_observations": n_observations,
                "total_cost_bps": float(base_bps) + float(slippage_bps),
                **defensive_metadata,
                **_performance_row_from_result(adaptive_results),
            }
        )
    return pd.DataFrame(rows)


def build_dashboard_recommendation(
    portfolio_payload: dict[str, object] | None,
    robustness_payload: dict[str, object] | None,
    *,
    investor_profile: str,
    base_bps: float,
    slippage_bps: float,
) -> StrategyRecommendation:
    """Build one recommendation from current results plus persisted validation evidence."""

    artifacts = load_selection_artifacts(project_root)
    if robustness_payload is not None:
        current_cpcv = robustness_payload.get("robustness_ranking")
        if not isinstance(current_cpcv, pd.DataFrame) or current_cpcv.empty:
            current_cpcv = robustness_payload.get("summary_table")
        if isinstance(current_cpcv, pd.DataFrame) and not current_cpcv.empty:
            artifacts["cpcv_summary"] = current_cpcv

    candidate_metrics = None
    current_regime = None
    n_observations = None
    if portfolio_payload is not None:
        candidate_metrics = build_selection_candidate_metrics(
            portfolio_payload,
            base_bps=base_bps,
            slippage_bps=slippage_bps,
        )
        n_observations = len(
            portfolio_payload.get("portfolio_returns", pd.Series(dtype=float))
        )
        regime_payload = portfolio_payload.get("market_regime_results") or {}
        current_regime = regime_payload.get(
            "current_decision_regime",
            regime_payload.get("current_observed_regime"),
        )

    return select_strategy_for_profile(
        investor_profile,
        candidate_metrics=candidate_metrics,
        current_regime=current_regime,
        base_bps=base_bps,
        slippage_bps=slippage_bps,
        hmm_walk_forward_valid=bool(HMM_AVAILABLE),
        n_observations=n_observations,
        artifacts=artifacts,
    )


def selection_gate_table(
    recommendation: StrategyRecommendation,
) -> pd.DataFrame:
    """Flatten gate dataclasses for research and developer diagnostics."""

    rows = []
    for strategy, gates in recommendation.gate_results.items():
        for gate in gates:
            rows.append({"strategy": strategy, **gate.to_dict()})
    return pd.DataFrame(rows)


def _format_selection_tradeoff_table(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    display = frame.copy()
    for column in ["cagr", "max_drawdown", "stress_period_return"]:
        if column in display:
            display[column] = display[column].map(format_percent)
    for column in ["calmar", "total_turnover"]:
        if column in display:
            display[column] = display[column].map(format_decimal)
    for column in ["final_value", "total_transaction_cost"]:
        if column in display:
            display[column] = display[column].map(format_currency)
    return display.rename(
        columns={column: net_metric_label(column) for column in display.columns}
    )


def render_selection_diagnostics(
    recommendation: StrategyRecommendation,
) -> None:
    """Render profile mapping, gates, scores, ranking, and playbook."""

    st.header("Strategy Selection Diagnostics")
    st.caption(
        "Selection is based on net metrics, walk-forward-safe evidence, CPCV coverage, "
        "replication, stress behavior, turnover, and defensive-source metadata."
    )
    st.write(
        f"Profile: **{recommendation.investor_profile}** · Scenarios: "
        f"**{', '.join(recommendation.scenario_categories)}**"
    )
    with st.expander("Selection Gates", expanded=True):
        st.dataframe(selection_gate_table(recommendation), use_container_width=True)
    with st.expander("Candidate Scores and Ranking", expanded=True):
        scores = recommendation.candidate_scores.reset_index().rename(
            columns={"index": "strategy"}
        )
        st.dataframe(scores, use_container_width=True)
    with st.expander("Profile and Role Mapping", expanded=False):
        role_frame = pd.DataFrame(
            [
                {"strategy": strategy, "assigned_role": role}
                for strategy, role in recommendation.role_assignments.items()
            ]
        )
        st.dataframe(role_frame, use_container_width=True)
    with st.expander("Scenario Playbook", expanded=False):
        st.dataframe(build_strategy_playbook(), use_container_width=True)


def render_manager_view(
    portfolio_payload: dict[str, object] | None,
    robustness_payload: dict[str, object] | None,
    *,
    selected_objective_label: str,
    recommendation: StrategyRecommendation | None = None,
) -> None:
    """Render only the decision inputs and outputs needed by a portfolio manager."""

    if recommendation is None:
        st.info(
            "Choose the portfolio universe, investment amount, date range, investor "
            "objective, and cost assumption, then select Run Recommendation."
        )
        st.caption(
            "The default Balanced workflow evaluates HERC with "
            "Regime-Adaptive HMM Walk-Forward — Conservative as the risk-control overlay."
        )
        return

    st.header("Strategy Recommendation")
    core_col, overlay_col, confidence_col = st.columns(3)
    core_col.metric("Core Portfolio", recommendation.main_strategy)
    overlay_col.metric(
        "Overlay / Reference",
        recommendation.overlay_strategy or "None",
    )
    confidence_col.metric(
        "Recommendation Confidence",
        recommendation.confidence,
        delta=f"{recommendation.confidence_score:.0%}",
    )
    st.success(recommendation.explanation)
    st.caption(
        f"Current scenario assessment: {', '.join(recommendation.scenario_categories)}"
    )

    st.header("Tradeoff Table")
    st.dataframe(
        _format_selection_tradeoff_table(
            recommendation.evidence.get("comparison_table", pd.DataFrame())
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.header("Why This Recommendation")
    st.write(
        f"**{recommendation.main_strategy}** is the selected fixed core. "
        f"**{recommendation.overlay_strategy or 'No adaptive strategy'}** is assigned "
        f"the role **{recommendation.overlay_role or 'Not selected'}**."
    )
    st.write(
        "The engine separates growth leadership from risk-control and robustness roles. "
        "It does not promote an adaptive strategy on a single backtest or gross result."
    )
    st.caption(
        f"{RULE_BASED_ROBUSTNESS_REFERENCE} remains the CPCV-favored robustness reference; "
        "limited successful-fold coverage reduces confidence."
    )

    st.header("Warnings and Assumptions")
    for warning in recommendation.warnings:
        st.warning(warning)
    for assumption in recommendation.assumptions:
        st.caption(f"• {assumption}")


def render_developer_view(
    portfolio_payload: dict[str, object] | None,
    sensitivity_payload: dict[str, object] | None,
    robustness_payload: dict[str, object] | None,
    *,
    selected_objective_label: str,
    recommendation: StrategyRecommendation | None = None,
) -> None:
    """Render raw audit and implementation diagnostics behind collapsed sections."""
    st.info(MODE_NOTES[DEVELOPER_VIEW])
    st.caption(f"Current research objective: {selected_objective_label}")

    if portfolio_payload is None and sensitivity_payload is None and robustness_payload is None:
        st.info("Run an analysis before opening raw developer diagnostics.")
        return

    regime_payload = (
        portfolio_payload.get("market_regime_results")
        if portfolio_payload is not None
        else None
    )
    adaptive_results = (
        portfolio_payload.get("adaptive_results")
        if portfolio_payload is not None
        else None
    )

    with st.expander("1. Raw HMM Diagnostics", expanded=False):
        if regime_payload is None:
            st.caption("No regime payload is available.")
        else:
            probabilities = regime_payload.get(
                "hmm_state_probabilities",
                pd.DataFrame(),
            )
            diagnostics = regime_payload.get("hmm_diagnostics", pd.DataFrame())
            comparison = regime_payload.get("method_comparison") or {}
            disagreements = comparison.get("dates_of_disagreement", pd.DataFrame())
            if not probabilities.empty:
                st.dataframe(probabilities.tail(100), use_container_width=True)
            render_dataframe_download(
                "Download Full HMM Probability Table",
                probabilities,
                "hmm_state_probabilities.csv",
                key="download_hmm_probabilities",
            )
            st.write("HMM fit diagnostics")
            st.dataframe(diagnostics, use_container_width=True)
            st.write("Rule-based versus HMM disagreement dates")
            st.dataframe(disagreements, use_container_width=True)

    with st.expander("2. Raw CPCV Diagnostics", expanded=False):
        if robustness_payload is None:
            st.caption("No CPCV payload is available.")
        else:
            split_diagnostics = robustness_payload.get(
                "split_diagnostics",
                pd.DataFrame(),
            )
            fold_results = robustness_payload.get("fold_results", pd.DataFrame())
            st.write("Split diagnostics")
            st.dataframe(split_diagnostics, use_container_width=True)
            st.write("Fold-level metrics and failed-fold traces")
            st.dataframe(fold_results, use_container_width=True)
            render_dataframe_download(
                "Download Full CPCV Fold Table",
                fold_results,
                "cpcv_fold_results.csv",
                key="developer_download_cpcv_folds",
            )

    with st.expander("3. Full Adaptive Decision Log", expanded=False):
        diagnostics = (
            adaptive_results.get("diagnostics", pd.DataFrame())
            if adaptive_results is not None
            else pd.DataFrame()
        )
        st.dataframe(diagnostics, use_container_width=True)
        render_dataframe_download(
            "Download Full Adaptive Diagnostics",
            diagnostics,
            "adaptive_daily_diagnostics.csv",
            key="download_adaptive_diagnostics",
        )

    with st.expander("4. Full Weight History", expanded=False):
        weights = (
            adaptive_results.get("weights", pd.DataFrame())
            if adaptive_results is not None
            else (
                portfolio_payload.get("backtest_results", {}).get(
                    "weights_history",
                    pd.DataFrame(),
                )
                if portfolio_payload is not None
                else pd.DataFrame()
            )
        )
        st.dataframe(weights, use_container_width=True)
        render_dataframe_download(
            "Download Full Weight History",
            weights,
            "full_weight_history.csv",
            key="download_full_weight_history",
        )

    with st.expander("5. Net/Gross Reconciliation", expanded=False):
        if portfolio_payload is None:
            st.caption("No portfolio payload is available.")
        else:
            cost_drag = portfolio_payload.get("cost_drag_summary", {})
            reconciliation = pd.DataFrame(
                [
                    {
                        "Gross Final Value": cost_drag.get("gross_final_value"),
                        "Net Final Value": cost_drag.get("net_final_value"),
                        "Cost Drag": cost_drag.get("cost_drag"),
                        "Cost Drag %": cost_drag.get("cost_drag_pct"),
                    }
                ]
            )
            st.dataframe(reconciliation, use_container_width=True)
            st.plotly_chart(
                plot_cost_adjusted_comparison(
                    portfolio_payload["gross_portfolio_value"],
                    portfolio_payload["portfolio_value"],
                ),
                use_container_width=True,
            )

    with st.expander("6. Defensive Return Reconciliation", expanded=False):
        if adaptive_results is None:
            st.caption("No adaptive defensive-return result is available.")
        else:
            metadata = dict(adaptive_results.get("defensive_metadata", {}))
            defensive_series = pd.to_numeric(
                adaptive_results.get("defensive_returns", pd.Series(dtype=float)),
                errors="coerce",
            ).dropna()
            reconciliation = pd.DataFrame(
                [
                    {
                        **metadata,
                        "observations": int(len(defensive_series)),
                        "mean_daily_return": (
                            float(defensive_series.mean())
                            if not defensive_series.empty
                            else np.nan
                        ),
                        "compounded_annualized_return": (
                            float(
                                (1.0 + defensive_series).prod()
                                ** (252.0 / len(defensive_series))
                                - 1.0
                            )
                            if not defensive_series.empty
                            else np.nan
                        ),
                    }
                ]
            )
            st.dataframe(reconciliation, use_container_width=True)
            render_dataframe_download(
                "Download Defensive Return Reconciliation",
                reconciliation,
                "defensive_return_reconciliation.csv",
                key="download_defensive_reconciliation",
            )

    with st.expander("7. Internal Config Dump", expanded=False):
        config_dump = {
            "selected_tickers": (
                portfolio_payload.get("selected_tickers")
                if portfolio_payload is not None
                else None
            ),
            "benchmark_strategy": (
                portfolio_payload.get("benchmark_strategy")
                if portfolio_payload is not None
                else None
            ),
            "covariance_method": (
                portfolio_payload.get("covariance_method")
                if portfolio_payload is not None
                else None
            ),
            "adaptive_regime_source": (
                portfolio_payload.get("adaptive_regime_source")
                if portfolio_payload is not None
                else None
            ),
            "adaptive_policy_preset": (
                portfolio_payload.get("adaptive_policy_preset")
                if portfolio_payload is not None
                else None
            ),
            "defensive_metadata": (
                portfolio_payload.get("defensive_metadata")
                if portfolio_payload is not None
                else None
            ),
            "research_objective": selected_objective_label,
        }
        st.json(config_dump)
        if sensitivity_payload is not None:
            render_dataframe_download(
                "Download Full Sensitivity Table",
                sensitivity_payload.get("experiment_results_df", pd.DataFrame()),
                "full_sensitivity_results.csv",
                key="developer_download_sensitivity",
            )

    with st.expander("8. Raw Strategy Recommendation", expanded=False):
        if recommendation is None:
            st.caption("No strategy recommendation is available.")
        else:
            raw = recommendation.to_dict()
            raw.pop("gate_results", None)
            raw.pop("candidate_scores", None)
            st.json(raw)

    with st.expander("9. Selection Gate Results", expanded=False):
        if recommendation is None:
            st.caption("No selection gates are available.")
        else:
            st.dataframe(
                selection_gate_table(recommendation),
                use_container_width=True,
            )

    with st.expander("10. Selection Artifact Diagnostics and Scoring", expanded=False):
        if recommendation is None:
            st.caption("No selection artifact diagnostics are available.")
        else:
            st.json(recommendation.artifact_diagnostics)
            st.write("Candidate scoring trace")
            st.dataframe(
                recommendation.candidate_scores.reset_index(),
                use_container_width=True,
            )


initialize_session_state()

st.set_page_config(page_title="Adaptive Portfolio Risk Analytics", layout="wide")
st.title("Adaptive Portfolio Risk Analytics")

all_asset_labels = [ticker_label(ticker) for ticker in INDIAN_ASSET_UNIVERSE]

st.sidebar.header("Dashboard Mode")
dashboard_mode = st.sidebar.radio(
    "View",
    DASHBOARD_MODES,
    key="ui_dashboard_mode",
    help=(
        "Manager View is decision-ready. Research View exposes methodology and "
        "validation controls. Developer / Debug View contains raw audit outputs."
    ),
)
if dashboard_mode in MODE_NOTES:
    st.sidebar.caption(MODE_NOTES[dashboard_mode])

st.sidebar.header("Portfolio Inputs")

manager_investor_profile = str(
    st.session_state.get("ui_manager_investor_profile", "Balanced")
)
manager_cost_assumption = str(
    st.session_state.get("ui_manager_cost_assumption", "Moderate")
)

if dashboard_mode == MANAGER_VIEW:
    with st.sidebar.expander("Portfolio Universe", expanded=True):
        preset = st.selectbox(
            "Portfolio Universe",
            ["Core Diversified", "Banks + IT + Gold", "Full Research Universe"],
            key="ui_manager_universe_preset",
        )
        start_date = st.date_input("Start Date", key="ui_manager_start_date")
        end_date = st.date_input("End Date", key="ui_manager_end_date")
        initial_capital = st.number_input(
            "Investment Amount",
            min_value=100_000.0,
            max_value=100_000_000.0,
            step=100_000.0,
            key="ui_manager_initial_capital",
        )

    with st.sidebar.expander("Recommendation Inputs", expanded=True):
        manager_investor_profile = st.selectbox(
            "Investor Objective",
            PROFILE_NAMES,
            key="ui_manager_investor_profile",
        )
        manager_cost_assumption = st.selectbox(
            "Cost Assumption",
            COST_ASSUMPTION_NAMES,
            key="ui_manager_cost_assumption",
        )
        if manager_cost_assumption == "Custom":
            base_bps = st.number_input(
                "Custom Base Cost (bps)",
                min_value=0.0,
                max_value=100.0,
                key="ui_manager_custom_base_bps",
            )
            slippage_bps = st.number_input(
                "Custom Slippage (bps)",
                min_value=0.0,
                max_value=100.0,
                key="ui_manager_custom_slippage_bps",
            )
        else:
            base_bps, slippage_bps = COST_ASSUMPTIONS[manager_cost_assumption]
            st.caption(
                f"Applied cost: {base_bps:.0f} bps base + "
                f"{slippage_bps:.0f} bps slippage."
            )
        run_portfolio_button = st.button(
            "Run Recommendation",
            key="ui_run_portfolio",
            type="primary",
        )

    strategy = "HERC"
    comparison_strategies = ["Equal Weight", "Inverse Volatility", "HRP", "HERC"]
    benchmark_strategy = "Equal Weight"
    global_research_objective_label = MANAGER_PROFILE_OBJECTIVES[
        manager_investor_profile
    ]
else:
    with st.sidebar.expander("Portfolio Scope", expanded=dashboard_mode != DEVELOPER_VIEW):
        preset = st.selectbox(
            "Universe Preset",
            ["Core Diversified", "Banks + IT + Gold", "Full Research Universe", "Custom"],
            key="ui_universe_preset",
        )
        select_all = st.checkbox(
            "Select All Assets In Preset",
            key="ui_select_all_assets",
        )

        if preset != st.session_state.get("_last_preset") or select_all != st.session_state.get(
            "_last_select_all"
        ):
            st.session_state["ui_selected_assets"] = update_selected_assets_for_preset(
                preset, select_all
            )
            st.session_state["_last_preset"] = preset
            st.session_state["_last_select_all"] = select_all

        asset_options = all_asset_labels if preset == "Custom" else default_labels_for_preset(preset)
        st.multiselect(
            "Selected Assets",
            options=asset_options,
            key="ui_selected_assets",
            help="Search by ticker or company name. Use Add Ticker below for symbols outside this list.",
        )

        with st.expander("Add Ticker", expanded=False):
            st.text_input(
                "Ticker Symbol",
                key="ui_ticker_to_add",
                help="Enter a Yahoo Finance symbol such as AAPL, RELIANCE.NS, or BTC-USD.",
            )
            add_ticker_col, clear_ticker_col = st.columns(2)
            add_ticker_col.button(
                "Add Ticker",
                key="ui_add_ticker",
                on_click=add_tickers_to_portfolio,
            )
            clear_ticker_col.button(
                "Clear Added",
                key="ui_clear_added_tickers",
                on_click=clear_added_tickers,
                disabled=not st.session_state.get("ui_added_tickers"),
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

    with st.sidebar.expander("Core Strategy Setup", expanded=False):
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
            key="ui_comparison_strategies",
        )
        benchmark_strategy = st.selectbox(
            "Benchmark Strategy",
            ["Equal Weight", "Inverse Volatility", "HRP", "HERC"],
            index=0,
            key="ui_benchmark_strategy",
        )
        base_bps = st.number_input(
            "Base Cost (bps)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
            key="ui_base_bps",
        )
        slippage_bps = st.number_input(
            "Slippage (bps)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=1.0,
            key="ui_slippage_bps",
        )
        global_research_objective_label = st.selectbox(
            "Research Objective",
            list(RESEARCH_OBJECTIVES),
            key="ui_research_objective",
        )
        run_portfolio_button = (
            st.button("Run Portfolio Analysis", key="ui_run_portfolio")
            if dashboard_mode != DEVELOPER_VIEW
            else False
        )

if dashboard_mode == RESEARCH_VIEW:
    with st.sidebar.expander("Advanced Strategy Controls", expanded=False):
        covariance_method = st.selectbox(
            "Covariance Method",
            COVARIANCE_METHOD_OPTIONS,
            help="Covariance estimator used by HRP, HERC, and research comparisons.",
            key="ui_covariance_method",
        )
        rebalance_mode = st.selectbox(
            "Rebalance Mode",
            ["calendar", "threshold", "calendar_or_threshold"],
            help="Calendar rebalances on schedule. Threshold modes use weight drift.",
            key="ui_rebalance_mode",
        )
        if rebalance_mode in {"threshold", "calendar_or_threshold"}:
            threshold = st.slider(
                "Threshold",
                min_value=0.01,
                max_value=0.20,
                step=0.01,
                help="Maximum absolute weight drift required before rebalancing.",
                key="ui_threshold",
            )
        else:
            threshold = 0.05
else:
    covariance_method = str(st.session_state.get("ui_covariance_method", "sample"))
    rebalance_mode = str(st.session_state.get("ui_rebalance_mode", "calendar"))
    threshold = float(st.session_state.get("ui_threshold", 0.05))

enable_vol_targeting = False
defensive_sleeve = "Synthetic 4% annualized"
synthetic_annual_rate = 0.04
vol_target_mode = "Adaptive"
base_target_vol = 0.10
realized_vol_window = 63
regime_lookback_window = 252
exposure_floor = 0.25
exposure_cap = 1.0
no_trade_band = 0.05

enable_regime_analytics = dashboard_mode == MANAGER_VIEW
regime_method = (
    DEFAULT_MANAGER_ADAPTIVE_OVERLAY["regime_method"]
    if HMM_AVAILABLE
    else "Rule-based"
)
regime_vol_lookback = 63
regime_trend_lookback = 126
regime_corr_lookback = 63
regime_crisis_drawdown = -0.15
regime_stress_drawdown = -0.08
use_lagged_decision_regime = True
hmm_n_states = 4
hmm_min_train_size = 504
hmm_refit_frequency = 21
hmm_covariance_type = "diag"
hmm_decision_lag = 1
hmm_feature_columns = list(DEFAULT_HMM_FEATURE_COLUMNS)

enable_adaptive_strategy = dashboard_mode == MANAGER_VIEW
adaptive_regime_source = (
    DEFAULT_MANAGER_ADAPTIVE_OVERLAY["regime_source"]
    if HMM_AVAILABLE
    else "Rule-based lagged decision regime"
)
adaptive_policy_preset = DEFAULT_MANAGER_ADAPTIVE_OVERLAY["policy_preset"]
adaptive_training_window = 252
adaptive_rebalance_frequency = "M"
adaptive_show_policy_table = False

sensitivity_strategies = ["HRP", "HERC"]
sensitivity_covariance_methods = ["sample", "ledoit_wolf", "ewma_ledoit_wolf"]
sensitivity_rebalance_modes = ["calendar", "threshold"]
sensitivity_thresholds = "0.03,0.05,0.10"
sensitivity_max_runs = 24
include_adaptive_sensitivity = False
sensitivity_adaptive_sources = ["rule_based_lagged"]
sensitivity_adaptive_presets = ["conservative", "balanced", "aggressive"]
sensitivity_adaptive_training_window = 252
sensitivity_adaptive_rebalance_frequency = "M"
sensitivity_max_adaptive_configs = 6
run_sensitivity_button = False

enable_robustness_validation = False
robustness_n_blocks = 6
robustness_n_test_blocks = 2
robustness_embargo_pct = 1.0
robustness_purge_window = 0
robustness_max_configs = 5
include_adaptive_in_cpcv = False
robustness_max_adaptive_configs = 2
run_robustness_button = False

if dashboard_mode == RESEARCH_VIEW:
    with st.sidebar.expander("Defensive Sleeve", expanded=False):
        defensive_options = [
            "Synthetic 4% annualized",
            "Cash / zero return",
            "LIQUIDBEES.NS",
            "LIQUIDETF.NS",
            "Provided series if available",
        ]
        if st.session_state.get("ui_defensive_sleeve") not in defensive_options:
            st.session_state["ui_defensive_sleeve"] = defensive_options[0]
        defensive_sleeve = st.selectbox(
            "Defensive sleeve source",
            defensive_options,
            key="ui_defensive_sleeve",
        )
        synthetic_annual_rate = st.number_input(
            "Synthetic annual rate",
            min_value=0.0,
            max_value=0.20,
            value=0.04,
            step=0.01,
            format="%.2f",
            key="ui_synthetic_annual_rate",
        )
        if defensive_sleeve == "Provided series if available":
            st.caption(
                "No uploaded series is available in the current dashboard flow; "
                "the run will record and use the synthetic fallback."
            )

    with st.sidebar.expander("Volatility Targeting", expanded=False):
        enable_vol_targeting = st.checkbox(
            "Enable Volatility Targeting",
            value=False,
            key="ui_enable_vol_targeting",
        )
        if enable_vol_targeting:
            vol_target_mode = st.selectbox(
                "Vol Target Mode",
                ["Adaptive", "Fixed"],
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
                min_value=0.0,
                max_value=0.20,
                value=0.05,
                step=0.01,
                key="ui_no_trade_band",
            )

    with st.sidebar.expander("Phase 3B — Regime Detection", expanded=False):
        enable_regime_analytics = st.checkbox(
            "Enable Regime Analytics",
            key="ui_enable_regime_analytics",
        )
        regime_method_options = ["Rule-based"]
        if HMM_AVAILABLE:
            regime_method_options.extend(
                ["HMM full-sample historical", "HMM walk-forward experimental"]
            )
        elif st.session_state.get("ui_regime_method") not in regime_method_options:
            st.session_state["ui_regime_method"] = "Rule-based"
        if enable_regime_analytics:
            if st.session_state.get("ui_regime_method") not in regime_method_options:
                st.session_state["ui_regime_method"] = regime_method_options[0]
            regime_method = st.selectbox(
                "Regime Method",
                regime_method_options,
                key="ui_regime_method",
                help="Full-sample HMM is historical-only and cannot drive adaptive trading.",
            )
            regime_vol_lookback = st.slider(
                "Volatility Lookback",
                min_value=21,
                max_value=126,
                value=63,
                key="ui_market_regime_vol_lookback",
            )
            regime_trend_lookback = st.slider(
                "Trend Lookback",
                min_value=63,
                max_value=252,
                value=126,
                key="ui_market_regime_trend_lookback",
            )
            regime_corr_lookback = st.slider(
                "Correlation Lookback",
                min_value=21,
                max_value=126,
                value=63,
                key="ui_market_regime_corr_lookback",
            )
            regime_crisis_drawdown = st.slider(
                "Crisis Drawdown Threshold",
                min_value=-0.40,
                max_value=-0.05,
                value=-0.15,
                step=0.01,
                format="%.2f",
                key="ui_market_regime_crisis_drawdown",
            )
            regime_stress_drawdown = st.slider(
                "Stress Drawdown Threshold",
                min_value=-0.25,
                max_value=-0.02,
                value=-0.08,
                step=0.01,
                format="%.2f",
                key="ui_market_regime_stress_drawdown",
            )
            use_lagged_decision_regime = st.checkbox(
                "Use Lagged Decision Regime",
                value=True,
                key="ui_use_lagged_decision_regime",
            )
            if regime_method != "Rule-based":
                hmm_n_states = st.selectbox(
                    "Number of Hidden States",
                    [2, 3, 4],
                    index=2,
                    key="ui_hmm_n_states",
                )
                hmm_covariance_type = st.selectbox(
                    "HMM Covariance Type",
                    ["diag", "full"],
                    key="ui_hmm_covariance_type",
                )
                hmm_decision_lag = st.number_input(
                    "HMM Decision Lag",
                    min_value=0,
                    max_value=21,
                    value=1,
                    key="ui_hmm_decision_lag",
                )
                hmm_feature_columns = st.multiselect(
                    "HMM Feature Columns",
                    DEFAULT_HMM_FEATURE_COLUMNS,
                    default=DEFAULT_HMM_FEATURE_COLUMNS,
                    key="ui_hmm_feature_columns",
                )
                if regime_method == "HMM walk-forward experimental":
                    hmm_min_train_size = st.number_input(
                        "HMM Minimum Training Size",
                        min_value=63,
                        max_value=2520,
                        value=504,
                        step=21,
                        key="ui_hmm_min_train_size",
                    )
                    hmm_refit_frequency = st.number_input(
                        "HMM Refit Frequency",
                        min_value=1,
                        max_value=252,
                        value=21,
                        key="ui_hmm_refit_frequency",
                    )

    with st.sidebar.expander("Phase 3C — Adaptive Allocation Policy", expanded=False):
        enable_adaptive_strategy = st.checkbox(
            "Enable Adaptive Regime Strategy",
            key="ui_enable_adaptive_strategy",
        )
        if enable_adaptive_strategy:
            adaptive_regime_sources = ["Rule-based lagged decision regime"]
            if HMM_AVAILABLE:
                adaptive_regime_sources.append("HMM walk-forward decision regime")
            if st.session_state.get("ui_adaptive_regime_source") not in adaptive_regime_sources:
                st.session_state["ui_adaptive_regime_source"] = adaptive_regime_sources[0]
            adaptive_regime_source = st.selectbox(
                "Adaptive Regime Source",
                adaptive_regime_sources,
                key="ui_adaptive_regime_source",
            )
            adaptive_policy_preset = st.selectbox(
                "Policy Preset",
                ["Conservative", "Balanced default", "Aggressive"],
                key="ui_adaptive_policy_preset",
            )
            adaptive_training_window = st.number_input(
                "Adaptive Training Window",
                min_value=60,
                max_value=756,
                value=252,
                step=21,
                key="ui_adaptive_training_window",
            )
            adaptive_rebalance_frequency = st.selectbox(
                "Adaptive Rebalance Frequency",
                ["M", "W", "Q"],
                format_func=lambda value: {
                    "M": "Monthly",
                    "W": "Weekly",
                    "Q": "Quarterly",
                }[value],
                key="ui_adaptive_rebalance_frequency",
            )
            adaptive_show_policy_table = st.checkbox(
                "Show Policy Table",
                value=True,
                key="ui_adaptive_show_policy_table",
            )
        st.caption(
            "Rule-based Conservative is the CPCV-favored robustness reference. "
            "HMM Conservative is the post-P0 low-drawdown overlay."
        )

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
            key="ui_sensitivity_thresholds",
        )
        sensitivity_max_runs = st.number_input(
            "Sensitivity Max Runs",
            min_value=1,
            max_value=500,
            value=24,
            key="ui_sensitivity_max_runs",
        )
        include_adaptive_sensitivity = st.checkbox(
            "Include Adaptive Regime Strategies",
            value=False,
            key="ui_include_adaptive_sensitivity",
        )
        if include_adaptive_sensitivity:
            adaptive_source_options = ["rule_based_lagged"]
            if HMM_AVAILABLE:
                adaptive_source_options.append("hmm_walk_forward")
            sensitivity_adaptive_sources = st.multiselect(
                "Adaptive Regime Sources",
                adaptive_source_options,
                default=["rule_based_lagged"],
                format_func=lambda value: {
                    "rule_based_lagged": "Rule-based lagged",
                    "hmm_walk_forward": "HMM walk-forward",
                }[value],
                key="ui_sensitivity_adaptive_sources",
            )
            sensitivity_adaptive_presets = st.multiselect(
                "Adaptive Policy Presets",
                ["conservative", "balanced", "aggressive"],
                default=["conservative", "balanced", "aggressive"],
                format_func=str.title,
                key="ui_sensitivity_adaptive_presets",
            )
            sensitivity_adaptive_training_window = st.number_input(
                "Adaptive Experiment Training Window",
                min_value=60,
                max_value=756,
                value=252,
                step=21,
                key="ui_sensitivity_adaptive_training_window",
            )
            sensitivity_adaptive_rebalance_frequency = st.selectbox(
                "Adaptive Experiment Rebalance Frequency",
                ["M", "W", "Q"],
                key="ui_sensitivity_adaptive_rebalance_frequency",
            )
            sensitivity_max_adaptive_configs = st.number_input(
                "Maximum Adaptive Configurations",
                min_value=1,
                max_value=12,
                value=6,
                key="ui_sensitivity_max_adaptive_configs",
            )
        run_sensitivity_button = st.button(
            "Run Sensitivity Study",
            key="ui_run_sensitivity",
        )

    with st.sidebar.expander(
        "Phase 3A — CPCV Robustness Validation",
        expanded=False,
    ):
        enable_robustness_validation = st.checkbox(
            "Enable Robustness Validation",
            value=False,
            key="ui_enable_robustness_validation",
        )
        if enable_robustness_validation:
            robustness_n_blocks = st.number_input(
                "Number of Time Blocks",
                min_value=2,
                max_value=12,
                value=6,
                key="ui_robustness_n_blocks",
            )
            robustness_n_test_blocks = st.number_input(
                "Test Blocks per Split",
                min_value=1,
                max_value=6,
                value=2,
                key="ui_robustness_n_test_blocks",
            )
            robustness_embargo_pct = st.number_input(
                "Embargo (%)",
                min_value=0.0,
                max_value=20.0,
                value=1.0,
                step=0.5,
                key="ui_robustness_embargo_pct",
            )
            robustness_purge_window = st.number_input(
                "Purge Window (observations)",
                min_value=0,
                max_value=63,
                value=0,
                key="ui_robustness_purge_window",
            )
            robustness_max_configs = st.number_input(
                "Maximum Configurations",
                min_value=1,
                max_value=10,
                value=5,
                key="ui_robustness_max_configs",
            )
            include_adaptive_in_cpcv = st.checkbox(
                "Include Adaptive Strategies in CPCV",
                value=False,
                key="ui_include_adaptive_in_cpcv",
            )
            if include_adaptive_in_cpcv:
                robustness_max_adaptive_configs = st.number_input(
                    "Maximum Adaptive Configurations in CPCV",
                    min_value=1,
                    max_value=5,
                    value=2,
                    key="ui_robustness_max_adaptive_configs",
                )
            st.caption(
                f"Current research objective: {global_research_objective_label}. "
                "Calmar is the fallback only when no objective is supplied."
            )
            run_robustness_button = st.button(
                "Run Robustness Validation",
                key="ui_run_robustness",
            )


if dashboard_mode == MANAGER_VIEW:
    selected_tickers = list(DEFAULT_UNIVERSES[preset])
else:
    selected_tickers = merge_portfolio_tickers(
        st.session_state["ui_selected_assets"],
        st.session_state.get("ui_added_tickers", []),
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
    if (
        validation_error is None
        and enable_regime_analytics
        and regime_crisis_drawdown >= regime_stress_drawdown
    ):
        validation_error = (
            "The crisis drawdown threshold must be more negative than the stress threshold."
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
                enable_regime_analytics=enable_regime_analytics,
                regime_vol_lookback=int(regime_vol_lookback),
                regime_trend_lookback=int(regime_trend_lookback),
                regime_corr_lookback=int(regime_corr_lookback),
                regime_crisis_drawdown=float(regime_crisis_drawdown),
                regime_stress_drawdown=float(regime_stress_drawdown),
                use_lagged_decision_regime=use_lagged_decision_regime,
                regime_objective_metric=objective_metric(
                    global_research_objective_label
                ),
                regime_method=regime_method,
                hmm_n_states=int(hmm_n_states),
                hmm_min_train_size=int(hmm_min_train_size),
                hmm_refit_frequency=int(hmm_refit_frequency),
                hmm_covariance_type=hmm_covariance_type,
                hmm_decision_lag=int(hmm_decision_lag),
                hmm_feature_columns=hmm_feature_columns,
                enable_adaptive_strategy=enable_adaptive_strategy,
                adaptive_regime_source=adaptive_regime_source,
                adaptive_policy_preset=adaptive_policy_preset,
                adaptive_training_window=int(adaptive_training_window),
                adaptive_rebalance_frequency=adaptive_rebalance_frequency,
                adaptive_show_policy_table=adaptive_show_policy_table,
            )
            portfolio_payload["strategy_recommendation"] = (
                build_dashboard_recommendation(
                    portfolio_payload,
                    st.session_state.get(ROBUSTNESS_RESULT_KEY),
                    investor_profile=manager_investor_profile,
                    base_bps=float(base_bps),
                    slippage_bps=float(slippage_bps),
                )
            )
            portfolio_payload["selection_investor_profile"] = manager_investor_profile
            portfolio_payload["selection_cost_assumption"] = manager_cost_assumption
            portfolio_payload["selection_base_bps"] = float(base_bps)
            portfolio_payload["selection_slippage_bps"] = float(slippage_bps)
            st.session_state[PORTFOLIO_RESULT_KEY] = portfolio_payload
            st.session_state[UI_MESSAGE_KEY] = (
                "info",
                "Recommendation updated."
                if dashboard_mode == MANAGER_VIEW
                else "Portfolio analysis updated.",
            )
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
                objective_label=global_research_objective_label,
                max_runs=int(sensitivity_max_runs),
                initial_capital=initial_capital,
                include_adaptive_strategies=include_adaptive_sensitivity,
                adaptive_regime_sources=sensitivity_adaptive_sources,
                adaptive_policy_presets=sensitivity_adaptive_presets,
                max_adaptive_configs=int(sensitivity_max_adaptive_configs),
                adaptive_training_window=int(sensitivity_adaptive_training_window),
                adaptive_rebalance_frequency=(sensitivity_adaptive_rebalance_frequency),
                hmm_n_states=int(hmm_n_states),
                hmm_min_train_size=int(hmm_min_train_size),
                hmm_refit_frequency=int(hmm_refit_frequency),
                hmm_covariance_type=hmm_covariance_type,
                hmm_decision_lag=max(1, int(hmm_decision_lag)),
            )
            st.session_state[SENSITIVITY_RESULT_KEY] = sensitivity_payload
            st.session_state[ROBUSTNESS_RESULT_KEY] = None
            st.session_state[UI_MESSAGE_KEY] = ("info", "Sensitivity study completed.")
        except ValueError as exc:
            st.session_state[UI_MESSAGE_KEY] = ("warning", str(exc))
        except Exception as exc:
            st.session_state[UI_MESSAGE_KEY] = ("error", f"Sensitivity study failed: {exc}")

if run_robustness_button:
    sensitivity_payload = st.session_state.get(SENSITIVITY_RESULT_KEY)
    if sensitivity_payload is None:
        st.session_state[UI_MESSAGE_KEY] = (
            "warning",
            "Run the sensitivity study before robustness validation.",
        )
    elif int(robustness_n_test_blocks) > int(robustness_n_blocks):
        st.session_state[UI_MESSAGE_KEY] = (
            "warning",
            "Test blocks per split cannot exceed the number of time blocks.",
        )
    else:
        try:
            robustness_payload = build_robustness_results(
                sensitivity_payload,
                n_blocks=int(robustness_n_blocks),
                n_test_blocks=int(robustness_n_test_blocks),
                embargo_pct=float(robustness_embargo_pct) / 100.0,
                purge_window=int(robustness_purge_window),
                max_configs=int(robustness_max_configs),
                objective_metric=objective_metric(global_research_objective_label),
                include_adaptive_in_cpcv=include_adaptive_in_cpcv,
                max_adaptive_configs=int(robustness_max_adaptive_configs),
            )
            st.session_state[ROBUSTNESS_RESULT_KEY] = robustness_payload
            st.session_state[UI_MESSAGE_KEY] = (
                "info",
                "Robustness validation completed.",
            )
        except ValueError as exc:
            st.session_state[UI_MESSAGE_KEY] = ("warning", str(exc))
        except Exception as exc:
            st.session_state[UI_MESSAGE_KEY] = (
                "error",
                f"Robustness validation failed: {exc}",
            )


if st.session_state.get(UI_MESSAGE_KEY) is not None:
    message_level, message_text = st.session_state[UI_MESSAGE_KEY]
    show_message(message_level, message_text)

portfolio_payload = st.session_state.get(PORTFOLIO_RESULT_KEY)
sensitivity_payload = st.session_state.get(SENSITIVITY_RESULT_KEY)
robustness_payload = st.session_state.get(ROBUSTNESS_RESULT_KEY)
selection_recommendation = (
    portfolio_payload.get("strategy_recommendation")
    if portfolio_payload is not None
    else None
)
if selection_recommendation is None and dashboard_mode != MANAGER_VIEW:
    try:
        selection_recommendation = build_dashboard_recommendation(
            portfolio_payload,
            robustness_payload,
            investor_profile=manager_investor_profile,
            base_bps=float(base_bps),
            slippage_bps=float(slippage_bps),
        )
    except ValueError:
        selection_recommendation = None

if dashboard_mode == MANAGER_VIEW:
    render_manager_view(
        portfolio_payload,
        robustness_payload,
        selected_objective_label=global_research_objective_label,
        recommendation=selection_recommendation,
    )
elif dashboard_mode == RESEARCH_VIEW:
    render_dashboard_tabs(
        portfolio_payload,
        sensitivity_payload,
        robustness_payload,
        selected_objective_label=global_research_objective_label,
    )
    if selection_recommendation is not None:
        render_selection_diagnostics(selection_recommendation)
else:
    render_developer_view(
        portfolio_payload,
        sensitivity_payload,
        robustness_payload,
        selected_objective_label=global_research_objective_label,
        recommendation=selection_recommendation,
    )
