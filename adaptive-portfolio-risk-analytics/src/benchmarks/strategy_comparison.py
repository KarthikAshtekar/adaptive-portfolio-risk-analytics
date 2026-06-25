"""Strategy comparison utilities for benchmark-aware research workflows."""

from __future__ import annotations

import pandas as pd

from src.analytics import PerformanceAnalytics
from src.backtesting import RollingBacktester
from src.backtesting.transaction_costs import TransactionCostModel

from .benchmark_factory import BenchmarkFactory


def run_strategy_comparison(
    returns_df: pd.DataFrame,
    strategy_names: list[str],
    covariance_method: str = "sample",
    train_window: int = 252,
    rebalance_frequency: str = "M",
    initial_capital: float = 1_000_000.0,
    covariance_kwargs: dict | None = None,
    rebalance_mode: str = "calendar",
    threshold: float = 0.05,
    transaction_cost_model: TransactionCostModel | None = None,
) -> dict[str, dict]:
    """Run the requested strategies and return benchmark-ready results."""
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame")
    if returns_df.empty:
        raise ValueError("returns_df must not be empty")
    if not strategy_names:
        raise ValueError("strategy_names must not be empty")

    strategy_results: dict[str, dict] = {}
    covariance_kwargs = dict(covariance_kwargs or {})

    for strategy_name in strategy_names:
        display_name = BenchmarkFactory.normalize_strategy_name(strategy_name)
        allocator = BenchmarkFactory.get_allocator(
            strategy_name=display_name,
            covariance_method=covariance_method,
            covariance_kwargs=covariance_kwargs,
        )
        results = RollingBacktester(
            allocator=allocator,
            train_window=train_window,
            rebalance_frequency=rebalance_frequency,
            initial_capital=initial_capital,
            rebalance_mode=rebalance_mode,
            threshold=threshold,
            transaction_cost_model=transaction_cost_model,
        ).run(returns_df)

        latest_weights = (
            results["weights_history"].iloc[-1].copy()
            if not results["weights_history"].empty
            else pd.Series(dtype=float, name="weight")
        )
        results["latest_weights"] = latest_weights
        strategy_results[display_name] = results

    return strategy_results


def build_performance_comparison_table(
    strategy_results: dict[str, dict],
) -> pd.DataFrame:
    """Build a standardized multi-strategy performance comparison table."""
    if not strategy_results:
        raise ValueError("strategy_results must not be empty")

    rows: dict[str, dict[str, float]] = {}
    for strategy_name, result in strategy_results.items():
        portfolio_returns = result["portfolio_returns"]
        portfolio_values = result["portfolio_values"]
        summary = PerformanceAnalytics.summary_table(portfolio_returns)
        rows[strategy_name] = {
            "cumulative_return": summary["cumulative_return"],
            "cagr": summary["cagr"],
            "sharpe": summary["sharpe"],
            "sortino": summary["sortino"],
            "volatility": summary["volatility"],
            "max_drawdown": summary["max_drawdown"],
            "calmar": summary["calmar"],
            "pain_index": summary["pain_index"],
            "pain_ratio": summary["pain_ratio"],
            "var_95": summary["var_95"],
            "cvar_95": summary["cvar_95"],
            "final_value": float(portfolio_values.iloc[-1]),
        }

    return pd.DataFrame.from_dict(rows, orient="index")


def compute_relative_performance(
    performance_comparison_df: pd.DataFrame,
    benchmark_name: str,
) -> pd.DataFrame:
    """Compute strategy performance relative to the chosen benchmark."""
    if not isinstance(performance_comparison_df, pd.DataFrame):
        raise TypeError("performance_comparison_df must be a pandas DataFrame")
    if performance_comparison_df.empty:
        raise ValueError("performance_comparison_df must not be empty")

    benchmark_display_name = BenchmarkFactory.normalize_strategy_name(benchmark_name)
    if benchmark_display_name not in performance_comparison_df.index:
        raise ValueError("benchmark_name must be present in performance_comparison_df index")

    benchmark_row = performance_comparison_df.loc[benchmark_display_name]
    rows = []
    for strategy_name, row in performance_comparison_df.iterrows():
        rows.append(
            {
                "strategy": strategy_name,
                "benchmark": benchmark_display_name,
                "excess_cagr": row["cagr"] - benchmark_row["cagr"],
                "excess_sharpe": row["sharpe"] - benchmark_row["sharpe"],
                "drawdown_difference": row["max_drawdown"] - benchmark_row["max_drawdown"],
                "volatility_difference": row["volatility"] - benchmark_row["volatility"],
                "final_value_difference": row["final_value"] - benchmark_row["final_value"],
            }
        )

    return pd.DataFrame(rows)
