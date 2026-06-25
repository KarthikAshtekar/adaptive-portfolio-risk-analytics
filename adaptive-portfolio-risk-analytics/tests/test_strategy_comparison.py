"""Tests for benchmark-aware strategy comparison utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.benchmarks import (
    build_performance_comparison_table,
    compute_relative_performance,
    run_strategy_comparison,
)


@pytest.fixture
def deterministic_returns() -> pd.DataFrame:
    rng = np.random.default_rng(2041)
    dates = pd.date_range(start="2021-01-01", periods=320, freq="B")
    common_factor = rng.normal(0.0002, 0.004, size=(len(dates), 1))
    idiosyncratic = rng.normal(
        loc=0.0003,
        scale=np.array([0.008, 0.011, 0.014, 0.010]),
        size=(len(dates), 4),
    )
    return pd.DataFrame(
        common_factor + idiosyncratic,
        index=dates,
        columns=["A", "B", "C", "D"],
    )


def test_run_strategy_comparison_returns_all_requested_strategies(
    deterministic_returns: pd.DataFrame,
) -> None:
    strategy_results = run_strategy_comparison(
        deterministic_returns,
        strategy_names=["Equal Weight", "Inverse Volatility", "HRP", "HERC"],
        train_window=60,
        rebalance_frequency="M",
    )

    assert list(strategy_results.keys()) == [
        "Equal Weight",
        "Inverse Volatility",
        "HRP",
        "HERC",
    ]


def test_each_strategy_result_contains_expected_keys(
    deterministic_returns: pd.DataFrame,
) -> None:
    strategy_results = run_strategy_comparison(
        deterministic_returns,
        strategy_names=["Equal Weight", "HRP"],
        train_window=60,
        rebalance_frequency="M",
    )

    for result in strategy_results.values():
        for key in (
            "portfolio_returns",
            "portfolio_values",
            "drawdown",
            "weights_history",
            "performance_metrics",
            "latest_weights",
        ):
            assert key in result


def test_performance_comparison_table_has_expected_columns(
    deterministic_returns: pd.DataFrame,
) -> None:
    strategy_results = run_strategy_comparison(
        deterministic_returns,
        strategy_names=["Equal Weight", "HRP"],
        train_window=60,
    )
    performance_df = build_performance_comparison_table(strategy_results)

    assert list(performance_df.columns) == [
        "cumulative_return",
        "cagr",
        "sharpe",
        "sortino",
        "volatility",
        "max_drawdown",
        "calmar",
        "pain_index",
        "pain_ratio",
        "var_95",
        "cvar_95",
        "final_value",
    ]


def test_relative_performance_table_has_expected_columns(
    deterministic_returns: pd.DataFrame,
) -> None:
    strategy_results = run_strategy_comparison(
        deterministic_returns,
        strategy_names=["Equal Weight", "HRP", "HERC"],
        train_window=60,
    )
    performance_df = build_performance_comparison_table(strategy_results)
    relative_df = compute_relative_performance(performance_df, benchmark_name="Equal Weight")

    assert list(relative_df.columns) == [
        "strategy",
        "benchmark",
        "excess_cagr",
        "excess_sharpe",
        "drawdown_difference",
        "volatility_difference",
        "final_value_difference",
    ]


def test_final_values_are_positive(deterministic_returns: pd.DataFrame) -> None:
    strategy_results = run_strategy_comparison(
        deterministic_returns,
        strategy_names=["Equal Weight", "Inverse Volatility", "HRP", "HERC"],
        train_window=60,
    )
    performance_df = build_performance_comparison_table(strategy_results)

    assert (performance_df["final_value"] > 0.0).all()


def test_no_strategy_output_is_empty(deterministic_returns: pd.DataFrame) -> None:
    strategy_results = run_strategy_comparison(
        deterministic_returns,
        strategy_names=["Equal Weight", "Inverse Volatility", "HRP", "HERC"],
        train_window=60,
    )

    for result in strategy_results.values():
        assert not result["portfolio_returns"].empty
        assert not result["portfolio_values"].empty
        assert not result["weights_history"].empty


@pytest.mark.parametrize("covariance_method", ["sample", "ledoit_wolf"])
def test_comparison_works_with_supported_covariance_methods(
    deterministic_returns: pd.DataFrame,
    covariance_method: str,
) -> None:
    strategy_results = run_strategy_comparison(
        deterministic_returns,
        strategy_names=["HRP", "HERC"],
        covariance_method=covariance_method,
        train_window=60,
    )
    performance_df = build_performance_comparison_table(strategy_results)

    assert not performance_df.empty
    assert set(performance_df.index) == {"HRP", "HERC"}
