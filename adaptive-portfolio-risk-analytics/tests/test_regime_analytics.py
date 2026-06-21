"""Tests for Phase 3B regime performance and transition analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.regime import (
    calculate_regime_performance,
    calculate_regime_transitions,
    calculate_strategy_regime_summary,
    select_best_strategy_by_regime,
)


def _series(values, name: str) -> pd.Series:
    return pd.Series(
        values,
        index=pd.date_range("2024-01-01", periods=len(values), freq="B"),
        name=name,
        dtype=float,
    )


def test_regime_counts_and_performance_metrics_are_calculated() -> None:
    strategy = _series([0.01, 0.02, -0.01, -0.02, 0.005, 0.01], "strategy")
    regimes = pd.Series(
        ["Calm", "Calm", "Stress", "Stress", "Normal", "Normal"],
        index=strategy.index,
        name="regime",
    )

    performance = calculate_regime_performance(strategy, regimes)
    calm = performance.set_index("regime").loc["Calm"]

    assert calm["number_of_days"] == 2
    assert np.isfinite(calm["strategy_cagr"])
    assert np.isfinite(calm["strategy_volatility"])
    assert "strategy_max_drawdown" in performance.columns


def test_benchmark_comparison_and_multi_strategy_summary_work() -> None:
    strategy_a = _series([0.02, 0.01, -0.01, 0.005], "a")
    strategy_b = _series([0.01, 0.005, -0.02, 0.001], "b")
    benchmark = _series([0.005, 0.004, -0.015, 0.0], "benchmark")
    regimes = pd.Series(
        ["Calm", "Calm", "Stress", "Stress"],
        index=strategy_a.index,
        name="regime",
    )

    summary = calculate_strategy_regime_summary(
        {"A": strategy_a, "B": strategy_b},
        regimes,
        benchmark_returns=benchmark,
        objective="sharpe",
    )

    performance = summary["performance"]
    assert set(performance["strategy"]) == {"A", "B"}
    assert performance["benchmark_cagr"].notna().all()
    assert performance["hit_ratio_vs_benchmark"].between(0.0, 1.0).all()
    assert not summary["best_strategy_by_regime"].empty
    assert summary["objective"] == "sharpe"


def test_regime_distribution_counts_are_correct() -> None:
    returns = _series([0.01, 0.01, -0.01, 0.0], "strategy")
    regimes = pd.Series(
        ["Calm", "Calm", "Stress", "Normal"],
        index=returns.index,
        name="regime",
    )

    summary = calculate_strategy_regime_summary({"A": returns}, regimes)
    distribution = summary["regime_distribution"].set_index("regime")

    assert distribution.loc["Calm", "number_of_days"] == 2
    assert distribution.loc["Calm", "percentage_of_days"] == 0.5


def test_transition_matrix_and_durations_are_correct() -> None:
    index = pd.date_range("2024-01-01", periods=7, freq="B")
    regimes = pd.Series(
        ["Calm", "Calm", "Normal", "Stress", "Stress", "Stress", "Normal"],
        index=index,
        name="regime",
    )

    transitions = calculate_regime_transitions(regimes)
    counts = transitions["transition_count_matrix"]
    durations = transitions["average_duration"].set_index("regime")

    assert counts.loc["Calm", "Calm"] == 1
    assert counts.loc["Calm", "Normal"] == 1
    assert counts.loc["Normal", "Stress"] == 1
    assert counts.loc["Stress", "Stress"] == 2
    assert durations.loc["Stress", "average_duration"] == 3.0
    assert transitions["current_regime"] == "Normal"
    assert transitions["current_regime_duration"] == 1


def test_best_strategy_selection_uses_requested_objective() -> None:
    performance = pd.DataFrame(
        {
            "strategy": ["A", "B"],
            "regime": ["Stress", "Stress"],
            "strategy_sharpe": [2.0, 1.0],
            "strategy_max_drawdown": [-0.20, -0.10],
        }
    )

    selection = select_best_strategy_by_regime(
        performance,
        objective="max_drawdown",
    )

    assert selection["objective"] == "max_drawdown"
    assert selection["fallback_used"] is False
    assert selection["table"].iloc[0]["best_strategy"] == "B"
