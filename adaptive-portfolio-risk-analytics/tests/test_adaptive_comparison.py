"""Tests for Phase 3D adaptive-vs-fixed evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments import (
    build_adaptive_stress_comparison,
    compare_adaptive_vs_fixed,
)


def _adaptive_table(calmar: float = 1.2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "strategy": "Regime-Adaptive Balanced",
                "strategy_type": "regime_adaptive",
                "status": "success",
                "cagr": 0.12,
                "sharpe": 1.10,
                "calmar": calmar,
                "max_drawdown": -0.10,
                "final_value": 1_250_000.0,
                "total_turnover": 2.0,
                "total_transaction_cost": 2_000.0,
            }
        ]
    )


def _fixed_table(calmar: float = 0.8) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "strategy": "HRP",
                "strategy_type": "fixed",
                "status": "success",
                "cagr": 0.10,
                "sharpe": 0.90,
                "calmar": calmar,
                "max_drawdown": -0.14,
                "final_value": 1_200_000.0,
                "total_turnover": 1.0,
                "total_transaction_cost": 1_000.0,
            }
        ]
    )


def test_adaptive_vs_fixed_comparison_computes_deltas() -> None:
    result = compare_adaptive_vs_fixed(
        _adaptive_table(),
        _fixed_table(),
        objective="calmar",
    )

    assert result["best_adaptive_strategy"] == "Regime-Adaptive Balanced"
    assert result["best_fixed_strategy"] == "HRP"
    assert result["adaptive_minus_fixed_CAGR"] == pytest.approx(0.02)
    assert result["adaptive_minus_fixed_MaxDrawdown"] == pytest.approx(0.04)
    assert "better by the selected objective" in result["interpretation"]


def test_comparison_handles_missing_result_sets_safely() -> None:
    missing_adaptive = compare_adaptive_vs_fixed(
        pd.DataFrame(),
        _fixed_table(),
    )
    missing_fixed = compare_adaptive_vs_fixed(
        _adaptive_table(),
        pd.DataFrame(),
    )

    assert missing_adaptive["best_adaptive_strategy"] is None
    assert missing_fixed["best_fixed_strategy"] is None


def test_comparison_does_not_force_adaptive_winner() -> None:
    result = compare_adaptive_vs_fixed(
        _adaptive_table(calmar=0.5),
        _fixed_table(calmar=1.0),
        objective="calmar",
    )

    assert "not superior" in result["interpretation"]


def test_adaptive_stress_comparison_has_common_period_metrics() -> None:
    index = pd.date_range("2019-01-01", "2023-12-31", freq="B")
    benchmark_returns = pd.Series(0.0002, index=index)
    benchmark_returns.loc["2020-02-20":"2020-03-31"] = -0.01
    fixed_returns = benchmark_returns * 0.9
    adaptive_returns = benchmark_returns * 0.6

    def result(returns: pd.Series) -> dict[str, object]:
        values = (1.0 + returns).cumprod()
        return {
            "portfolio_returns": returns,
            "portfolio_values": values,
            "performance_metrics": {
                "total_turnover": 1.0,
                "total_transaction_cost": 100.0,
            },
        }

    stress = build_adaptive_stress_comparison(
        {**result(adaptive_returns), "strategy": "Adaptive"},
        {
            "HRP": result(fixed_returns),
            "Equal Weight": result(benchmark_returns),
        },
        benchmark_name="Equal Weight",
        objective="calmar",
    )

    assert {
        "COVID Crash",
        "2022 Rate/Inflation Shock",
        "Worst 1-month",
        "Worst 3-month",
        "Worst 6-month",
    }.issubset(set(stress["stress_period"]))
    assert np.isfinite(
        stress.loc[
            stress["stress_period"].eq("COVID Crash"),
            "adaptive_drawdown_reduction_vs_benchmark",
        ].iloc[0]
    )
