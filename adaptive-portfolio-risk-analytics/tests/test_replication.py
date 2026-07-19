"""Tests for the bounded Phase 3E replication and tuning harness."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments import (
    run_policy_tuning_study,
    run_replication_study,
    summarize_replication_results,
)


def _returns(periods: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(20260620)
    index = pd.date_range("2020-01-02", periods=periods, freq="B")
    common = rng.normal(0.0003, 0.006, periods)
    return pd.DataFrame(
        {
            "A": common + rng.normal(0.0, 0.004, periods),
            "B": common + rng.normal(0.0, 0.006, periods),
            "C": common + rng.normal(0.0, 0.008, periods),
            "D": common + rng.normal(0.0, 0.010, periods),
        },
        index=index,
    )


def test_replication_runs_complete_bounded_scenario() -> None:
    returns = _returns()
    result = run_replication_study(
        universes={"Synthetic Universe": returns},
        date_windows=[("2020-06-01", None)],
        cost_scenarios=[(10.0, 5.0)],
        defensive_sleeves=["synthetic_4pct"],
        policy_presets=["conservative"],
        regime_sources=["rule_based_lagged"],
        max_runs=4,
    )

    assert len(result) == 4
    assert {"Equal Weight", "HRP", "HERC"}.issubset(set(result["strategy"]))
    assert result["strategy_type"].eq("regime_adaptive").sum() == 1
    assert result["status"].eq("success").all()
    assert (
        result.loc[
            result["strategy_type"].eq("regime_adaptive"),
            "defensive_source_used",
        ]
        .eq("synthetic")
        .all()
    )


def test_replication_summary_classifies_adaptive_without_forcing_win() -> None:
    results = run_replication_study(
        universes={"Synthetic Universe": _returns()},
        date_windows=[("2020-06-01", None)],
        cost_scenarios=[(0.0, 0.0), (25.0, 10.0)],
        defensive_sleeves=["synthetic_4pct"],
        policy_presets=["conservative"],
        regime_sources=["rule_based_lagged"],
        max_runs=8,
    )

    summary = summarize_replication_results(results)

    assert not summary.empty
    assert (
        summary["classification"]
        .isin(
            {
                "First-class main strategy",
                "Risk-control overlay",
                "Experimental only",
            }
        )
        .all()
    )
    assert summary["number_of_successful_runs"].gt(0).all()


def test_policy_tuning_outputs_requested_comparisons() -> None:
    result = run_policy_tuning_study(
        _returns(),
        regime_sources=["rule_based_lagged"],
        evaluation_start="2020-06-01",
        max_variants=2,
    )

    assert result["status"].eq("success").all()
    assert set(result["policy_variant"]) == {
        "Conservative base",
        "Conservative faster re-risking",
    }
    assert {
        "cagr",
        "calmar",
        "max_drawdown",
        "final_value",
        "recovery_duration",
        "turnover",
        "transaction_cost",
        "stress_period_return",
        "best_recovery_improvement",
        "best_calmar_improvement",
    }.issubset(result.columns)
