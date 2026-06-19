"""Tests for Phase 3D adaptive experiment configuration and execution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.experiments.adaptive as adaptive_experiments
from src.experiments import (
    FULL_SAMPLE_HMM_ERROR,
    AdaptiveExperimentConfig,
    generate_adaptive_parameter_grid,
    run_adaptive_experiment_grid,
)


def _returns(periods: int = 280) -> pd.DataFrame:
    rng = np.random.default_rng(20260619)
    index = pd.date_range("2021-01-04", periods=periods, freq="B")
    common = rng.normal(0.0003, 0.006, periods)
    return pd.DataFrame(
        {
            "A": common + rng.normal(0.0, 0.005, periods),
            "B": common + rng.normal(0.0, 0.007, periods),
            "C": common + rng.normal(0.0, 0.009, periods),
        },
        index=index,
    )


def _config(
    *,
    sources: list[str] | None = None,
    presets: list[str] | None = None,
) -> AdaptiveExperimentConfig:
    return AdaptiveExperimentConfig(
        experiment_name="phase3d_unit",
        regime_sources=sources or ["rule_based_lagged"],
        policy_presets=presets or ["balanced"],
        training_windows=[40],
        defensive_assets=["Synthetic Risk-Free"],
        transaction_cost_bps=[10.0],
        slippage_bps=[5.0],
    )


def test_adaptive_configs_are_generated_and_limited() -> None:
    grid = generate_adaptive_parameter_grid(
        _config(presets=["conservative", "balanced", "aggressive"]),
        max_adaptive_configs=2,
    )

    assert len(grid) == 2
    assert grid["strategy_type"].eq("regime_adaptive").all()
    assert {
        "regime_source",
        "policy_preset",
        "training_window",
        "hmm_n_states",
        "hmm_min_train_size",
        "hmm_refit_frequency",
    }.issubset(grid.columns)


def test_full_sample_hmm_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="historical-only"):
        _config(sources=["hmm_full_sample"])

    assert "historical-only" in FULL_SAMPLE_HMM_ERROR


def test_rule_based_adaptive_strategy_runs_in_grid() -> None:
    result = run_adaptive_experiment_grid(_returns(), _config())
    row = result["results"].iloc[0]

    assert row["status"] == "success"
    assert row["regime_source"] == "rule_based_lagged"
    assert str(row["config_id"]) in result["backtests"]


def test_hmm_unavailable_is_skipped_cleanly(monkeypatch) -> None:
    monkeypatch.setattr(adaptive_experiments, "HMM_AVAILABLE", False)

    result = run_adaptive_experiment_grid(
        _returns(),
        _config(sources=["hmm_walk_forward"]),
    )

    assert result["results"].iloc[0]["status"] == "skipped"
    assert result["warnings"]
    assert "hmmlearn" in result["warnings"][0]


def test_adaptive_metrics_include_standard_and_specific_diagnostics() -> None:
    result = run_adaptive_experiment_grid(_returns(), _config())
    row = result["results"].iloc[0]

    expected = {
        "cagr",
        "volatility",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "final_value",
        "total_turnover",
        "total_transaction_cost",
        "number_of_rebalances",
        "var_95",
        "cvar_95",
        "max_drawdown_duration",
        "average_risky_exposure",
        "minimum_risky_exposure",
        "maximum_risky_exposure",
        "average_defensive_weight",
        "maximum_defensive_weight",
        "number_of_policy_switches",
        "most_common_regime",
        "allocator_usage_distribution",
        "covariance_method_usage_distribution",
    }
    assert expected.issubset(result["results"].columns)
    assert 0.0 <= row["average_risky_exposure"] <= 1.0
