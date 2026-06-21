"""Tests for fold robustness summaries and CPCV experiment validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation import (
    calculate_stability_score,
    rank_by_robustness,
    run_cpcv_validation,
    summarize_fold_metrics,
)


def test_stability_score_is_finite_and_bounded() -> None:
    score = calculate_stability_score([0.8, 0.9, -0.1, 0.85])

    assert np.isfinite(score)
    assert 0.0 <= score <= 1.0
    assert calculate_stability_score([]) == 0.0


def test_summary_metrics_are_calculated_correctly() -> None:
    folds = pd.DataFrame(
        {
            "split_id": [0, 1, 2],
            "calmar": [1.0, 2.0, 3.0],
            "sharpe": [0.5, 1.0, 1.5],
            "status": ["success", "success", "success"],
        }
    )

    summary = summarize_fold_metrics(folds, metric_names=["calmar", "sharpe"])
    calmar = summary.set_index("metric").loc["calmar"]

    assert calmar["mean"] == 2.0
    assert calmar["median"] == 2.0
    assert calmar["std"] == pytest.approx(np.std([1.0, 2.0, 3.0], ddof=0))
    assert calmar["min"] == 1.0
    assert calmar["max"] == 3.0
    assert calmar["best_fold"] == 2
    assert calmar["worst_fold"] == 0


def test_robustness_ranking_sorts_stronger_configuration_first() -> None:
    summary = pd.DataFrame(
        [
            {
                "config_id": 0,
                "strategy": "Stable",
                "metric": "calmar",
                "median": 1.0,
                "min": 0.7,
                "max": 1.2,
                "stability_score": 0.95,
                "higher_is_better": True,
            },
            {
                "config_id": 1,
                "strategy": "Unstable",
                "metric": "calmar",
                "median": 0.8,
                "min": -0.5,
                "max": 2.0,
                "stability_score": 0.40,
                "higher_is_better": True,
            },
        ]
    )

    ranked = rank_by_robustness(summary, objective="calmar")

    assert ranked.iloc[0]["strategy"] == "Stable"
    assert ranked["robustness_score"].is_monotonic_decreasing


def test_robustness_ranking_uses_passed_objective_and_calmar_fallback() -> None:
    summary = pd.DataFrame(
        [
            {
                "config_id": 0,
                "strategy": "Calmar Leader",
                "metric": "calmar",
                "median": 2.0,
                "min": 1.0,
                "max": 2.5,
                "stability_score": 0.90,
                "higher_is_better": True,
            },
            {
                "config_id": 1,
                "strategy": "Sharpe Leader",
                "metric": "sharpe",
                "median": 3.0,
                "min": 2.0,
                "max": 3.5,
                "stability_score": 0.95,
                "higher_is_better": True,
            },
        ]
    )

    sharpe_ranking = rank_by_robustness(summary, objective="Sharpe")
    fallback_ranking = rank_by_robustness(summary, objective=None)

    assert sharpe_ranking["metric"].eq("sharpe").all()
    assert sharpe_ranking.iloc[0]["strategy"] == "Sharpe Leader"
    assert fallback_ranking["metric"].eq("calmar").all()
    assert fallback_ranking.iloc[0]["strategy"] == "Calmar Leader"


def test_cpcv_runner_returns_all_result_tables() -> None:
    rng = np.random.default_rng(20260618)
    index = pd.date_range("2022-01-03", periods=160, freq="B")
    returns = pd.DataFrame(
        rng.normal(0.0004, 0.01, size=(len(index), 3)),
        index=index,
        columns=["A", "B", "C"],
    )
    configs = [
        {
            "experiment_name": "cpcv_unit",
            "strategy": "Equal Weight",
            "covariance_method": "sample",
            "rebalance_mode": "calendar",
            "threshold": None,
            "transaction_cost_bps": 10.0,
            "slippage_bps": 5.0,
            "vol_targeting_enabled": False,
            "target_vol": None,
            "defensive_asset": None,
            "train_window": 20,
            "initial_capital": 1_000_000.0,
        }
    ]

    result = run_cpcv_validation(
        returns,
        configs,
        n_blocks=4,
        n_test_blocks=1,
        embargo_pct=0.0,
        objective="calmar",
        max_configs=1,
    )

    assert set(result) == {
        "fold_results",
        "summary_table",
        "robustness_ranking",
        "split_diagnostics",
    }
    assert len(result["split_diagnostics"]) == 4
    assert (result["fold_results"]["status"] == "success").any()
    assert (result["fold_results"]["status"] == "failed").any()
    assert not result["summary_table"].empty
    assert not result["robustness_ranking"].empty


def test_fixed_cpcv_final_value_includes_transaction_cost_drag() -> None:
    rng = np.random.default_rng(20260619)
    index = pd.date_range("2020-01-02", periods=320, freq="B")
    common = rng.normal(0.0003, 0.004, len(index))
    returns = pd.DataFrame(
        {
            "Low Vol": common + rng.normal(0.0, 0.002, len(index)),
            "Medium Vol": common + rng.normal(0.0, 0.008, len(index)),
            "High Vol": common + rng.normal(0.0, 0.016, len(index)),
        },
        index=index,
    )
    base_config = {
        "experiment_name": "cpcv_cost_parity",
        "strategy": "Inverse Volatility",
        "strategy_type": "fixed",
        "covariance_method": "sample",
        "rebalance_mode": "calendar",
        "threshold": 0.05,
        "vol_targeting_enabled": False,
        "target_vol": None,
        "defensive_asset": None,
        "train_window": 40,
        "initial_capital": 1_000_000.0,
    }
    configs = [
        {
            **base_config,
            "transaction_cost_bps": 0.0,
            "slippage_bps": 0.0,
        },
        {
            **base_config,
            "transaction_cost_bps": 100.0,
            "slippage_bps": 50.0,
        },
    ]

    result = run_cpcv_validation(
        returns,
        configs,
        n_blocks=4,
        n_test_blocks=1,
        embargo_pct=0.0,
        objective="final_value",
    )
    successful = result["fold_results"].loc[
        result["fold_results"]["status"].eq("success"),
        ["config_id", "split_id", "final_value"],
    ]
    paired = successful.pivot(
        index="split_id",
        columns="config_id",
        values="final_value",
    ).dropna()

    assert not paired.empty
    assert (paired[1] < paired[0]).all()
