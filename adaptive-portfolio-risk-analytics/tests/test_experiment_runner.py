"""Tests for Phase 2D experiment grid execution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments import ExperimentConfig, run_experiment_grid, run_single_experiment


def _returns() -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    dates = pd.date_range("2021-01-01", periods=320, freq="B")
    data = rng.normal(0.0004, 0.01, size=(len(dates), 4))
    return pd.DataFrame(data, index=dates, columns=["A", "B", "C", "D"])


def test_single_experiment_returns_expected_metric_keys() -> None:
    returns = _returns()
    result = run_single_experiment(
        returns,
        {
            "experiment_name": "unit_single",
            "strategy": "HRP",
            "covariance_method": "sample",
            "rebalance_mode": "calendar",
            "threshold": None,
            "transaction_cost_bps": 10.0,
            "slippage_bps": 5.0,
            "vol_targeting_enabled": False,
            "target_vol": None,
            "defensive_asset": None,
            "train_window": 60,
            "initial_capital": 1_000_000.0,
        },
    )

    expected = {
        "strategy",
        "covariance_method",
        "rebalance_mode",
        "threshold",
        "transaction_cost_bps",
        "slippage_bps",
        "vol_targeting_enabled",
        "target_vol",
        "defensive_asset",
        "cagr",
        "sharpe",
        "sortino",
        "volatility",
        "max_drawdown",
        "calmar",
        "final_value",
        "total_turnover",
        "average_turnover",
        "total_transaction_cost",
        "number_of_rebalances",
        "status",
        "error",
    }
    assert expected.issubset(set(result))
    assert result["status"] == "success"


def test_grid_runner_returns_one_row_per_config() -> None:
    returns = _returns()
    config = ExperimentConfig(
        experiment_name="grid_rows",
        strategies=["HRP"],
        covariance_methods=["sample"],
        rebalance_modes=["calendar", "threshold"],
        thresholds=[0.05],
        transaction_cost_bps=[10.0],
        slippage_bps=[5.0],
        enable_vol_targeting=[False],
        target_vols=[0.10],
        defensive_assets=["Synthetic Risk-Free"],
        start_date="2021-01-01",
        end_date="2022-12-31",
        train_window=60,
        initial_capital=1_000_000.0,
    )

    results = run_experiment_grid(returns, config)

    assert len(results) == 2


def test_failed_run_is_recorded_without_stopping_all_runs() -> None:
    returns = _returns()
    config = ExperimentConfig(
        experiment_name="grid_fail",
        strategies=["HRP", "INVALID"],
        covariance_methods=["sample"],
        rebalance_modes=["calendar"],
        thresholds=[0.05],
        transaction_cost_bps=[10.0],
        slippage_bps=[5.0],
        enable_vol_targeting=[False],
        target_vols=[0.10],
        defensive_assets=["Synthetic Risk-Free"],
        start_date="2021-01-01",
        end_date="2022-12-31",
        train_window=60,
        initial_capital=1_000_000.0,
    )

    results = run_experiment_grid(returns, config)

    assert len(results) == 2
    assert (results["status"] == "failed").sum() == 1
    assert (results["status"] == "success").sum() == 1


def test_max_runs_limits_execution() -> None:
    returns = _returns()
    config = ExperimentConfig(
        experiment_name="grid_limit",
        strategies=["HRP", "HERC"],
        covariance_methods=["sample", "ledoit_wolf"],
        rebalance_modes=["calendar"],
        thresholds=[0.05],
        transaction_cost_bps=[10.0],
        slippage_bps=[5.0],
        enable_vol_targeting=[False],
        target_vols=[0.10],
        defensive_assets=["Synthetic Risk-Free"],
        start_date="2021-01-01",
        end_date="2022-12-31",
        train_window=60,
        initial_capital=1_000_000.0,
    )

    results = run_experiment_grid(returns, config, max_runs=2)

    assert len(results) == 2
