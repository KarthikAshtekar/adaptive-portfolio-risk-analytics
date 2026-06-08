"""Tests for Phase 2D experiment configuration and grid generation."""

from __future__ import annotations

from src.experiments import default_phase2d_config, generate_parameter_grid


def test_default_config_creates_valid_object() -> None:
    config = default_phase2d_config()

    assert config.experiment_name == "phase2d_default_sensitivity"
    assert config.strategies == ["HRP", "HERC"]
    assert config.covariance_methods == ["sample", "ledoit_wolf", "ewma_ledoit_wolf"]


def test_generated_grid_is_non_empty() -> None:
    config = default_phase2d_config()
    grid = generate_parameter_grid(config)

    assert not grid.empty


def test_grid_contains_expected_columns() -> None:
    config = default_phase2d_config()
    grid = generate_parameter_grid(config)

    expected = {
        "experiment_name",
        "strategy",
        "covariance_method",
        "rebalance_mode",
        "threshold",
        "transaction_cost_bps",
        "slippage_bps",
        "vol_targeting_enabled",
        "target_vol",
        "defensive_asset",
        "start_date",
        "end_date",
        "train_window",
        "initial_capital",
    }

    assert expected.issubset(set(grid.columns))
