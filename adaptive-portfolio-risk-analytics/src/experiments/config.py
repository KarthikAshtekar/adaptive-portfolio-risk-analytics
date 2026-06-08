"""Experiment configuration objects for Phase 2D research orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    strategies: list[str] = field(default_factory=list)
    covariance_methods: list[str] = field(default_factory=list)
    rebalance_modes: list[str] = field(default_factory=list)
    thresholds: list[float] = field(default_factory=list)
    transaction_cost_bps: list[float] = field(default_factory=list)
    slippage_bps: list[float] = field(default_factory=list)
    enable_vol_targeting: list[bool] = field(default_factory=list)
    target_vols: list[float] = field(default_factory=list)
    defensive_assets: list[str] = field(default_factory=list)
    start_date: str = "2020-01-01"
    end_date: str = "2025-01-01"
    train_window: int = 252
    initial_capital: float = 1_000_000.0

    def __post_init__(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must not be empty")
        if not self.strategies:
            raise ValueError("strategies must not be empty")
        if not self.covariance_methods:
            raise ValueError("covariance_methods must not be empty")
        if not self.rebalance_modes:
            raise ValueError("rebalance_modes must not be empty")
        if not self.thresholds:
            raise ValueError("thresholds must not be empty")
        if not self.transaction_cost_bps:
            raise ValueError("transaction_cost_bps must not be empty")
        if not self.slippage_bps:
            raise ValueError("slippage_bps must not be empty")
        if not self.enable_vol_targeting:
            raise ValueError("enable_vol_targeting must not be empty")
        if not self.target_vols:
            raise ValueError("target_vols must not be empty")
        if not self.defensive_assets:
            raise ValueError("defensive_assets must not be empty")
        if self.train_window < 20:
            raise ValueError("train_window must be >= 20")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")


def default_phase2d_config() -> ExperimentConfig:
    """Return a compact local-friendly sensitivity grid for Phase 2D."""
    return ExperimentConfig(
        experiment_name="phase2d_default_sensitivity",
        strategies=["HRP", "HERC"],
        covariance_methods=["sample", "ledoit_wolf", "ewma_ledoit_wolf"],
        rebalance_modes=["calendar", "threshold"],
        thresholds=[0.03, 0.05, 0.10],
        transaction_cost_bps=[10.0],
        slippage_bps=[5.0],
        enable_vol_targeting=[False, True],
        target_vols=[0.10],
        defensive_assets=["Synthetic Risk-Free"],
        start_date="2020-01-01",
        end_date="2025-01-01",
        train_window=252,
        initial_capital=1_000_000.0,
    )
