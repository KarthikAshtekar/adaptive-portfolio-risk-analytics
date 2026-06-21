"""Experiment configuration objects for Phase 2D research orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

ADAPTIVE_REGIME_SOURCES = {
    "rule_based_lagged",
    "hmm_walk_forward",
}
ADAPTIVE_POLICY_PRESETS = {
    "conservative",
    "balanced",
    "aggressive",
}
FULL_SAMPLE_HMM_ERROR = (
    "Full-sample HMM is historical-only and cannot be used for " "trading-safe adaptive backtests."
)


def normalize_adaptive_regime_source(source: str) -> str:
    """Normalize and validate a trading-safe adaptive regime source."""
    normalized = str(source).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "rule_based": "rule_based_lagged",
        "rule_based_lagged_decision_regime": "rule_based_lagged",
        "hmm_walk_forward_decision_regime": "hmm_walk_forward",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in {"hmm_full_sample", "full_sample_hmm", "hmm_historical"}:
        raise ValueError(FULL_SAMPLE_HMM_ERROR)
    if normalized not in ADAPTIVE_REGIME_SOURCES:
        supported = ", ".join(sorted(ADAPTIVE_REGIME_SOURCES))
        raise ValueError(f"unsupported adaptive regime source '{source}'. Supported: {supported}")
    return normalized


def normalize_adaptive_policy_preset(preset: str) -> str:
    """Normalize an adaptive policy preset name."""
    normalized = str(preset).strip().lower().replace("_", " ")
    if normalized in {"balanced default", "default"}:
        normalized = "balanced"
    if normalized not in ADAPTIVE_POLICY_PRESETS:
        supported = ", ".join(sorted(ADAPTIVE_POLICY_PRESETS))
        raise ValueError(f"unsupported adaptive policy preset '{preset}'. Supported: {supported}")
    return normalized


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
    defensive_annual_rate: float = 0.04
    defensive_fallback: str = "synthetic"

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
        if float(self.defensive_annual_rate) <= -1.0:
            raise ValueError("defensive_annual_rate must be greater than -1")
        if str(self.defensive_fallback).strip().lower() not in {
            "synthetic",
            "cash_zero",
        }:
            raise ValueError("defensive_fallback must be synthetic or cash_zero")


@dataclass(frozen=True)
class AdaptiveExperimentConfig:
    """Compact Phase 3D grid for trading-safe adaptive strategy research."""

    experiment_name: str
    regime_sources: list[str] = field(default_factory=list)
    policy_presets: list[str] = field(default_factory=list)
    training_windows: list[int] = field(default_factory=list)
    defensive_assets: list[str] = field(default_factory=list)
    transaction_cost_bps: list[float] = field(default_factory=list)
    slippage_bps: list[float] = field(default_factory=list)
    rebalance_frequencies: list[str] = field(default_factory=lambda: ["M"])
    hmm_n_states: int = 4
    hmm_min_train_size: int = 504
    hmm_refit_frequency: int = 21
    hmm_covariance_type: str = "diag"
    hmm_decision_lag: int = 1
    initial_capital: float = 1_000_000.0
    defensive_annual_rate: float = 0.04
    defensive_fallback: str = "synthetic"

    def __post_init__(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must not be empty")
        if not self.regime_sources:
            raise ValueError("regime_sources must not be empty")
        if not self.policy_presets:
            raise ValueError("policy_presets must not be empty")
        if not self.training_windows:
            raise ValueError("training_windows must not be empty")
        if not self.defensive_assets:
            raise ValueError("defensive_assets must not be empty")
        if not self.transaction_cost_bps:
            raise ValueError("transaction_cost_bps must not be empty")
        if not self.slippage_bps:
            raise ValueError("slippage_bps must not be empty")
        if not self.rebalance_frequencies:
            raise ValueError("rebalance_frequencies must not be empty")

        for source in self.regime_sources:
            normalize_adaptive_regime_source(source)
        for preset in self.policy_presets:
            normalize_adaptive_policy_preset(preset)
        if any(int(window) < 20 for window in self.training_windows):
            raise ValueError("adaptive training windows must be at least 20")
        if any(str(value).upper() not in {"M", "W", "Q"} for value in self.rebalance_frequencies):
            raise ValueError("rebalance frequencies must be one of: M, W, Q")
        if int(self.hmm_n_states) < 2:
            raise ValueError("hmm_n_states must be at least 2")
        if int(self.hmm_min_train_size) <= 0:
            raise ValueError("hmm_min_train_size must be positive")
        if int(self.hmm_refit_frequency) <= 0:
            raise ValueError("hmm_refit_frequency must be positive")
        if int(self.hmm_decision_lag) < 1:
            raise ValueError("hmm_decision_lag must be at least 1")
        if float(self.initial_capital) <= 0.0:
            raise ValueError("initial_capital must be positive")
        if float(self.defensive_annual_rate) <= -1.0:
            raise ValueError("defensive_annual_rate must be greater than -1")
        if str(self.defensive_fallback).strip().lower() not in {
            "synthetic",
            "cash_zero",
        }:
            raise ValueError("defensive_fallback must be synthetic or cash_zero")


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
