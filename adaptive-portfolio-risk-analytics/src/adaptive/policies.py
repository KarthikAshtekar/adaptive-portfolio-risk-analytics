"""Regime policy definitions and presets for Phase 3C."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import isfinite
from typing import Mapping

import pandas as pd

from src.covariance.covariance_factory import SUPPORTED_COVARIANCE_METHODS

SUPPORTED_ALLOCATORS = {
    "equal_weight",
    "inverse_volatility",
    "hrp",
    "herc",
}
SUPPORTED_REBALANCE_MODES = {
    "calendar",
    "threshold",
    "calendar_or_threshold",
}
REQUIRED_REGIMES = {"Calm", "Normal", "Stress", "Crisis", "Unknown"}


@dataclass(frozen=True)
class RegimePolicy:
    """Portfolio behavior selected for one market regime."""

    regime: str
    allocator: str
    covariance_method: str
    target_volatility: float
    rebalance_mode: str
    rebalance_threshold: float
    defensive_weight_floor: float
    risky_exposure_cap: float
    notes: str = ""


DEFAULT_REGIME_POLICY = {
    "Calm": RegimePolicy(
        regime="Calm",
        allocator="herc",
        covariance_method="ewma_ledoit_wolf",
        target_volatility=0.12,
        rebalance_mode="threshold",
        rebalance_threshold=0.20,
        defensive_weight_floor=0.00,
        risky_exposure_cap=1.00,
        notes="Risk-on regime: allow higher risky exposure.",
    ),
    "Normal": RegimePolicy(
        regime="Normal",
        allocator="herc",
        covariance_method="ledoit_wolf",
        target_volatility=0.10,
        rebalance_mode="threshold",
        rebalance_threshold=0.10,
        defensive_weight_floor=0.05,
        risky_exposure_cap=1.00,
        notes="Balanced regime.",
    ),
    "Stress": RegimePolicy(
        regime="Stress",
        allocator="hrp",
        covariance_method="ledoit_wolf",
        target_volatility=0.06,
        rebalance_mode="threshold",
        rebalance_threshold=0.05,
        defensive_weight_floor=0.20,
        risky_exposure_cap=0.70,
        notes="Risk-off regime: lower target volatility and cap risky exposure.",
    ),
    "Crisis": RegimePolicy(
        regime="Crisis",
        allocator="hrp",
        covariance_method="ewma_ledoit_wolf",
        target_volatility=0.03,
        rebalance_mode="calendar_or_threshold",
        rebalance_threshold=0.03,
        defensive_weight_floor=0.40,
        risky_exposure_cap=0.40,
        notes="Crisis regime: defensive allocation and low risky exposure.",
    ),
    "Unknown": RegimePolicy(
        regime="Unknown",
        allocator="equal_weight",
        covariance_method="ledoit_wolf",
        target_volatility=0.08,
        rebalance_mode="calendar",
        rebalance_threshold=0.10,
        defensive_weight_floor=0.10,
        risky_exposure_cap=0.80,
        notes="Fallback when regime is not available.",
    ),
}


def _normalize_policy_map(
    policy_map: Mapping[str, RegimePolicy] | None,
) -> dict[str, RegimePolicy]:
    return dict(policy_map or DEFAULT_REGIME_POLICY)


def validate_policy_map(
    policy_map: Mapping[str, RegimePolicy],
) -> dict[str, RegimePolicy]:
    """Validate policy coverage and portfolio-control bounds."""
    if not isinstance(policy_map, Mapping):
        raise TypeError("policy_map must be a mapping")
    normalized = dict(policy_map)
    missing = REQUIRED_REGIMES - set(normalized)
    if missing:
        raise ValueError("policy_map is missing required regimes: " + ", ".join(sorted(missing)))

    for regime, policy in normalized.items():
        if not isinstance(policy, RegimePolicy):
            raise TypeError(f"policy for '{regime}' must be a RegimePolicy")
        if policy.regime != regime:
            raise ValueError(f"policy regime '{policy.regime}' must match key '{regime}'")
        if policy.allocator not in SUPPORTED_ALLOCATORS:
            raise ValueError(f"unsupported allocator '{policy.allocator}'")
        if policy.covariance_method not in SUPPORTED_COVARIANCE_METHODS:
            raise ValueError(f"unsupported covariance method '{policy.covariance_method}'")
        if not isfinite(policy.target_volatility) or policy.target_volatility <= 0.0:
            raise ValueError("target_volatility must be finite and positive")
        if policy.rebalance_mode not in SUPPORTED_REBALANCE_MODES:
            raise ValueError(f"unsupported rebalance mode '{policy.rebalance_mode}'")
        if not 0.0 <= policy.rebalance_threshold <= 1.0:
            raise ValueError("rebalance_threshold must be between 0 and 1")
        if not 0.0 <= policy.defensive_weight_floor <= 1.0:
            raise ValueError("defensive_weight_floor must be between 0 and 1")
        if not 0.0 <= policy.risky_exposure_cap <= 1.0:
            raise ValueError("risky_exposure_cap must be between 0 and 1")
    return normalized


def get_policy_for_regime(
    regime,
    policy_map: Mapping[str, RegimePolicy] | None = None,
) -> RegimePolicy:
    """Return a policy for rule-based or two-state HMM regime labels."""
    policies = validate_policy_map(_normalize_policy_map(policy_map))
    normalized = str(regime or "Unknown").strip().lower()
    aliases = {
        "calm": "Calm",
        "normal": "Normal",
        "stress": "Stress",
        "crisis": "Crisis",
        "unknown": "Unknown",
        "risk-on": "Calm",
        "risk_on": "Calm",
        "risk off": "Stress",
        "risk-off": "Stress",
        "risk_off": "Stress",
    }
    return policies.get(aliases.get(normalized, "Unknown"), policies["Unknown"])


def policy_map_to_dataframe(
    policy_map: Mapping[str, RegimePolicy] | None = None,
) -> pd.DataFrame:
    """Return a stable dashboard/report representation of a policy map."""
    policies = validate_policy_map(_normalize_policy_map(policy_map))
    order = ["Calm", "Normal", "Stress", "Crisis", "Unknown"]
    return pd.DataFrame([asdict(policies[regime]) for regime in order])


def _scaled_policy(
    policy: RegimePolicy,
    *,
    target_multiplier: float,
    defensive_addition: float,
    cap_addition: float,
    preset_name: str,
) -> RegimePolicy:
    defensive_floor = float(min(1.0, max(0.0, policy.defensive_weight_floor + defensive_addition)))
    risky_cap = float(min(1.0, max(0.0, policy.risky_exposure_cap + cap_addition)))
    return replace(
        policy,
        target_volatility=float(policy.target_volatility * target_multiplier),
        defensive_weight_floor=defensive_floor,
        risky_exposure_cap=risky_cap,
        notes=f"{policy.notes} {preset_name} preset.",
    )


def get_policy_preset(name: str) -> dict[str, RegimePolicy]:
    """Return Conservative, Balanced default, or Aggressive policy settings."""
    normalized = str(name).strip().lower()
    if normalized in {"balanced", "balanced default", "default"}:
        return dict(DEFAULT_REGIME_POLICY)
    if normalized == "conservative":
        return {
            regime: _scaled_policy(
                policy,
                target_multiplier=0.80,
                defensive_addition=0.10,
                cap_addition=-0.10,
                preset_name="Conservative",
            )
            for regime, policy in DEFAULT_REGIME_POLICY.items()
        }
    if normalized == "aggressive":
        return {
            regime: _scaled_policy(
                policy,
                target_multiplier=1.20,
                defensive_addition=-0.05,
                cap_addition=0.10,
                preset_name="Aggressive",
            )
            for regime, policy in DEFAULT_REGIME_POLICY.items()
        }
    raise ValueError("unknown policy preset. Supported: Conservative, Balanced default, Aggressive")
