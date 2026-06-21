"""Regime-to-policy selection and exposure control for Phase 3C."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.benchmarks import BenchmarkFactory

from .policies import (
    DEFAULT_REGIME_POLICY,
    RegimePolicy,
    get_policy_for_regime,
    validate_policy_map,
)


class RegimeAdaptiveController:
    """Select adaptive portfolio behavior from lagged market regimes."""

    def __init__(
        self,
        policy_map: Mapping[str, RegimePolicy] | None = None,
        default_policy: RegimePolicy | None = None,
        use_lagged_regimes: bool = True,
    ):
        self.policy_map = validate_policy_map(dict(policy_map or DEFAULT_REGIME_POLICY))
        if default_policy is not None and not isinstance(default_policy, RegimePolicy):
            raise TypeError("default_policy must be a RegimePolicy")
        self.default_policy = default_policy or self.policy_map["Unknown"]
        self.use_lagged_regimes = bool(use_lagged_regimes)

    def _regime_for_date(self, date, regime_series) -> str:
        if not isinstance(regime_series, pd.Series) or regime_series.empty:
            return "Unknown"

        regimes = regime_series.sort_index()
        timestamp = pd.Timestamp(date)
        available = regimes.loc[regimes.index <= timestamp]
        if available.empty:
            return "Unknown"
        position = len(available) - 1 - int(self.use_lagged_regimes)
        if position < 0:
            return "Unknown"
        value = available.iloc[position]
        return "Unknown" if pd.isna(value) else str(value)

    def get_policy(self, date, regime_series) -> RegimePolicy:
        """Return the effective policy at ``date`` using the configured lag."""
        regime = self._regime_for_date(date, regime_series)
        try:
            policy = get_policy_for_regime(regime, self.policy_map)
            return self.default_policy if policy.regime == "Unknown" else policy
        except (KeyError, ValueError):
            return self.default_policy

    def select_allocator(self, date, regime_series):
        """Instantiate the allocator selected by the effective policy."""
        policy = self.get_policy(date, regime_series)
        return BenchmarkFactory.get_allocator(
            policy.allocator,
            covariance_method=policy.covariance_method,
        )

    def select_covariance_method(self, date, regime_series) -> str:
        return self.get_policy(date, regime_series).covariance_method

    def select_target_volatility(self, date, regime_series) -> float:
        return float(self.get_policy(date, regime_series).target_volatility)

    def select_rebalance_params(self, date, regime_series) -> dict[str, object]:
        policy = self.get_policy(date, regime_series)
        return {
            "rebalance_mode": policy.rebalance_mode,
            "rebalance_threshold": float(policy.rebalance_threshold),
        }

    @staticmethod
    def calculate_risky_exposure(
        realized_volatility,
        target_volatility,
        cap,
        floor: float = 0.0,
    ) -> tuple[float, float]:
        """Return risky exposure and defensive weight under policy constraints."""
        cap = float(np.clip(cap, 0.0, 1.0))
        floor = float(np.clip(floor, 0.0, 1.0))
        target = float(target_volatility)
        try:
            realized = float(realized_volatility)
        except (TypeError, ValueError):
            realized = np.nan

        if not np.isfinite(realized) or realized <= 0.0:
            raw_exposure = cap
        else:
            raw_exposure = target / realized

        risky_exposure = float(np.clip(raw_exposure, 0.0, cap))
        defensive_weight = float(max(1.0 - risky_exposure, floor))
        defensive_weight = float(np.clip(defensive_weight, 0.0, 1.0))
        risky_exposure = float(1.0 - defensive_weight)
        return risky_exposure, defensive_weight

    def decision_row(
        self,
        *,
        date,
        regime_series,
        realized_volatility,
    ) -> dict[str, object]:
        """Return the selected controls and exposure for one decision date."""
        regime = self._regime_for_date(date, regime_series)
        policy = self.get_policy(date, regime_series)
        risky_exposure, defensive_weight = self.calculate_risky_exposure(
            realized_volatility,
            policy.target_volatility,
            policy.risky_exposure_cap,
            floor=policy.defensive_weight_floor,
        )
        try:
            realized_value = float(realized_volatility)
        except (TypeError, ValueError):
            realized_value = np.nan
        return {
            "date": pd.Timestamp(date),
            "regime": regime,
            "allocator": policy.allocator,
            "covariance_method": policy.covariance_method,
            "target_volatility": float(policy.target_volatility),
            "rebalance_mode": policy.rebalance_mode,
            "rebalance_threshold": float(policy.rebalance_threshold),
            "realized_volatility": realized_value if np.isfinite(realized_value) else np.nan,
            "risky_exposure": risky_exposure,
            "defensive_weight": defensive_weight,
            "policy_notes": policy.notes,
        }
