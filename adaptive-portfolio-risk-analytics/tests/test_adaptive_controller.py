"""Tests for the Phase 3C regime-adaptive controller."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.adaptive import DEFAULT_REGIME_POLICY, RegimeAdaptiveController


def _regimes() -> pd.Series:
    return pd.Series(
        ["Calm", "Normal", "Stress", "Crisis"],
        index=pd.date_range("2024-01-01", periods=4, freq="B"),
        name="regime",
    )


def test_controller_selects_lagged_policy() -> None:
    controller = RegimeAdaptiveController(use_lagged_regimes=True)
    regimes = _regimes()

    policy = controller.get_policy(regimes.index[2], regimes)

    assert policy.regime == "Normal"
    assert controller.select_covariance_method(regimes.index[2], regimes) == "ledoit_wolf"


def test_unknown_and_missing_dates_fall_back_safely() -> None:
    controller = RegimeAdaptiveController(use_lagged_regimes=True)
    regimes = _regimes()

    before_history = controller.get_policy(pd.Timestamp("2023-12-01"), regimes)
    first_date = controller.get_policy(regimes.index[0], regimes)

    assert before_history.regime == "Unknown"
    assert first_date.regime == "Unknown"


def test_custom_default_policy_is_used_for_missing_regime() -> None:
    controller = RegimeAdaptiveController(
        default_policy=DEFAULT_REGIME_POLICY["Normal"],
        use_lagged_regimes=True,
    )

    policy = controller.get_policy(pd.Timestamp("2023-12-01"), _regimes())

    assert policy.regime == "Normal"


def test_exposure_calculation_handles_normal_realized_volatility() -> None:
    risky, defensive = RegimeAdaptiveController.calculate_risky_exposure(
        realized_volatility=0.20,
        target_volatility=0.10,
        cap=0.80,
        floor=0.20,
    )

    assert risky == pytest.approx(0.50)
    assert defensive == pytest.approx(0.50)


@pytest.mark.parametrize("realized_volatility", [0.0, np.nan, None])
def test_exposure_calculation_handles_zero_or_missing_volatility(
    realized_volatility,
) -> None:
    risky, defensive = RegimeAdaptiveController.calculate_risky_exposure(
        realized_volatility=realized_volatility,
        target_volatility=0.10,
        cap=0.80,
        floor=0.10,
    )

    assert risky == pytest.approx(0.80)
    assert defensive == pytest.approx(0.20)


def test_defensive_floor_and_risky_cap_are_respected() -> None:
    risky, defensive = RegimeAdaptiveController.calculate_risky_exposure(
        realized_volatility=0.01,
        target_volatility=0.20,
        cap=1.00,
        floor=0.40,
    )

    assert risky == pytest.approx(0.60)
    assert defensive == pytest.approx(0.40)
