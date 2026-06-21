"""Tests for Phase 3C adaptive regime policies."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.adaptive import (
    DEFAULT_REGIME_POLICY,
    get_policy_for_regime,
    get_policy_preset,
    policy_map_to_dataframe,
    validate_policy_map,
)


def test_default_policy_map_contains_required_regimes() -> None:
    assert {"Calm", "Normal", "Stress", "Crisis", "Unknown"} == set(DEFAULT_REGIME_POLICY)


def test_two_state_hmm_labels_map_safely() -> None:
    risk_on = get_policy_for_regime("Risk-On")
    risk_off = get_policy_for_regime("Risk-Off")

    assert risk_on.regime == "Calm"
    assert risk_off.regime == "Stress"


def test_policy_validation_rejects_invalid_bounds() -> None:
    invalid = dict(DEFAULT_REGIME_POLICY)
    invalid["Calm"] = replace(
        invalid["Calm"],
        target_volatility=0.0,
    )

    with pytest.raises(ValueError, match="target_volatility"):
        validate_policy_map(invalid)

    invalid = dict(DEFAULT_REGIME_POLICY)
    invalid["Stress"] = replace(
        invalid["Stress"],
        risky_exposure_cap=1.2,
    )
    with pytest.raises(ValueError, match="risky_exposure_cap"):
        validate_policy_map(invalid)

    invalid = dict(DEFAULT_REGIME_POLICY)
    invalid["Normal"] = replace(
        invalid["Normal"],
        target_volatility=np.nan,
    )
    with pytest.raises(ValueError, match="target_volatility"):
        validate_policy_map(invalid)


def test_policy_table_is_generated_correctly() -> None:
    table = policy_map_to_dataframe(DEFAULT_REGIME_POLICY)

    assert len(table) == 5
    assert {
        "regime",
        "allocator",
        "covariance_method",
        "target_volatility",
        "defensive_weight_floor",
        "risky_exposure_cap",
    }.issubset(table.columns)


def test_policy_presets_adjust_risk_controls() -> None:
    conservative = get_policy_preset("Conservative")
    balanced = get_policy_preset("Balanced default")
    aggressive = get_policy_preset("Aggressive")

    assert (
        conservative["Normal"].target_volatility
        < balanced["Normal"].target_volatility
        < aggressive["Normal"].target_volatility
    )
    assert (
        conservative["Stress"].defensive_weight_floor
        > balanced["Stress"].defensive_weight_floor
        > aggressive["Stress"].defensive_weight_floor
    )
