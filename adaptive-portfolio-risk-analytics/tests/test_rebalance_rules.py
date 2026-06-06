"""Tests for rebalance rules."""

from __future__ import annotations

import numpy as np

from src.backtesting import normalize_rebalance_frequency, should_rebalance_threshold


def test_threshold_rule_triggers_when_drift_exceeds_threshold() -> None:
    current = np.array([0.60, 0.40])
    target = np.array([0.50, 0.50])

    assert should_rebalance_threshold(current, target, threshold=0.05) is True


def test_threshold_rule_does_not_trigger_below_threshold() -> None:
    current = np.array([0.52, 0.48])
    target = np.array([0.50, 0.50])

    assert should_rebalance_threshold(current, target, threshold=0.05) is False


def test_frequency_normalization_maps_m_to_me() -> None:
    assert normalize_rebalance_frequency("M") == "ME"


def test_frequency_normalization_keeps_me() -> None:
    assert normalize_rebalance_frequency("ME") == "ME"
