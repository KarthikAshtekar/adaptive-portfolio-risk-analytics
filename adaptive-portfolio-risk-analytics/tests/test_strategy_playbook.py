"""Tests for the manager-facing scenario playbook."""

from __future__ import annotations

from src.selection.config import HERC, HMM_CONSERVATIVE, PROFILE_NAMES, RULE_CONSERVATIVE
from src.selection.playbook import build_strategy_playbook


def test_playbook_covers_every_investor_profile() -> None:
    playbook = build_strategy_playbook()
    profile_rows = playbook.loc[playbook["investor_profile"] != "All"]

    assert set(profile_rows["investor_profile"]) == set(PROFILE_NAMES)
    assert set(profile_rows["core_strategy"]) == {HERC}


def test_playbook_contains_operational_fallbacks() -> None:
    playbook = build_strategy_playbook().set_index("scenario")

    assert playbook.loc["HMM Unstable", "overlay_or_reference"] == RULE_CONSERVATIVE
    assert playbook.loc["Stress / Crisis", "overlay_or_reference"] == HMM_CONSERVATIVE
    assert playbook.loc["Insufficient Data", "overlay_usage"] == "none"
