"""Source-level guardrails for the simplified Manager View."""

from __future__ import annotations

from pathlib import Path

from src.dashboard.modes import (
    MANAGER_HIDDEN_ADVANCED_LABELS,
    MANAGER_INPUT_LABELS,
    MANAGER_PROFILE_OBJECTIVES,
)


APP_SOURCE = (
    Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py"
).read_text(encoding="utf-8")


def _manager_sidebar_source() -> str:
    start = APP_SOURCE.index(
        'if dashboard_mode == MANAGER_VIEW:\n'
        '    with st.sidebar.expander("Portfolio Universe"'
    )
    end = APP_SOURCE.index(
        'else:\n    with st.sidebar.expander("Portfolio Scope"',
        start,
    )
    return APP_SOURCE[start:end]


def test_manager_sidebar_contains_only_decision_inputs() -> None:
    manager_source = _manager_sidebar_source()

    for label in MANAGER_INPUT_LABELS:
        assert label in manager_source
    for label in MANAGER_HIDDEN_ADVANCED_LABELS:
        assert f'"{label}"' not in manager_source


def test_manager_defaults_are_balanced_moderate_and_hmm_conservative() -> None:
    assert '"ui_manager_investor_profile": "Balanced"' in APP_SOURCE
    assert '"ui_manager_cost_assumption": "Moderate"' in APP_SOURCE
    assert 'DEFAULT_MANAGER_ADAPTIVE_OVERLAY["policy_preset"]' in APP_SOURCE


def test_manager_profile_maps_to_internal_objective() -> None:
    assert MANAGER_PROFILE_OBJECTIVES["Growth"] == "Net Final Value"
    assert MANAGER_PROFILE_OBJECTIVES["Balanced"] == "Net Calmar"
    assert MANAGER_PROFILE_OBJECTIVES["Capital Preservation"] == "Max Drawdown"


def test_research_and_developer_selection_diagnostics_remain_available() -> None:
    assert "Strategy Selection Diagnostics" in APP_SOURCE
    assert "Selection Gate Results" in APP_SOURCE
    assert "Selection Artifact Diagnostics and Scoring" in APP_SOURCE

