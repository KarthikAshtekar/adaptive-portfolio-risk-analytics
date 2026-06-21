"""Lightweight tests for dashboard mode and manager-overlay contracts."""

from __future__ import annotations

from src.dashboard.modes import (
    DASHBOARD_MODES,
    DEFAULT_DASHBOARD_MODE,
    DEFAULT_MANAGER_ADAPTIVE_OVERLAY,
    DEFAULT_RESEARCH_OBJECTIVE,
    DEVELOPER_SECTIONS,
    DEVELOPER_VIEW,
    MANAGER_SECTIONS,
    MANAGER_VIEW,
    RESEARCH_SECTIONS,
    RESEARCH_VIEW,
    RULE_BASED_ROBUSTNESS_REFERENCE,
    adaptive_overlay_name,
    classify_recommended_use,
    objective_metric,
    research_objective_label,
)


def test_default_dashboard_mode_is_manager_view() -> None:
    assert DEFAULT_DASHBOARD_MODE == MANAGER_VIEW
    assert DASHBOARD_MODES == (MANAGER_VIEW, RESEARCH_VIEW, DEVELOPER_VIEW)


def test_mode_sections_preserve_research_and_debug_capabilities() -> None:
    assert "Experiment Sensitivity" in RESEARCH_SECTIONS
    assert "Phase 3A — CPCV Robustness Validation" in RESEARCH_SECTIONS
    assert "Phase 3B — Regime Detection" in RESEARCH_SECTIONS
    assert "Raw HMM Diagnostics" in DEVELOPER_SECTIONS
    assert "Raw RBI Documents" in DEVELOPER_SECTIONS
    assert "Raw CPCV Diagnostics" in DEVELOPER_SECTIONS
    assert "Net/Gross Reconciliation" in DEVELOPER_SECTIONS
    assert "Strategy Recommendation" in MANAGER_SECTIONS
    assert "RBI Macro-Sentiment Confirmation" in MANAGER_SECTIONS
    assert "NLP Data Status" in MANAGER_SECTIONS
    assert (
        "Phase 4A.7 — Real NLP Data Intake Workflow"
        in RESEARCH_SECTIONS
    )
    assert "Composite NLP Risk Index" in DEVELOPER_SECTIONS
    assert "Source Quality Components" in DEVELOPER_SECTIONS
    assert "Real NLP Corpus Intake Diagnostics" in DEVELOPER_SECTIONS


def test_manager_default_overlay_is_hmm_conservative() -> None:
    assert DEFAULT_MANAGER_ADAPTIVE_OVERLAY["regime_source"].startswith("HMM")
    assert DEFAULT_MANAGER_ADAPTIVE_OVERLAY["policy_preset"] == "Conservative"
    assert (
        DEFAULT_MANAGER_ADAPTIVE_OVERLAY["display_name"]
        == "Regime-Adaptive HMM Walk-Forward — Conservative"
    )
    assert adaptive_overlay_name(
        DEFAULT_MANAGER_ADAPTIVE_OVERLAY["regime_source"],
        DEFAULT_MANAGER_ADAPTIVE_OVERLAY["policy_preset"],
    ) == DEFAULT_MANAGER_ADAPTIVE_OVERLAY["display_name"]
    assert RULE_BASED_ROBUSTNESS_REFERENCE.endswith("Rule-Based — Conservative")


def test_research_objective_defaults_to_calmar() -> None:
    assert DEFAULT_RESEARCH_OBJECTIVE == "Net Calmar"
    assert objective_metric(DEFAULT_RESEARCH_OBJECTIVE) == "calmar"
    assert objective_metric(None) == "calmar"
    assert research_objective_label("calmar") == "Net Calmar"


def test_recommended_use_detects_risk_control_overlay() -> None:
    fixed = {
        "cagr": 0.15,
        "calmar": 0.79,
        "max_drawdown": -0.19,
        "final_value": 2_400_000,
    }
    adaptive = {
        "cagr": 0.10,
        "calmar": 1.20,
        "max_drawdown": -0.08,
        "final_value": 1_800_000,
    }

    assert classify_recommended_use(fixed, adaptive) == "Risk-Control Overlay"
