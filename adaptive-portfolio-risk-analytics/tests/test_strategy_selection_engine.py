"""Tests for profile-aware, evidence-gated strategy selection."""

from __future__ import annotations

import pandas as pd

from src.selection.config import (
    EQUAL_WEIGHT,
    HERC,
    HMM_CONSERVATIVE,
    REJECTED_ROLE,
    RULE_CONSERVATIVE,
)
from src.selection.selector import load_selection_artifacts, select_strategy_for_profile


def test_balanced_profile_keeps_herc_core_and_hmm_overlay() -> None:
    recommendation = select_strategy_for_profile("Balanced")

    assert recommendation.main_strategy == HERC
    assert recommendation.overlay_strategy == HMM_CONSERVATIVE
    assert recommendation.confidence in {"High", "Moderate", "Low"}
    assert recommendation.evidence["phase3e_available"] is True
    assert "net return metrics" in recommendation.assumptions[0].lower()


def test_robustness_first_uses_rule_conservative_as_reference() -> None:
    recommendation = select_strategy_for_profile("Robustness First")

    assert recommendation.main_strategy == HERC
    assert recommendation.overlay_strategy == RULE_CONSERVATIVE
    assert recommendation.overlay_role == "Robustness Reference"


def test_unstable_hmm_is_rejected_and_falls_back_to_rule() -> None:
    recommendation = select_strategy_for_profile(
        "Capital Preservation",
        hmm_walk_forward_valid=False,
    )

    assert "HMM Unstable" in recommendation.scenario_categories
    assert recommendation.role_assignments[HMM_CONSERVATIVE] == REJECTED_ROLE
    assert recommendation.overlay_strategy == RULE_CONSERVATIVE
    assert any("full-sample HMM" in warning for warning in recommendation.warnings)


def test_high_cost_scenario_is_explicit() -> None:
    recommendation = select_strategy_for_profile(
        "Balanced",
        base_bps=50.0,
        slippage_bps=25.0,
    )

    assert "High Cost" in recommendation.scenario_categories
    assert any("High cost assumptions" in warning for warning in recommendation.warnings)


def test_post_p0_fallback_warns_without_crashing(tmp_path) -> None:
    report_dir = (
        tmp_path / "outputs" / "reports" / "post_p0_adaptive_validation"
    )
    report_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "strategy": EQUAL_WEIGHT,
                "strategy_type": "fixed",
                "return_basis": "net",
                "cagr": 0.10,
                "calmar": 0.4,
                "max_drawdown": -0.25,
                "final_value": 1_800_000,
                "total_turnover": 2,
                "total_transaction_cost": 4_000,
            },
            {
                "strategy": HERC,
                "strategy_type": "fixed",
                "return_basis": "net",
                "cagr": 0.14,
                "calmar": 0.8,
                "max_drawdown": -0.18,
                "final_value": 2_300_000,
                "total_turnover": 7,
                "total_transaction_cost": 18_000,
            },
            {
                "strategy": HMM_CONSERVATIVE,
                "strategy_type": "regime_adaptive",
                "regime_source": "hmm_walk_forward",
                "return_basis": "net",
                "cagr": 0.10,
                "calmar": 1.2,
                "max_drawdown": -0.08,
                "final_value": 1_850_000,
                "total_turnover": 9,
                "total_transaction_cost": 23_000,
                "defensive_source_used": "synthetic",
            },
            {
                "strategy": RULE_CONSERVATIVE,
                "strategy_type": "regime_adaptive",
                "regime_source": "rule_based_lagged",
                "return_basis": "net",
                "cagr": 0.11,
                "calmar": 1.1,
                "max_drawdown": -0.10,
                "final_value": 1_900_000,
                "total_turnover": 20,
                "total_transaction_cost": 45_000,
                "defensive_source_used": "synthetic",
            },
        ]
    ).to_csv(report_dir / "metrics_comparison.csv", index=False)

    artifacts = load_selection_artifacts(tmp_path)
    recommendation = select_strategy_for_profile("Balanced", artifacts=artifacts)

    assert artifacts["fallback_used"] is True
    assert recommendation.main_strategy == HERC
    assert any("post-P0 evidence" in warning for warning in recommendation.warnings)
