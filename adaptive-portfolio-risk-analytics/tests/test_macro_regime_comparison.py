"""Tests for RBI macro and quantitative regime comparison."""

from __future__ import annotations

import pandas as pd

from src.selection import select_strategy_for_profile
from src.sentiment import compare_macro_to_regimes, macro_agrees_with_regime


def test_comparison_reports_agreement_leads_disagreements_and_coverage() -> None:
    index = pd.bdate_range("2024-01-01", periods=6)
    macro = pd.DataFrame(
        {
            "decision_macro_label": [
                "risk_on_macro",
                "risk_off_macro",
                "risk_off_macro",
                "neutral_macro",
                "risk_on_macro",
                "insufficient_macro_data",
            ],
            "decision_sentence_count": [2, 2, 2, 2, 2, 0],
        },
        index=index,
    )
    rule = pd.Series(
        ["Calm", "Normal", "Stress", "Normal", "Crisis", "Normal"],
        index=index,
    )
    hmm = pd.Series(
        ["Risk-On", "Normal", "Risk-Off", "Normal", "Risk-Off", "Normal"],
        index=index,
    )

    comparison = compare_macro_to_regimes(
        macro,
        rule,
        hmm,
        max_shift=2,
        lead_window=2,
    )

    assert comparison["agreement_with_rule_based"] == 0.6
    assert comparison["agreement_with_hmm_walk_forward"] == 0.6
    assert comparison["coverage_ratio"] == 5 / 6
    assert not comparison["lead_lag_diagnostics"].empty
    assert not comparison["macro_risk_off_before_stress_dates"].empty
    assert not comparison["dates_of_major_disagreement"].empty


def test_macro_commentary_does_not_change_selection_or_confidence() -> None:
    baseline = select_strategy_for_profile("Balanced")
    macro_confirmed = select_strategy_for_profile(
        "Balanced",
        macro_sentiment_label="risk_off_macro",
        macro_sentiment_confirmation="Confirmed Risk-Off",
        macro_sentiment_coverage=12,
    )

    assert macro_confirmed.main_strategy == baseline.main_strategy
    assert macro_confirmed.overlay_strategy == baseline.overlay_strategy
    assert macro_confirmed.confidence_score == baseline.confidence_score
    pd.testing.assert_series_equal(
        macro_confirmed.candidate_scores["selection_score"],
        baseline.candidate_scores["selection_score"],
    )


def test_macro_mapping_matches_prompt_contract() -> None:
    assert macro_agrees_with_regime("risk_on_macro", "Calm") is True
    assert macro_agrees_with_regime("risk_on_macro", "Normal") is True
    assert macro_agrees_with_regime("neutral_macro", "Normal") is True
    assert macro_agrees_with_regime("risk_off_macro", "Stress") is True
    assert macro_agrees_with_regime("risk_off_macro", "Crisis") is True
