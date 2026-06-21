"""Guardrails for Phase 4A dashboard and selection integration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.selection import select_strategy_for_profile


APP_SOURCE = (
    Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py"
).read_text(encoding="utf-8")


def _function_source(name: str, next_name: str) -> str:
    start = APP_SOURCE.index(f"def {name}(")
    end = APP_SOURCE.index(f"def {next_name}(", start)
    return APP_SOURCE[start:end]


def test_sentiment_commentary_does_not_change_selected_strategy_or_scores() -> None:
    baseline = select_strategy_for_profile("Balanced")
    confirmed = select_strategy_for_profile(
        "Balanced",
        sentiment_confirmation_status="Confirmed Risk-Off",
        sentiment_label="risk_off",
        sentiment_coverage=8,
    )

    assert confirmed.main_strategy == baseline.main_strategy
    assert confirmed.overlay_strategy == baseline.overlay_strategy
    assert confirmed.confidence_score == baseline.confidence_score
    pd.testing.assert_series_equal(
        confirmed.candidate_scores["selection_score"],
        baseline.candidate_scores["selection_score"],
    )
    assert confirmed.sentiment_confirmation_status == "Confirmed Risk-Off"
    assert "Sentiment confirms" in confirmed.explanation


def test_manager_view_has_compact_card_without_raw_headlines() -> None:
    manager_source = _function_source("render_manager_view", "render_developer_view")

    assert 'st.header("Sentiment Confirmation")' in manager_source
    assert "Current Quant Regime" in manager_source
    assert "Article Coverage" in manager_source
    assert "raw_records" not in manager_source
    assert "scored_records" not in manager_source
    assert 'st.header("RBI Macro-Sentiment Confirmation")' in manager_source
    assert "Macro Label" in manager_source
    assert "Coverage status" in manager_source
    assert "documents" in manager_source
    assert '"documents"' not in manager_source
    assert '"scored_sentences"' not in manager_source


def test_research_and_developer_views_expose_sentiment_diagnostics() -> None:
    assert "Phase 4A — Sentiment Regime Confirmation" in APP_SOURCE
    assert "Major disagreement dates" in APP_SOURCE
    assert "Raw Sentiment and Alignment Diagnostics" in APP_SOURCE
    assert "Timestamp and look-ahead checks" in APP_SOURCE
    assert "Phase 4A.2 — RBI Macro-Sentiment Confirmation" in APP_SOURCE
    assert "plot_macro_stance_shares" in APP_SOURCE
    assert "plot_macro_uncertainty_share" in APP_SOURCE
    assert "RBI Macro vs Rule-Based Decision Regime" in APP_SOURCE
    assert "RBI Macro vs HMM Walk-Forward Decision Regime" in APP_SOURCE
    assert "Macro risk-off before stress dates" in APP_SOURCE
    assert "Raw RBI Macro-Sentiment Diagnostics" in APP_SOURCE
    assert "Transformer fallback diagnostics" in APP_SOURCE
    assert "Timestamp, look-ahead, and construction diagnostics" in APP_SOURCE
