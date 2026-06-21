"""Dashboard guardrails for real-RBI corpus mode and fallback behavior."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.dashboard.app import build_rbi_macro_sentiment_results


APP_SOURCE = (
    Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py"
).read_text(encoding="utf-8")


def test_dashboard_falls_back_when_real_manifest_is_missing(
    tmp_path: Path,
) -> None:
    index = pd.bdate_range("2022-01-03", "2026-06-19")
    regimes = pd.Series("Normal", index=index)

    payload = build_rbi_macro_sentiment_results(
        manifest_path=tmp_path / "missing.csv",
        market_index=index,
        regime_payload={
            "rule_based_decision_regimes": regimes,
            "method": "Rule-based",
        },
        corpus_mode="real_rbi",
    )

    assert payload["corpus_type"] == "synthetic_fixture"
    assert payload["real_corpus_available"] is False
    assert payload["current"]["illustrative_only"] is True
    assert "synthetic fixture" in payload["fallback_reason"].lower()


def test_manager_and_research_expose_real_corpus_contract() -> None:
    assert '"RBI Corpus"' in APP_SOURCE
    assert "Macro confirmation is illustrative only" in APP_SOURCE
    assert '"RBI Corpus Mode"' in APP_SOURCE
    assert '"Minimum Coverage Threshold"' in APP_SOURCE
    assert "Corpus Diagnostics" in APP_SOURCE
    assert "Invalid real-corpus documents" in APP_SOURCE
