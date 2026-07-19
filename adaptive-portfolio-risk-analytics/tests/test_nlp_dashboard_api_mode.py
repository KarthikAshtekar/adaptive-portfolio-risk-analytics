"""Dashboard and selection guardrails for API-based NLP monitoring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.dashboard.app import build_api_nlp_monitoring_results
from src.selection import select_strategy_for_profile
from src.sentiment.providers import EarningsCallProvider, GDELTProvider


APP_SOURCE = (Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py").read_text(
    encoding="utf-8"
)


def _function_source(name: str, next_name: str) -> str:
    start = APP_SOURCE.index(f"def {name}(")
    end = APP_SOURCE.index(f"def {next_name}(", start)
    return APP_SOURCE[start:end]


def test_manager_view_is_compact_and_does_not_expose_raw_provider_data() -> None:
    manager = _function_source("render_manager_view", "render_developer_view")

    assert 'st.header("NLP Data Status")' in manager
    assert "NLP Data Status" in manager
    assert "Composite NLP Confirmation" in manager
    assert "Coverage Quality" in manager
    assert "Latest Text Date" in manager
    assert "Provider Mix" in manager
    assert "raw_provider_records" not in manager
    assert "normalized_records" not in manager
    assert "fallback_reason" not in manager


def test_research_and_developer_views_expose_nlp_diagnostics() -> None:
    assert "NLP Provider Selector" in APP_SOURCE
    assert "NLP Query Preset" in APP_SOURCE
    assert "Provider Configuration and API Diagnostics" in APP_SOURCE
    assert "Coverage and Freshness" in APP_SOURCE
    assert "Source Quality" in APP_SOURCE
    assert "Reaction Warning Rate" in APP_SOURCE
    assert "Decision-Lagged Composite NLP Risk Index" in APP_SOURCE
    assert "Raw API / Ex-Ante NLP Diagnostics" in APP_SOURCE
    assert "Possible reaction-data records" in APP_SOURCE
    assert "FinBERT and lexicon fallback metadata" in APP_SOURCE
    assert "Ex-ante validation and timestamp alignment checks" in APP_SOURCE
    assert "Phase 4A.13 — NLP Shadow Impact" in APP_SOURCE
    assert "Pain Index / Pain Ratio Comparison" in APP_SOURCE
    assert "NLP Shadow Strategy Comparison" in APP_SOURCE
    assert "Phase 4A.13 NLP Shadow Impact Audit" in APP_SOURCE
    assert "Overlay decision audit table" in APP_SOURCE


def test_dashboard_nlp_builder_runs_offline_with_fixture_providers(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    providers = [
        EarningsCallProvider(root / "data" / "sentiment" / "earnings_calls" / "manifest.csv"),
        GDELTProvider(
            enabled=True,
            fixture_path=(root / "data" / "sentiment" / "provider_fixtures" / "gdelt_sample.json"),
        ),
    ]
    index = pd.bdate_range("2024-01-01", "2026-06-19")
    regimes = pd.Series("Stress", index=index)

    payload = build_api_nlp_monitoring_results(
        providers=providers,
        start_date="2024-01-01",
        end_date="2026-06-19",
        market_index=index,
        regime_payload={"rule_based_decision_regimes": regimes},
        query_terms=["India inflation"],
        scoring_method="lexicon",
        decision_lag=1,
        output_dir=tmp_path,
    )

    assert payload["error"] is None
    assert payload["providers_with_data"] == []
    assert set(payload["providers_with_any_data"]) == {
        "earnings_calls",
        "gdelt",
    }
    assert payload["current"]["nlp_data_status"] == "Real Data Unavailable"
    assert payload["current"]["coverage_quality"] == "Insufficient"
    assert payload["ex_ante_validation"]["is_ex_ante_valid"].all()
    assert payload["reaction_data_warnings"].shape[0] == 1
    assert payload["composite_index"]["decision_lag"].eq(1).all()


def test_nlp_commentary_does_not_change_strategy_selection_or_scores() -> None:
    baseline = select_strategy_for_profile("Balanced")
    monitored = select_strategy_for_profile(
        "Balanced",
        nlp_risk_label="nlp_risk_off",
        nlp_confirmation_status="Confirms Quantitative Stress",
        nlp_coverage=0.75,
    )

    assert monitored.main_strategy == baseline.main_strategy
    assert monitored.overlay_strategy == baseline.overlay_strategy
    assert monitored.confidence_score == baseline.confidence_score
    pd.testing.assert_series_equal(
        monitored.candidate_scores["selection_score"],
        baseline.candidate_scores["selection_score"],
    )
    assert monitored.nlp_risk_label == "nlp_risk_off"
    assert "composite ex-ante NLP monitor" in monitored.explanation
