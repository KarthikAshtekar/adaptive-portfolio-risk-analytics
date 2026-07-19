"""NLP evidence coverage and dashboard presentation tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.sentiment import calculate_nlp_coverage


APP_SOURCE = (Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py").read_text(
    encoding="utf-8"
)


def test_sparse_real_data_is_marked_insufficient() -> None:
    records = pd.DataFrame(
        {
            "publication_time": ["2026-06-01T09:00:00Z"],
            "provider": ["rbi"],
        }
    )
    index = pd.bdate_range("2026-01-01", "2026-06-21")
    composite = pd.DataFrame(
        {"decision_composite_nlp_label": "insufficient_nlp_data"},
        index=index,
    )
    diagnostics = pd.DataFrame(
        {"provider": ["rbi", "earnings_calls"], "configured_enabled": [True, True]}
    )

    coverage = calculate_nlp_coverage(
        records,
        composite_index=composite,
        provider_diagnostics=diagnostics,
        start_date="2026-01-01",
        end_date="2026-06-21",
        min_records=50,
        min_distinct_dates=20,
        min_coverage_ratio=0.20,
    )

    assert coverage["record_count"] == 1
    assert coverage["provider_coverage"] == 0.5
    assert coverage["coverage_quality"] == "insufficient"


def test_dashboard_keeps_manager_compact_and_diagnostics_available() -> None:
    manager_start = APP_SOURCE.index("def render_manager_view(")
    developer_start = APP_SOURCE.index("def render_developer_view(", manager_start)
    manager = APP_SOURCE[manager_start:developer_start]

    assert 'st.header("NLP Data Status")' in manager
    assert "Composite NLP Confirmation" in manager
    assert "Coverage Quality" in manager
    assert "Latest Text Date" in manager
    assert "Provider Mix" in manager
    assert ("NLP signal is monitoring-only due to insufficient real-data coverage.") in APP_SOURCE
    assert "source_quality_score" not in manager
    assert "Provider Configuration and API Diagnostics" in APP_SOURCE
    assert "Source-quality components and data provenance" in APP_SOURCE
    assert "Pre-Stress Warning Diagnostics" in APP_SOURCE
