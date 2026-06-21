"""Dashboard contracts for real NLP intake readiness."""

from __future__ import annotations

from pathlib import Path

from src.dashboard.app import NLP_INTAKE_INACTIVE_MESSAGE


APP_SOURCE = (
    Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py"
).read_text(encoding="utf-8")


def test_manager_marks_nlp_inactive_or_illustrative() -> None:
    assert NLP_INTAKE_INACTIVE_MESSAGE == (
        "NLP monitoring is inactive or illustrative until real text coverage "
        "is sufficient."
    )


def test_research_and_developer_expose_intake_status() -> None:
    assert "Real NLP Data Intake Status" in APP_SOURCE
    assert 'for corpus in ("rbi", "earnings", "news")' in APP_SOURCE
    assert 'st.write(f"{corpus.title()} intake rows")' in APP_SOURCE
    assert "docs/nlp_real_data_acquisition_guide.md" in APP_SOURCE
