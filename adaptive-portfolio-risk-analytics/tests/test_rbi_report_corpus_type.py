"""Corpus provenance checks for Phase 4A.2 and Phase 4A.3 reports."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_report_is_explicitly_marked() -> None:
    summary = (
        REPO_ROOT / "outputs" / "reports" / "phase_4a2_rbi_macro_sentiment" / "summary.md"
    ).read_text(encoding="utf-8")

    assert "Corpus type: `synthetic_fixture`" in summary


def test_phase4a3_report_marks_current_fallback_mode() -> None:
    report_dir = REPO_ROOT / "outputs" / "reports" / "phase_4a3_real_rbi_macro_validation"
    summary = (report_dir / "summary.md").read_text(encoding="utf-8")
    report = (report_dir / "report.html").read_text(encoding="utf-8")

    assert "Corpus type: `synthetic_fixture`" in summary
    assert "Real RBI corpus unavailable" in summary
    assert "synthetic_fixture" in report
