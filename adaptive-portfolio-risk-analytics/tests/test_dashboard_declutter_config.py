"""Source-level guardrails for dashboard decluttering decisions."""

from pathlib import Path

from src.dashboard.modes import DEFAULT_MANAGER_ADAPTIVE_OVERLAY


APP_SOURCE = (
    Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py"
).read_text(encoding="utf-8")


def test_dashboard_has_one_global_research_objective_widget() -> None:
    assert APP_SOURCE.count('"Research Objective"') == 1
    assert '"Takeaway Objective"' not in APP_SOURCE
    assert '"Sensitivity Objective"' not in APP_SOURCE


def test_manager_overlay_and_rule_based_caveat_are_present() -> None:
    assert DEFAULT_MANAGER_ADAPTIVE_OVERLAY["display_name"] in APP_SOURCE
    assert "CPCV-favored robustness reference" in APP_SOURCE


def test_large_raw_tables_use_download_controls() -> None:
    for label in (
        "Download Full Sensitivity Table",
        "Download Full CPCV Fold Table",
        "Download Full Adaptive Diagnostics",
        "Download Full HMM Probability Table",
        "Download Full Weight History",
    ):
        assert label in APP_SOURCE
