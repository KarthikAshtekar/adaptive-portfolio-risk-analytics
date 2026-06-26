"""Lightweight v1.3.0 project-release smoke checks."""

from __future__ import annotations

import contextlib
import importlib
import io
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VERSION_LABEL = "v1.3.0 — Final Integrated Portfolio Risk Analytics Release"
REQUIRED_MODULES = (
    "src.analytics",
    "src.backtesting",
    "src.adaptive",
    "src.regime",
    "src.validation",
    "src.selection",
    "src.sentiment",
)
REQUIRED_REPORTS = (
    "INDEX.md",
    "final_project_summary.md",
    "architecture_summary.md",
    "dashboard_user_guide.md",
    "methodology_report.md",
    "final_results_summary.md",
    "presentation_outline.md",
    "viva_questions_and_answers.md",
    "resume_bullets.md",
    "final_validation_checklist.md",
)
REQUIRED_V1_3_0_RELEASE_FILES = (
    "executive_summary.md",
    "final_report.md",
    "final_report.html",
    "technical_methodology.md",
    "portfolio_results.md",
    "pain_ratio_analysis.md",
    "nlp_shadow_impact.md",
    "risk_analytics_summary.md",
    "strategy_selection_summary.md",
    "dashboard_guide.md",
    "reproducibility_commands.md",
    "limitations.md",
    "viva_questions_and_answers.md",
    "final_metrics_table.csv",
    "strategy_ranking_table.csv",
    "pain_ratio_comparison.csv",
    "nlp_shadow_impact_table.csv",
    "evidence_matrix.csv",
)
REQUIRED_TEAM_HANDOFF_FILES = (
    "README_FOR_TEAM.md",
    "README_TECHNICAL_APPENDIX.md",
    "REPORT_WRITING_GUIDE.md",
    "TABLES_FIGURES_RESULTS_INSIGHTS.md",
    "artifact_index.csv",
    "ready_to_use_tables/table_manifest.csv",
    "ready_to_use_figures/figure_manifest.csv",
)


def main() -> int:
    failures: list[str] = []

    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
            print(f"PASS import {module_name}")
        except Exception as exc:  # pragma: no cover - smoke-test diagnostics
            failures.append(f"import {module_name}: {exc}")

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            importlib.import_module("src.dashboard.app")
        print("PASS import src.dashboard.app")
    except Exception as exc:  # pragma: no cover - smoke-test diagnostics
        failures.append(f"import src.dashboard.app: {exc}")

    selection = importlib.import_module("src.selection")
    if not callable(getattr(selection, "select_strategy_for_profile", None)):
        failures.append("selection engine entry point is not callable")
    else:
        print("PASS selection engine entry point")

    pack_dir = REPO_ROOT / "outputs" / "final_project_pack"
    for filename in REQUIRED_REPORTS:
        path = pack_dir / filename
        if not path.is_file():
            failures.append(f"missing report: {path.relative_to(REPO_ROOT)}")
    if not any(item.startswith("missing report:") for item in failures):
        print(f"PASS {len(REQUIRED_REPORTS)} final-pack reports exist")

    readme_path = REPO_ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8") if readme_path.is_file() else ""
    if VERSION_LABEL not in readme:
        failures.append("README does not contain the v1.3.0 label")
    else:
        print("PASS README contains v1.3.0 label")

    phase4a_dir = (
        REPO_ROOT
        / "outputs"
        / "reports"
        / "phase_4a_sentiment_confirmation"
    )
    for filename in (
        "report.html",
        "summary.md",
        "sentiment_regime_comparison.csv",
        "sentiment_signal.csv",
        "disagreement_dates.csv",
        "source_notes.md",
    ):
        if not (phase4a_dir / filename).is_file():
            failures.append(f"missing Phase 4A report artifact: {filename}")
    if not any(
        item.startswith("missing Phase 4A report artifact:")
        for item in failures
    ):
        print("PASS Phase 4A report artifacts exist")

    phase4a2_dir = (
        REPO_ROOT
        / "outputs"
        / "reports"
        / "phase_4a2_rbi_macro_sentiment"
    )
    for filename in (
        "report.html",
        "summary.md",
        "rbi_documents.csv",
        "rbi_sentence_scores.csv",
        "macro_stance_index.csv",
        "macro_regime_comparison.csv",
        "disagreement_dates.csv",
        "source_notes.md",
    ):
        if not (phase4a2_dir / filename).is_file():
            failures.append(f"missing Phase 4A.2 report artifact: {filename}")
    if not any(
        item.startswith("missing Phase 4A.2 report artifact:")
        for item in failures
    ):
        print("PASS Phase 4A.2 report artifacts exist")

    phase4a3_dir = (
        REPO_ROOT
        / "outputs"
        / "reports"
        / "phase_4a3_real_rbi_macro_validation"
    )
    for filename in (
        "report.html",
        "summary.md",
        "rbi_documents.csv",
        "rbi_sentence_scores.csv",
        "macro_stance_index.csv",
        "macro_regime_comparison.csv",
        "disagreement_dates.csv",
        "coverage_diagnostics.csv",
        "corpus_diagnostics.csv",
        "source_notes.md",
    ):
        if not (phase4a3_dir / filename).is_file():
            failures.append(f"missing Phase 4A.3 report artifact: {filename}")
    if not any(
        item.startswith("missing Phase 4A.3 report artifact:")
        for item in failures
    ):
        print("PASS Phase 4A.3 report artifacts exist")

    phase4a5_dir = (
        REPO_ROOT
        / "outputs"
        / "reports"
        / "phase_4a5_api_sentiment_ingestion"
    )
    for filename in (
        "report.html",
        "summary.md",
        "provider_diagnostics.csv",
        "normalized_sentiment_records.csv",
        "ex_ante_validation.csv",
        "finbert_scores.csv",
        "composite_nlp_risk_index.csv",
        "nlp_regime_comparison.csv",
        "reaction_data_warnings.csv",
        "source_notes.md",
    ):
        if not (phase4a5_dir / filename).is_file():
            failures.append(f"missing Phase 4A.5 report artifact: {filename}")
    if not any(
        item.startswith("missing Phase 4A.5 report artifact:")
        for item in failures
    ):
        print("PASS Phase 4A.5 report artifacts exist")

    phase4a6_dir = (
        REPO_ROOT
        / "outputs"
        / "reports"
        / "phase_4a6_real_nlp_validation"
    )
    for filename in (
        "report.html",
        "summary.md",
        "raw_provider_records.jsonl",
        "normalized_sentiment_records.csv",
        "deduped_sentiment_records.csv",
        "provider_diagnostics.csv",
        "provider_query_diagnostics.csv",
        "gdelt_query_diagnostics.csv",
        "ex_ante_validation.csv",
        "collection_summary.json",
        "scored_records.csv",
        "daily_nlp_signal.csv",
        "signal_construction_diagnostics.csv",
        "composite_nlp_risk_index.csv",
        "nlp_regime_comparison.csv",
        "coverage_diagnostics.csv",
        "reaction_data_warnings.csv",
        "source_quality.csv",
        "source_notes.md",
    ):
        if not (phase4a6_dir / filename).is_file():
            failures.append(f"missing Phase 4A.6 report artifact: {filename}")
    if not any(
        item.startswith("missing Phase 4A.6 report artifact:")
        for item in failures
    ):
        print("PASS Phase 4A.6 report artifacts exist")

    phase4a8_dir = (
        REPO_ROOT
        / "outputs"
        / "reports"
        / "phase_4a8_multisource_nlp_monitoring"
    )
    if phase4a8_dir.is_dir():
        for filename in (
            "report.html",
            "summary.md",
            "rbi_corpus_status.csv",
            "rbi_sentence_scores.csv",
            "rbi_macro_index.csv",
            "scored_news_records.csv",
            "daily_nlp_signal.csv",
            "source_mix_diagnostics.csv",
            "multi_source_nlp_comparison.csv",
            "source_notes.md",
        ):
            if not (phase4a8_dir / filename).is_file():
                failures.append(f"missing Phase 4A.8 report artifact: {filename}")
        if not any(
            item.startswith("missing Phase 4A.8 report artifact:")
            for item in failures
        ):
            print("PASS Phase 4A.8 report artifacts exist")

    intake_dir = (
        REPO_ROOT
        / "outputs"
        / "reports"
        / "nlp_corpus_intake_validation"
    )
    for filename in (
        "intake_status.csv",
        "rbi_status.csv",
        "earnings_status.csv",
        "news_status.csv",
        "summary.md",
    ):
        if not (intake_dir / filename).is_file():
            failures.append(f"missing NLP intake artifact: {filename}")
    intake_paths = (
        "docs/nlp_real_data_acquisition_guide.md",
        "scripts/bootstrap_rbi_real_corpus.py",
        "scripts/import_rbi_text_document.py",
        "scripts/check_rbi_corpus_status.py",
        "scripts/fetch_rbi_documents.py",
        "data/sentiment/rbi_real/manifest_template.csv",
        "data/sentiment/rbi_real/intake_notes.md",
        "data/sentiment/earnings_calls/manifest_template.csv",
        "data/sentiment/earnings_calls/intake_notes.md",
        "data/sentiment/news_real/manifest_template.csv",
        "data/sentiment/news_real/intake_notes.md",
    )
    for relative in intake_paths:
        if not (REPO_ROOT / relative).is_file():
            failures.append(f"missing NLP intake file: {relative}")
    if not any(
        item.startswith("missing NLP intake") for item in failures
    ):
        print("PASS Phase 4A.7 intake workflow artifacts exist")

    fetcher_dir = (
        REPO_ROOT
        / "outputs"
        / "reports"
        / "rbi_official_fetcher"
    )
    if fetcher_dir.is_dir():
        for filename in (
            "fetch_summary.md",
            "fetch_diagnostics.csv",
            "manual_fallback_required.csv",
            "downloaded_documents.csv",
        ):
            if not (fetcher_dir / filename).is_file():
                failures.append(f"missing RBI fetcher artifact: {filename}")
        if not any(
            item.startswith("missing RBI fetcher artifact:")
            for item in failures
        ):
            print("PASS Phase 4A.9 RBI fetcher artifacts exist")

    phase4a12_dir = (
        REPO_ROOT
        / "outputs"
        / "reports"
        / "phase_4a12_nlp_monitoring_final_pack"
    )
    for filename in (
        "report.html",
        "summary.md",
        "technical_methodology.md",
        "data_sources.md",
        "validation_results.md",
        "limitations.md",
        "reproducibility_commands.md",
        "dashboard_guide.md",
        "source_mix_summary.csv",
        "rbi_corpus_summary.csv",
        "news_signal_summary.csv",
        "daily_nlp_signal_snapshot.csv",
    ):
        if not (phase4a12_dir / filename).is_file():
            failures.append(f"missing Phase 4A.12 final-pack artifact: {filename}")
    if not any(
        item.startswith("missing Phase 4A.12 final-pack artifact:")
        for item in failures
    ):
        print("PASS Phase 4A.12 NLP monitoring final-pack artifacts exist")

    phase4a13_dir = (
        REPO_ROOT
        / "outputs"
        / "reports"
        / "phase_4a13_nlp_shadow_impact"
    )
    if phase4a13_dir.is_dir():
        for filename in (
            "report.html",
            "summary.md",
            "strategy_metrics.csv",
            "pain_ratio_comparison.csv",
            "drawdown_comparison.csv",
            "overlay_decisions.csv",
            "nlp_signal_alignment.csv",
            "lookahead_diagnostics.csv",
            "limitations.md",
        ):
            if not (phase4a13_dir / filename).is_file():
                failures.append(f"missing Phase 4A.13 artifact: {filename}")
        if not any(
            item.startswith("missing Phase 4A.13 artifact:")
            for item in failures
        ):
            print("PASS Phase 4A.13 NLP shadow-impact artifacts exist")

    v1_3_0_dir = (
        REPO_ROOT
        / "outputs"
        / "reports"
        / "v1_3_0_final_integrated_release"
    )
    for filename in REQUIRED_V1_3_0_RELEASE_FILES:
        if not (v1_3_0_dir / filename).is_file():
            failures.append(f"missing v1.3.0 final release artifact: {filename}")
    if not any(
        item.startswith("missing v1.3.0 final release artifact:")
        for item in failures
    ):
        print("PASS v1.3.0 final integrated release-pack artifacts exist")

    team_handoff_dir = (
        REPO_ROOT
        / "outputs"
        / "reports"
        / "team_report_handoff_pack"
    )
    for filename in REQUIRED_TEAM_HANDOFF_FILES:
        if not (team_handoff_dir / filename).is_file():
            failures.append(f"missing team handoff artifact: {filename}")
    if not any(
        item.startswith("missing team handoff artifact:")
        for item in failures
    ):
        print("PASS team report handoff artifacts exist")

    if failures:
        print("\nSmoke test failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nFinal smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
