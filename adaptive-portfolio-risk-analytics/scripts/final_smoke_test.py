"""Lightweight v1.2.2 project-freeze smoke checks."""

from __future__ import annotations

import contextlib
import importlib
import io
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VERSION_LABEL = "v1.2.2 — Real NLP Data Intake Workflow"
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
        failures.append("README does not contain the v1.2.2 label")
    else:
        print("PASS README contains v1.2.2 label")

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
        "ex_ante_validation.csv",
        "collection_summary.json",
        "scored_records.csv",
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

    if failures:
        print("\nSmoke test failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nFinal smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
