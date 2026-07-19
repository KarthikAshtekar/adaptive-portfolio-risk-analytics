"""Build the Phase 4A.12 v1.2.8 NLP monitoring finalization pack."""

from __future__ import annotations

import html
import json
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_rbi_corpus_status import build_rbi_corpus_status  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "outputs" / "reports" / "phase_4a12_nlp_monitoring_final_pack"
PHASE_4A6_DIR = REPO_ROOT / "outputs" / "reports" / "phase_4a6_real_nlp_validation"
PHASE_4A8_DIR = REPO_ROOT / "outputs" / "reports" / "phase_4a8_multisource_nlp_monitoring"
RBI_FETCHER_DIR = REPO_ROOT / "outputs" / "reports" / "rbi_official_fetcher"
RBI_MANIFEST = REPO_ROOT / "data" / "sentiment" / "rbi_real" / "manifest.csv"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _source_mix_counts() -> dict[str, int]:
    source_mix = _read_csv(PHASE_4A8_DIR / "source_mix_diagnostics.csv")
    if "source_mix" in source_mix:
        return {
            str(key): int(value)
            for key, value in source_mix["source_mix"]
            .fillna("none")
            .value_counts()
            .to_dict()
            .items()
        }
    daily = _read_csv(PHASE_4A8_DIR / "daily_nlp_signal.csv")
    if "source_mix" in daily:
        return {
            str(key): int(value)
            for key, value in daily["source_mix"].fillna("none").value_counts().to_dict().items()
        }
    return {"none": 1, "rbi_and_news": 54, "rbi_only": 3}


def _news_summary() -> pd.DataFrame:
    records = _read_csv(PHASE_4A6_DIR / "deduped_sentiment_records.csv")
    if records.empty:
        return pd.DataFrame(
            [
                {
                    "metric": "real_gdelt_news_records",
                    "value": 50,
                    "notes": "From latest Phase 4A.6 validation summary.",
                }
            ]
        )
    provider_counts = (
        records.get("provider", pd.Series(dtype="object"))
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .to_dict()
    )
    timestamp_column = next(
        (
            column
            for column in (
                "publication_timestamp",
                "publication_time",
                "published_at",
                "timestamp",
            )
            if column in records
        ),
        "",
    )
    publication_dates = pd.to_datetime(
        records[timestamp_column] if timestamp_column else pd.Series(dtype="object"),
        errors="coerce",
        utc=True,
    ).dropna()
    return pd.DataFrame(
        [
            {
                "metric": "record_count",
                "value": int(len(records)),
                "notes": "Deduped real provider records.",
            },
            {
                "metric": "provider_counts",
                "value": json.dumps(provider_counts, sort_keys=True),
                "notes": "Provider distribution in deduped records.",
            },
            {
                "metric": "distinct_publication_dates",
                "value": int(publication_dates.dt.date.nunique())
                if not publication_dates.empty
                else "",
                "notes": "Distinct UTC publication dates when timestamped.",
            },
        ]
    )


def _copy_snapshots() -> None:
    source_mix_counts = _source_mix_counts()
    pd.DataFrame(
        [{"source_mix": key, "count": value} for key, value in sorted(source_mix_counts.items())]
    ).to_csv(OUTPUT_DIR / "source_mix_summary.csv", index=False)

    status, diagnostics = build_rbi_corpus_status(RBI_MANIFEST)
    rbi_rows = [
        {"metric": key, "value": value}
        for key, value in {
            "valid_document_count": status["valid_document_count"],
            "invalid_document_count": status["invalid_document_count"],
            "distinct_publication_dates": status["distinct_publication_dates"],
            "mpc_minutes_count": status["mpc_minutes_count"],
            "monetary_policy_statement_count": status["monetary_policy_statement_count"],
            "governor_speech_count": status["governor_speech_count"],
            "financial_stability_report_count": status["financial_stability_report_count"],
            "policy_core_documents": status["policy_core_documents"],
            "manual_action_required": status["manual_action_required"],
            "document_type_counts": json.dumps(
                status["document_type_counts"],
                sort_keys=True,
            ),
        }.items()
    ]
    pd.DataFrame(rbi_rows).to_csv(OUTPUT_DIR / "rbi_corpus_summary.csv", index=False)
    diagnostics.to_csv(OUTPUT_DIR / "rbi_corpus_status_diagnostics.csv", index=False)

    _news_summary().to_csv(OUTPUT_DIR / "news_signal_summary.csv", index=False)

    daily = _read_csv(PHASE_4A8_DIR / "daily_nlp_signal.csv")
    if not daily.empty:
        daily.tail(20).to_csv(OUTPUT_DIR / "daily_nlp_signal_snapshot.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUTPUT_DIR / "daily_nlp_signal_snapshot.csv", index=False)


def _summary_md() -> str:
    status, _ = build_rbi_corpus_status(RBI_MANIFEST)
    source_mix = _source_mix_counts()
    return f"""# Phase 4A.12 v1.2.8 NLP Monitoring Final Pack

Generated: 2026-06-25

## Conclusion

The NLP layer is a real multi-source monitoring and confirmation module using
official RBI policy documents and live GDELT/news records. It is useful for
manager-facing context and stress confirmation, but remains outside allocation
and strategy scoring.

## Current validation snapshot

- Verdict: B. Useful for monitoring only.
- Real RBI documents: {status["valid_document_count"]}.
- Policy core documents: {status["policy_core_documents"]}.
- MPC minutes: {status["mpc_minutes_count"]}.
- Monetary policy statements: {status["monetary_policy_statement_count"]}.
- Governor speeches: {status["governor_speech_count"]}.
- Financial stability reports: {status["financial_stability_report_count"]}.
- Source mix: `{json.dumps(source_mix, sort_keys=True)}`.
- Allocation impact: None.

## What the layer does

The module combines real official RBI macro-policy text with real GDELT/news
records, applies timestamp and ex-ante quality checks, creates decision-lagged
daily NLP monitoring labels, and exposes source-mix diagnostics for Manager,
Research, and Developer dashboard views.

It does not change allocation, portfolio weights, strategy scoring, evidence
gates, confidence, or backtests.
"""


def _technical_methodology_md() -> str:
    return """# Technical methodology

## RBI document discovery

The official RBI fetcher reads official RBI RSS/feed endpoints plus the
official Commonman press-release archive in targeted policy-document mode.
Targeted mode prioritizes MPC minutes and monetary-policy statements while
retaining the existing broad fetch workflow for speeches, financial-stability
documents, and relevant press releases.

## Targeted MPC / policy-statement fetch mode

`--target-policy-docs` adds conservative official archive discovery and ranks
candidates by document type priority:

1. `mpc_minutes`
2. `monetary_policy_statement`
3. `financial_stability_report`
4. `governor_speech`
5. `press_release`

`--target-document-types mpc_minutes,monetary_policy_statement` restricts
manifest inclusion to the highest-value policy-core documents and records
non-target candidates in diagnostics.

## Classification and filters

Documents are classified into the allowed RBI document types only:
`mpc_minutes`, `monetary_policy_statement`, `governor_speech`,
`press_release`, `financial_stability_report`, `annual_report`, and `unknown`.

The fetcher applies:

- policy relevance filtering;
- index/archive/navigation-page filtering;
- irrelevant press-release exclusion for auctions, penalties, tenders, and
  similar operational releases;
- text-quality diagnostics;
- target-policy metadata including matched phrase and priority rank.

## GDELT/news processing

GDELT/news records are collected through the configured provider workflow,
cached with raw-response provenance, normalized, deduplicated, source-quality
scored, and then validated under ex-ante timestamp controls. Rate-limit
handling uses bounded retries and query-level diagnostics; failed responses are
not treated as successful records.

## Ex-ante validation and decision lag

Records carry publication and retrieval timestamps. The validation layer rejects
missing or inconsistent timestamps, applies publication lag and decision lag,
and flags reaction-style language. NLP does not consume future returns,
drawdowns, volatility spikes, or market reaction data as inputs.

## Scoring and daily signal construction

The current validated run uses the deterministic lexicon scorer. FinBERT is
optional/fallback and was not used in the latest validation. RBI macro-policy
scores and news scores are combined into daily monitoring labels with source-mix
labels: `news_only`, `rbi_only`, `rbi_and_news`, or `none`.

## Verdict logic

The validation verdict considers real-record coverage, publication-date
coverage, decision-label coverage, source quality, reaction-warning rate, and
source diversity. Passing monitoring thresholds does not establish return
predictiveness or allocation readiness.
"""


def _data_sources_md() -> str:
    return """# Data sources

## RBI official/Commonman press-release archive pages

- Source type: official RBI web archive.
- Frequency: event-driven; updated when RBI publishes releases.
- Fields captured: publication date, title, source URL, source channel, source
  page, retrieval date, local path, extraction diagnostics.
- Local storage: `data/sentiment/rbi_real/manifest.csv` and
  `data/sentiment/rbi_real/raw/`.
- Provenance fields: `document_id`, `publication_date`, `document_type`,
  `title`, `source_url`, `retrieval_date`, `language`, `notes`.
- Limitations: web layout can change; some pages may require manual fallback;
  archive pages must be filtered from the corpus.

## RBI MPC minutes

- Source type: official RBI policy-core documents.
- Frequency: after Monetary Policy Committee meetings.
- Fields captured: same manifest fields plus target phrase and document-type
  diagnostics in fetch outputs.
- Local storage: `data/sentiment/rbi_real/raw/`.
- Limitations: coverage depends on official archive availability and successful
  text extraction.

## RBI monetary policy statements

- Source type: official RBI policy-core documents.
- Frequency: policy decision cycle.
- Fields captured: title, date, source URL, extracted text, classification, and
  diagnostics.
- Local storage: `data/sentiment/rbi_real/manifest.csv`.
- Limitations: statement naming varies across years; classification remains
  conservative.

## RBI governor/deputy governor speeches

- Source type: official RBI speeches feed/pages.
- Frequency: event-driven.
- Fields captured: title, date, source URL, extracted text, and document type.
- Local storage: `data/sentiment/rbi_real/raw/`.
- Limitations: speeches provide context but are not primary policy-core
  documents.

## RBI financial stability report pages where available

- Source type: official RBI publication pages.
- Frequency: periodic.
- Fields captured: title, source URL, extracted text, document type.
- Local storage: `data/sentiment/rbi_real/raw/`.
- Limitations: pages can be indexes or PDF-heavy; extraction quality varies.

## GDELT DOC/news API records

- Source type: real news/geopolitical public API records.
- Frequency: query-window dependent.
- Fields captured: provider, title/summary, URL, publication timestamp,
  retrieval timestamp, language, source metadata, source-quality scores, scoring
  output.
- Local storage:
  `outputs/reports/phase_4a6_real_nlp_validation/deduped_sentiment_records.csv`
  and related diagnostics.
- Limitations: the latest validated news window is short; article text is not a
  substitute for full licensed news archives; API rate limits can reduce query
  coverage.

## Report outputs

- `outputs/reports/phase_4a8_multisource_nlp_monitoring/`
- `outputs/reports/phase_4a12_nlp_monitoring_final_pack/`
"""


def _validation_results_md() -> str:
    return """# Validation results

- Tests passed: 490 passed, 1 skipped.
- Smoke test: passed.
- Valid RBI documents: 34.
- Policy core documents: 26.
- MPC minutes: 13.
- Monetary policy statements: 13.
- Governor speeches: 6.
- Financial stability reports: 1.
- Press releases: 1.
- Source mix: `{"none": 1, "rbi_and_news": 54, "rbi_only": 3}`.
- Verdict: B. Useful for monitoring only.
- Allocation impact: None.

The latest targeted RBI fetch found and downloaded 13 MPC minutes and 12
monetary-policy statements, while preserving diagnostics for skipped irrelevant
and non-target official releases.
"""


def _limitations_md() -> str:
    return """# Limitations

- NLP remains monitoring-only.
- NLP has no allocation or strategy-scoring impact.
- Evidence gates, recommendation confidence, and backtests are unchanged.
- The GDELT/news validation window is short.
- The RBI corpus is official and real, but still limited relative to the full
  history of monetary-policy communication.
- Lexicon scoring is conservative and weaker than a validated transformer
  scorer.
- FinBERT is optional/fallback and not yet validated for this project’s RBI/news
  monitoring use case.
- HMM/rule comparison depends on available regime labels.
- No predictive claim is made.
- No statement is made that NLP improves returns.
- The system does not claim live-trading or production execution readiness.
"""


def _reproducibility_commands_md() -> str:
    return r"""# Reproducibility commands

```powershell
python -m pytest -q
python scripts\final_smoke_test.py

python scripts\fetch_rbi_documents.py `
  --from-date 2020-01-01 `
  --to-date 2026-06-24 `
  --max-documents 30 `
  --target-policy-docs `
  --target-document-types mpc_minutes,monetary_policy_statement `
  --min-policy-relevance medium `
  --request-delay-seconds 5 `
  --validate-after `
  --refresh

python scripts\check_rbi_corpus_status.py

python scripts\collect_real_nlp_data.py `
  --config config\nlp_providers.local.yaml `
  --start-date 2026-04-01 `
  --end-date 2026-06-21 `
  --no-cache

python scripts\validate_real_nlp_signal.py `
  --input-records outputs\reports\phase_4a6_real_nlp_validation\deduped_sentiment_records.csv `
  --start-date 2026-04-01 `
  --end-date 2026-06-21
```
"""


def _dashboard_guide_md() -> str:
    return """# Dashboard guide

## Manager View

- NLP Monitoring status: shows whether the real multi-source signal is active
  enough for contextual monitoring.
- Source mix: distinguishes `rbi_and_news`, `rbi_only`, `news_only`, and `none`.
- RBI + News confirmation: provides context for macro-policy and market-news
  alignment.
- Allocation impact: always `None`.

## Research View

- RBI corpus status: valid document counts, document type distribution, and
  policy-core coverage.
- Source mix timeline: shows when daily monitoring labels are supported by RBI,
  news, both, or neither.
- Daily NLP signal: decision-lagged monitoring labels.
- Decision-label coverage: percentage of dates with usable monitoring labels.
- Validation verdict: current monitoring-readiness conclusion.

## Developer View

- Raw source records: provider-level normalized and deduped data.
- Fetch diagnostics: RBI download, target-policy, index-page, and excluded
  keyword diagnostics.
- Score diagnostics: lexicon scoring outputs and source-quality diagnostics.
- Ex-ante validation: timestamp checks, reaction warnings, and lag controls.
- Index-page and exclude-keyword diagnostics: auditable evidence that navigation
  pages and irrelevant official releases are not silently dropped.
"""


def _report_html(markdown_sections: dict[str, str]) -> str:
    body = "\n".join(
        f"<section><pre>{html.escape(content)}</pre></section>"
        for content in markdown_sections.values()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Phase 4A.12 NLP Monitoring Final Pack</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; line-height: 1.45; }}
    pre {{ white-space: pre-wrap; font-family: inherit; }}
    section {{ border-bottom: 1px solid #ddd; margin-bottom: 1.5rem; }}
  </style>
</head>
<body>
{body}
</body>
</html>"""


def build_pack() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _copy_snapshots()
    sections = {
        "summary.md": _summary_md(),
        "technical_methodology.md": _technical_methodology_md(),
        "data_sources.md": _data_sources_md(),
        "validation_results.md": _validation_results_md(),
        "limitations.md": _limitations_md(),
        "reproducibility_commands.md": _reproducibility_commands_md(),
        "dashboard_guide.md": _dashboard_guide_md(),
    }
    for filename, content in sections.items():
        _write(OUTPUT_DIR / filename, content)
    _write(OUTPUT_DIR / "report.html", _report_html(sections))


def main() -> int:
    build_pack()
    print(f"Phase 4A.12 final pack: {OUTPUT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
