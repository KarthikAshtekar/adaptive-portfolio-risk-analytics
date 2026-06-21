# Phase 4A.5 Source Notes

Generated: 2026-06-21

## Report contract

- Delivery mode: portable technical HTML.
- Audience: technical.
- Provider mode: offline fixtures and local manifests.
- Live API calls: disabled.
- Market history: deterministic synthetic data for pipeline comparison only.
- Selection/allocation impact: none.

## Source inventory

- RBI provider: empty real manifest with synthetic local fallback; rows without source URLs remain invalid provider diagnostics.
- Earnings provider: synthetic local transcript fixtures.
- GDELT provider: fixture JSON.
- Alpha Vantage provider: enabled test path with missing key; skipped safely.
- RBI macro component: existing synthetic RBI fixture pipeline.

## Chart map

- `provider_coverage.png`: ranked horizontal bars; valid records by provider; supports provider-return and provenance findings.
- `composite_label_distribution.png`: horizontal bars; market sessions by lagged composite label; supports coverage and label-distribution findings.

## Required structure mapping

- Technical summary: provider status, ex-ante validity, fallback, coverage, and conclusion.
- Key findings: provider coverage and composite label distribution.
- Scope/data/definitions: normalized schema, source types, and coverage definitions.
- Methodology: provider ingestion, timestamp validation, reaction flags, scoring, lagging, and regime comparison.
- Limitations/robustness: fixtures, no live API evidence, no predictive claim.
- Recommended next steps: governed provider trials and out-of-sample monitoring.
- Further questions: minimum coverage and source-governance thresholds.

## Verification

- Full suite: 418 passed, 1 skipped, 64% statement coverage.
- Dashboard root and health endpoints: HTTP 200.
- Final smoke test: passed.
