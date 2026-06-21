# Phase 4A.3 Source Notes

Generated: 2026-06-21

## Report contract

- Delivery mode: portable technical HTML.
- Audience: technical.
- Corpus type: `synthetic_fixture`.
- Active manifest: `data\sentiment\rbi_documents\sample_manifest.csv`.
- Real manifest: `data/sentiment/rbi_real/manifest.csv`.
- Real corpus status: Real RBI corpus unavailable; synthetic fixture mode used only for pipeline validation.

## Source and methodology

- Real-corpus validation uses the exact nine-column Phase 4A.3 manifest.
- Synthetic fixtures remain under `data/sentiment/rbi_documents/`.
- Scoring uses `lexicon`; transformer models remain optional.
- Publication dates align to the first market date on or after publication.
- Macro decisions are shifted by 1 market session before comparison.
- HMM comparison uses the repository walk-forward implementation.
- The report market history is deterministic synthetic data, not live market evidence.

## Chart map

- `document_type_distribution.png`: category comparison; document type and count; shows active fallback composition.
- `agreement_and_coverage.png`: horizontal rate comparison; agreement, stress confirmation, and coverage; shows pipeline diagnostics only.

## Required structure mapping

- Technical summary: corpus availability and main conclusion.
- Key findings: active-corpus composition and macro-regime diagnostics.
- Scope/data/definitions: corpus and metric definitions.
- Methodology: validation, scoring, alignment, lagging, comparison.
- Limitations/robustness: synthetic fallback and no empirical claim.
- Recommended next steps: populate and validate a real corpus.
- Further questions: coverage and out-of-sample promotion criteria.

## Verification

- Full suite: 402 passed, 1 skipped, 64% statement coverage.
- Dashboard root and health endpoints: HTTP 200.
- Final smoke test: passed.
