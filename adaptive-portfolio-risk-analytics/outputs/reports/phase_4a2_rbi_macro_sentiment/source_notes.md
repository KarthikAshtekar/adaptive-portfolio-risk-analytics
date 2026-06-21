# Phase 4A.2 Source Notes

Generated: 2026-06-21

## RBI document source

- Corpus type: `synthetic_fixture`.
- Manifest: `data/sentiment/rbi_documents/sample_manifest.csv`
- Canonical manifest columns: `document_id`, `publication_date`, `document_type`, `title`, `local_path`, `source_url`.
- Documents: 4 locally stored synthetic RBI-style fixtures.
- The fixture does not reproduce long passages from RBI publications.
- `manifest.csv` is also provided as the default local-ingestion convention.

## Processing and scoring

- `.txt`, `.md`, and `.csv` are supported directly.
- PDF extraction is optional through `pypdf`; unavailable or unreadable PDFs are flagged without aborting the corpus.
- Sentence IDs preserve document order and are stable across reruns.
- Lexicon scoring classifies hawkish/neutral/dovish stance, certain/uncertain/neutral certainty, and forward/backward/current/unknown time orientation.
- Optional Hugging Face model IDs are configured for RBI stance, certainty, and time labels; failed model loads fall back per sentence to the lexicon.

## Quantitative comparison source

- Deterministic synthetic eight-asset daily return fixture from 2022-01-03 to 2026-06-19.
- Rule-based labels use the repository feature and classification code with a one-session lag.
- HMM labels use the repository expanding-window walk-forward implementation.
- Full-sample HMM is not used for decision-facing comparison.

## Look-ahead controls

- Publication dates are assigned to the first market date on or after publication.
- Observed macro labels are shifted by one market session before becoming decision labels.
- Decision-source dates precede their decision dates.

## Limitations

- This report demonstrates reproducible local ingestion, not a live or comprehensive RBI archive.
- The bundled corpus is intentionally small and synthetic.
- Lexicon labels can miss negation, context, speaker differences, and policy nuance.
- Transformer outputs require independent model and data validation.
- Lead/lag diagnostics are descriptive and do not establish predictive power.
- RBI macro-sentiment does not alter strategy scores, gates, confidence, adaptive policy, allocation, or portfolio weights.

## Verification

- `python -m pytest -q`: 393 passed, 1 skipped.
- Total statement coverage: 63%.
- Headless Streamlit root and health endpoints: HTTP 200.
- `python scripts/final_smoke_test.py`: passed.
