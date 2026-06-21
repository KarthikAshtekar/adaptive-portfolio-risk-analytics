# Phase 4A Source Notes

Generated: 2026-06-21

## Sentiment source

- Bundled file: `data/sentiment/sample_market_news.csv`
- Records: 32
- Source type: synthetic research sample
- No current or scraped news is embedded.

## Quantitative comparison source

- Deterministic synthetic eight-asset daily return fixture from 2022-01-03 to 2026-06-19.
- Rule-based labels use the repository Phase 3B feature and classification code.
- HMM labels use the repository two-state expanding-window walk-forward implementation.
- Full-sample HMM is not used.

## Look-ahead controls

- Timestamps are parsed before scoring.
- Records are assigned to the first market date on or after their calendar date.
- Observed rolling sentiment is shifted by one market session before becoming a decision label.
- Decision-source timestamps are checked to precede each decision date.
- All alignment checks passed: True.

## Scoring

- Model: `phase4a_lexicon`, version `1.0`.
- Positive score means risk-on.
- Negative score means risk-off.
- Near-zero score means neutral.
- VADER and FinBERT are not required or used.

## Limitations

- Both news and market history in this report are synthetic fixtures.
- The feed is intentionally sparse, so article and decision coverage are limited.
- Lexicon scoring does not resolve context, negation, source credibility, or entity relevance.
- Publication-time and market-close rules require source-specific governance for real data.
- Leading/lagging diagnostics are descriptive and do not establish prediction.
- Sentiment does not change allocation, policy selection, strategy scores, or confidence.

## Verification

- `python -m pytest -q`: 375 passed, 1 skipped.
- Total statement coverage: 63%.
- Headless Streamlit root and health endpoints: HTTP 200.
- `python scripts/final_smoke_test.py`: passed.
