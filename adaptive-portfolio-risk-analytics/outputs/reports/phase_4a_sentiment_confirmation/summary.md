# Phase 4A Sentiment Regime Confirmation

Generated: 2026-06-21

## Conclusion

**Sentiment is an auxiliary confirmation layer. It is not yet used for allocation.**

## Evidence summary

- Source: 32 synthetic, timestamped market-news records from `data/sentiment/sample_market_news.csv`.
- Scoring: deterministic Phase 4A risk-on/risk-off lexicon; no external model dependency.
- Signal: five-market-day rolling mean with a one-market-day decision lag.
- Rule-based agreement: 26.5%.
- HMM walk-forward agreement: 56.1%.
- Rule stress/crisis risk-off confirmation: 26.5%.
- HMM risk-off confirmation: 35.7%.
- Article-day coverage: 2.7%.
- Decision-label coverage: 13.5%.
- Current synthetic-fixture confirmation: Quant-Sentiment Disagreement.
- Alignment checks passed: True.

## Interpretation

The sparse synthetic feed demonstrates ingestion, scoring, lagging, alignment,
agreement, disagreement, and dashboard commentary. The agreement rates are not
empirical market evidence and do not establish that sentiment predicts returns.
Phase 4B should be considered only after timestamped real-source data, source
governance, broader coverage, and out-of-sample validation are available.
