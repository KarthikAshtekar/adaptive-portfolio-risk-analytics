# Phase 4A.5 API-Based Ex-Ante NLP Risk Monitoring

Generated: 2026-06-21

## Technical summary

**API-based NLP risk is a confirmation and monitoring layer only. It is not yet promoted to allocation or strategy scoring.**

- Provider mode: offline fixtures and local manifests.
- Providers configured: rbi, earnings_calls, gdelt, alpha_vantage_news.
- Providers returning valid records: earnings_calls, gdelt.
- Ex-ante valid records: 7 of 7.
- Possible reaction-data warnings: 1.
- FinBERT status: lexicon fallback for 7 records.
- Composite NLP decision coverage: 4.2%.
- HMM walk-forward agreement: 42.9%.
- Rule-based agreement: 65.3%.
- Suitable for allocation testing: False.

This fixture-backed run validates provider normalization, timestamp discipline,
reaction-data warnings, fallback scoring, composite construction, and lagged
regime comparison. It is not empirical evidence from live provider coverage.
