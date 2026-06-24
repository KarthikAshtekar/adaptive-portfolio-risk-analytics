# Phase 4A.6 Real NLP Signal Validation

Generated: 2026-06-23

## Technical summary

**B. Useful for monitoring only**

GDELT/news-only NLP is a real news monitoring signal, not an allocation signal. Allocation impact: None.

- Real provider records: 50.
- Real RBI documents: 0.
- Real GDELT/news records: 50.
- Coverage quality: limited.
- Decision-label coverage: 93.1%.
- Valid decision-label dates: 54.
- Source mix: `{"news_only": 54, "none": 4}`.
- Source families: `["news"]`.
- RBI/news agreement: N/A.
- Multi-source monitoring active: No.
- RBI manual action required: Yes.
- Insufficient reasons: `{"decision_lag_no_prior_signal": 1, "no_valid_nlp_source_in_rolling_window": 3}`.
- Source-quality distribution: `{"high": 33, "low": 0, "medium": 17, "unknown": 0}`.
- Reaction-warning rate: 0.0%.
- Scoring status: Lexicon configured; FinBERT not used.
- HMM walk-forward agreement: N/A.
- Rule-based agreement: N/A.
- Corpus intake status: manual_action_required.
- Manual intake action required: Yes.

Real provider records were evaluated under ex-ante, reaction, quality, freshness, and coverage controls. Passing any threshold does not establish predictiveness.

GDELT-only NLP is a real news monitoring signal, not an allocation signal.
