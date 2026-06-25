# Phase 4A.6 Real NLP Signal Validation

Generated: 2026-06-25

## Technical summary

**B. Useful for monitoring only**

RBI + news NLP is useful for multi-source monitoring only. Allocation impact: None.

- Real provider records: 50.
- Real RBI documents: 34.
- Real GDELT/news records: 50.
- Coverage quality: sufficient.
- Decision-label coverage: 98.3%.
- Valid decision-label dates: 57.
- Source mix: `{"none": 1, "rbi_and_news": 54, "rbi_only": 3}`.
- Source families: `["news", "rbi_macro"]`.
- RBI/news agreement: 85.2%.
- Multi-source monitoring active: Yes.
- RBI manual action required: No.
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
