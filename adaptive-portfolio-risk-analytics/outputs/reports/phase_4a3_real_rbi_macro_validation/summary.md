# Phase 4A.3 Real RBI Corpus Validation

Generated: 2026-06-21

## Technical summary

**Real RBI corpus unavailable; synthetic fixture mode used only for pipeline validation.**

- Corpus type: `synthetic_fixture`.
- Real RBI documents ingested: 0.
- Real corpus date range: Unavailable.
- Active fallback documents: 4; active sentences: 16.
- Scoring method: `lexicon`.
- Decision coverage: 19.2%.
- HMM walk-forward agreement: 64.3%.
- Rule-based agreement: 39.1%.
- Coverage sufficient for empirical conclusions: False.

## Conservative conclusion

**RBI macro sentiment remains a confirmation layer until real-document coverage is sufficient and out-of-sample validation supports promotion.**

The current report does not contain empirical real-RBI results. Synthetic
fixtures and deterministic market history are retained only to verify corpus
fallback, sentence scoring, lagging, coverage diagnostics, and regime
comparison.
