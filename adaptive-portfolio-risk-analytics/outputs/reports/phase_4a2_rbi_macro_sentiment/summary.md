# Phase 4A.2 RBI Macro-Sentiment Confirmation

Generated: 2026-06-21

## Conclusion

**RBI macro-sentiment is now a real/reproducible confirmation layer, but it remains commentary-only until validated out of sample.**

## Evidence summary

- Corpus type: `synthetic_fixture`.
- Source: local manifest with 4 synthetic RBI-style documents; 4 loaded and 0 flagged.
- Sentence corpus: 16 cleaned and deterministically ordered sentences.
- Scoring: RBI stance/certainty/time lexicon; transformer fallback count 0.
- Signal: 63-market-day rolling stance index with a one-market-day decision lag.
- Rule-based agreement: 39.1%.
- HMM walk-forward agreement: 65.3%.
- Rule stress/crisis risk-off confirmation: 39.1%.
- HMM stress/crisis risk-off confirmation: 64.9%.
- Decision-label coverage: 19.2%.
- Current fixture macro label: risk_on_macro.
- Current fixture confirmation: Quant-Macro Disagreement.

## Interpretation

The manifest, sentence processing, scoring, rolling index, lagging, and regime
comparison are reproducible. The bundled documents and market history are
synthetic fixtures, so these statistics are pipeline evidence rather than
empirical evidence about RBI language or future returns.

## Required questions answered

1. **What RBI documents were ingested?** Four locally stored synthetic RBI-style MPC-minute and monetary-policy-statement fixtures listed in `rbi_documents.csv`.
2. **How were sentences scored?** Text was cleaned, split into stable ordered sentences, and classified with phrase dictionaries for stance, certainty, and time orientation.
3. **Which scorer was used?** The deterministic `rbi_macro_lexicon` version `1.0`; transformer models were not required.
4. **How was look-ahead avoided?** Publication dates were aligned to the first available market date, then observed macro labels were shifted by one market session.
5. **What is the Macro-Stance Index?** A 63-session rolling index where `net_stance_score = hawkish_share - dovish_share` and `macro_risk_score = net_stance_score + uncertainty_share`.
6. **Does it confirm HMM regimes?** HMM walk-forward agreement was 65.3%; stress/risk-off confirmation was 64.9%.
7. **Does it confirm rule-based regimes?** Rule-based agreement was 39.1%; stress/crisis risk-off confirmation was 39.1%.
8. **Are risk-off signals visible before or during stress?** 0 pre-stress macro risk-off alignments were identified; during-stress confirmation rates are reported above and in `macro_regime_comparison.csv`.
9. **Is coverage sufficient?** Insufficient for empirical or allocation conclusions because the bundled corpus is synthetic, sparse, and covers only 19.2% of decision dates.
10. **Should it move into allocation testing?** No. It should remain commentary-only until a governed real-document corpus is validated out of sample.
