# Regime-Aware Portfolio Risk Analytics Platform

**v1.2.0 — API-Based Ex-Ante NLP Risk Monitoring**

An evidence-gated portfolio research platform that combines hierarchical risk allocation, FRM risk diagnostics, regime-aware adaptive overlays, timestamped market-news and RBI macro-sentiment confirmation, CPCV-style robustness validation, and a simplified manager-facing decision interface.

## 1. Executive summary

This project is a Python research platform for portfolio construction, historical backtesting, risk diagnostics, regime analysis, adaptive risk control, sentiment confirmation, and evidence-gated strategy selection. It combines fixed allocation methods with lagged rule-based and HMM walk-forward regime decisions, evaluates net-of-cost outcomes, and exposes the results through a Streamlit decision-support dashboard.

The platform is designed to answer a practical question: should a regime-aware strategy replace a strong fixed portfolio, or should it be used selectively for risk control? The validated strategic conclusion remains role-based rather than winner-takes-all. Phase 4A through Phase 4A.5 add sentiment and NLP only as timestamped confirmation, monitoring, and explanation layers.

## 2. Final strategic conclusion

This project does not claim one universal best strategy.

It classifies strategies by role:

- **HERC:** strategic growth core
- **HMM Conservative:** drawdown-control overlay
- **Rule-based Conservative:** robustness reference and HMM fallback
- **Equal Weight:** benchmark

HERC remains the preferred growth core because it produced the strongest CAGR and terminal value in the matched primary scenario. HMM Conservative materially improved drawdown and drawdown efficiency, but gave up growth, so it is positioned as a risk-control overlay rather than a replacement. Rule-based Conservative is simpler, achieved the strongest adaptive CPCV rank in the current artifact, and remains the robustness reference when HMM evidence is unavailable or unstable.

Recommendation confidence is **Moderate** because adaptive CPCV successful-fold coverage is limited.

## 3. Architecture overview

```text
Data Layer
  → preprocessing and aligned return/risk matrices
Strategy Layer
  → Equal Weight / Inverse Volatility / HRP / HERC
Risk Analytics Layer
  → performance, tail risk, drawdown, stress, liquidity, and active risk
Regime Layer
  → lagged rule-based regimes / HMM walk-forward regimes
Sentiment Layer
  → market-news CSV + RBI manifest ingestion / sentence scoring / lagged confirmation
Adaptive Layer
  → Conservative / Balanced / Aggressive policies and defensive sleeves
Validation Layer
  → sensitivity / stress / CPCV-style robustness / replication
Selection Layer
  → evidence gates / profile mapping / strategy roles
Dashboard Layer
  → Manager / Research / Developer views
```

Implementation is organized under `src/` by data pipeline, covariance, clustering, optimization, backtesting, analytics, regime, sentiment, adaptive, experiments, validation, selection, and dashboard concerns. The v1.0 architecture freeze remains documented in the [final architecture summary](outputs/final_project_pack/architecture_summary.md).

## 4. Phase-wise implementation summary

- **Phase 1 — Portfolio construction and backtesting:** data preparation, Equal Weight, Inverse Volatility, HRP, HERC, covariance estimators, rolling backtests, rebalancing, turnover, transaction costs, and performance comparison.
- **Phase 2 — FRM risk analytics:** VaR, ES/CVaR, exceptions, stress testing, drawdown duration, concentration, liquidity diagnostics, and benchmark-relative active risk.
- **Phase 3A — Robustness validation:** purged and embargoed time-block combinations, fold summaries, stability scores, and objective-specific robustness ranking.
- **Phase 3B — Regime detection:** explainable rule-based regimes, full-sample HMM historical visualization, and trading-safe HMM walk-forward decisions.
- **Phase 3C — Adaptive allocation:** regime-dependent allocator, covariance, volatility target, risky-cap, defensive-floor, and rebalance behavior.
- **Phase 3D — Adaptive experimentation:** fixed-versus-adaptive sensitivity, attribution, stress comparison, and bounded adaptive CPCV.
- **Phase 3E — Replication and consistency:** centralized defensive-return handling, matched scenario replication, cost/sleeve sensitivity, and policy-tuning checks.
- **Phase 3F — Strategy selection:** PASS/WARN/FAIL/NOT_AVAILABLE evidence gates, investor-profile mapping, role classification, confidence adjustment, and simplified manager output.
- **Phase 4A — Sentiment Regime Confirmation Layer:** local timestamped CSV ingestion, lightweight lexicon scoring, market-index alignment, rolling sentiment, one-session decision lag, quantitative-regime comparison, and dashboard commentary.
- **Phase 4A.2 — RBI Macro-Sentiment Confirmation:** manifest-driven local RBI document ingestion, sentence-level stance/certainty/time scoring, optional Hugging Face adapters with deterministic fallback, a lagged macro stance index, quantitative-regime comparison, and dashboard commentary.
- **Phase 4A.3 — Real RBI Corpus Validation:** governed real-document directory and manifest contracts, validation diagnostics, invalid-row isolation, real-versus-synthetic dashboard provenance, coverage thresholds, and a corpus-typed empirical validation report.
- **Phase 4A.5 — API-Based Ex-Ante Sentiment Ingestion:** optional RBI, earnings-call, GDELT/news, and Alpha Vantage providers; provenance-preserving normalization; ex-ante timestamp validation; reaction-data warnings; optional local FinBERT scoring; and a decision-lagged composite NLP risk monitor.

### Phase 4A — Sentiment Regime Confirmation Layer

Sentiment is used only as a regime-confirmation and explanation layer. It does not directly change portfolio weights in v1.2.0.

The default implementation uses a local CSV and dependency-light lexicon scorer. It produces observed and lagged decision sentiment, compares sentiment with rule-based and HMM walk-forward regimes, reports coverage and disagreement, and exposes timestamp/look-ahead checks. It does not claim that sentiment predicts returns.

### Phase 4A.2 — RBI Macro-Sentiment Confirmation

Phase 4A.2 ingests locally stored RBI documents through a manifest, preserves row-level load errors, extracts stable ordered sentences, and scores monetary-policy stance, certainty, and time orientation. It supports `.txt`, `.md`, and `.csv` directly; PDF extraction is optional. A 63-market-day macro stance index is shifted by at least one session before comparison with rule-based and HMM walk-forward decision regimes.

The bundled RBI-style corpus is synthetic and intended for reproducible pipeline validation. Optional Hugging Face models are local-first and fall back sentence-by-sentence to the deterministic lexicon. RBI macro-sentiment does not enter strategy ranking, evidence gates, recommendation confidence, adaptive allocation, backtests, or portfolio weights.

The lexicon is a deterministic fallback and is not equivalent to the optional transformer-based RBI models. Transformer models remain optional, and the usefulness of this layer depends on real document coverage and governed timestamps.

### Phase 4A.3 — Real RBI Corpus Validation

Phase 4A.3 adds a local real-RBI corpus contract under `data/sentiment/rbi_real/`, including raw and processed directories, an exact nine-column manifest, a directory-to-manifest builder, manifest validation, and a loader that excludes invalid records while preserving diagnostics. The supported document types are MPC minutes, monetary-policy statements, governor speeches, press releases, financial-stability reports, annual reports, and unknown.

The empirical runner writes corpus-typed document, sentence, macro-index, regime-comparison, disagreement, coverage, and corpus-diagnostic outputs. Dashboard Manager, Research, and Developer views distinguish `real_rbi` from `synthetic_fixture`, expose coverage and validation status, and fall back safely when no valid real corpus exists. The current repository contains no valid real RBI documents, so the Phase 4A.3 report states: **Real RBI corpus unavailable; synthetic fixture mode used only for pipeline validation.**

See the [manual RBI corpus download guide](docs/rbi_corpus_manual_download.md) before populating the local manifest. RBI content remains commentary-only and does not affect allocation, scoring, gates, recommendation confidence, or backtests.

### Phase 4A.5 — API-Based Ex-Ante Sentiment Ingestion

Phase 4A.5 adds a common provider contract under `src/sentiment/providers/`. `RBIProvider` supports optional configured feeds with local-manifest fallback, `EarningsCallProvider` loads reviewed local transcripts, `GDELTProvider` supports mockable news queries, and `AlphaVantageNewsProvider` skips safely unless explicitly enabled with `ALPHAVANTAGE_API_KEY`.

Optional provider environment variables are `RBI_FEEDS_ENABLED`,
`RBI_FEED_URLS`, `RBI_LOCAL_MANIFEST_PATH`, `GDELT_NEWS_ENABLED`,
`ALPHAVANTAGE_NEWS_ENABLED`, and `ALPHAVANTAGE_API_KEY`. Feed and API switches
default to disabled.

The unified ingestion runner stores raw-response provenance, normalized records, provider diagnostics, and deduplicated records. Every normalized external record includes publication and retrieval timestamps, provider, source URL, language, and raw metadata. Tests use local fixtures and mocks; internet access is not required.

Ex-ante validation rejects missing or inconsistent timestamps, applies a publication lag, and flags phrases that may describe market reaction rather than forward-looking information. Flagged records remain auditable but are excluded from composite source scoring. Optional FinBERT scoring uses local model files only and falls back to the existing deterministic lexicon with explicit fallback metadata.

The composite NLP index combines available RBI macro, earnings-sector, and news/geopolitical textual risk components, requires adequate source coverage, aligns to the trading calendar, and is decision-lagged before comparison with rule-based and HMM walk-forward regimes. Provider sentiment is retained as metadata and is not blindly trusted.

**NLP remains commentary-only and does not affect allocation, portfolio weights, strategy scoring, evidence gates, recommendation confidence, or backtests.**

## 5. Dashboard modes

- **Manager View:** a simplified recommendation interface showing the strategic core, optional overlay or fallback, confidence, net trade-offs, and compact market-news and RBI macro-confirmation cards without raw text.
- **Research View:** methodology controls for covariance, rebalancing, regimes, sentiment sources, RBI manifest/scorer/lag settings, adaptive policy, sensitivity, CPCV, attribution, and selection diagnostics.
- **Developer / Debug View:** raw HMM, market-news, RBI documents, sentence scores, alignment, fallback, look-ahead, CPCV, adaptive, gate, and reconciliation diagnostics.

The default manager workflow uses a Balanced objective, Moderate cost assumption, and HMM Walk-Forward Conservative as the risk-control candidate. The recommendation remains evidence-gated and can fall back to Rule-based Conservative.

## 6. Methodology

The fixed benchmark layer supports Equal Weight, Inverse Volatility, HRP, and HERC. Covariance estimation supports sample, Ledoit-Wolf, EWMA, and EWMA plus Ledoit-Wolf methods. Backtests apply weights out of sample, include transaction costs, and retain separate gross and net return series.

Regime-aware decisions use information available before the applied return:

- Rule-based observed labels are shifted by at least one period.
- HMM adaptive experiments use expanding-window walk-forward inference and a decision lag.
- Full-sample HMM is historical visualization only.
- Trading-safe recommendations use HMM walk-forward decisions with lagging.

CPCV-style validation uses ordered time blocks, test-block combinations, purge and embargo controls, fold-level reruns, and objective-specific stability and robustness scores. It is a pragmatic robustness framework, not complete independent-path CPCV.

Sentiment records are parsed and deduplicated, scored with a fixed risk-on/risk-off lexicon, assigned to observed market dates, aggregated over a rolling window, and shifted by at least one market session before becoming decision-facing confirmation. Sentiment does not enter `score_candidates`, adaptive policy selection, or portfolio-weight calculations.

RBI documents are loaded from a local manifest, split into deterministic ordered sentences, and classified for hawkish/neutral/dovish stance, certainty, and time orientation. The rolling macro index uses `hawkish_share - dovish_share + uncertainty_share` as a descriptive macro-risk score and is lagged before decision-facing comparison. It remains commentary-only.

See the [final methodology report](outputs/final_project_pack/methodology_report.md) for concise definitions and implementation caveats.

## 7. Key results

The latest matched primary scenario uses the Core Diversified universe, evaluation from January 1, 2020 through June 19, 2026, initial capital of 1,000,000, 10 bps base cost plus 5 bps slippage, and a synthetic 4% annual defensive sleeve for adaptive strategies.

| Strategy | Role | Net CAGR | Net Calmar | Max drawdown | Net final value |
| --- | --- | ---: | ---: | ---: | ---: |
| HERC | Strategic growth core | 15.01% | 0.794 | -18.91% | 2,434,518 |
| HMM Walk-Forward Conservative | Risk-control overlay | 11.84% | 1.521 | -7.78% | 2,037,180 |
| Rule-based Conservative | Robustness reference / fallback | 10.80% | 1.114 | -9.69% | 1,920,070 |
| Equal Weight | Benchmark | 12.56% | 0.381 | -32.94% | 2,122,050 |

The current CPCV artifact ranks Rule-based Conservative first on Calmar robustness, but only 6 of 15 folds succeeded. HMM Conservative succeeded on 3 of 15 folds. These coverage limits are why confidence remains Moderate despite favorable successful-fold results.

See the [final results summary](outputs/final_project_pack/final_results_summary.md) for interpretation and source notes.

## 8. Validation and tests

The project includes unit and integration coverage across allocation, covariance, backtesting, costs, risk analytics, regimes, sentiment ingestion/scoring/alignment, adaptive policies, CPCV, strategy selection, dashboard modes, net labels, and reconciliation.

Final freeze verification:

```bash
python -m pytest -q
python scripts/final_smoke_test.py
```

The dashboard is also launched headlessly and checked through Streamlit's health endpoint. Exact freeze results are recorded in the [final validation checklist](outputs/final_project_pack/final_validation_checklist.md).

Verified on June 21, 2026: **418 passed, 1 skipped, 64% statement coverage, Phase 4A.5 report generated, smoke test passed, dashboard root and health HTTP 200**.

## 9. Reports and artifacts

- [Final project pack index](outputs/final_project_pack/INDEX.md)
- [Phase 4A sentiment-confirmation report](outputs/reports/phase_4a_sentiment_confirmation/report.html)
- [Phase 4A.2 RBI macro-sentiment report](outputs/reports/phase_4a2_rbi_macro_sentiment/report.html)
- [Phase 4A.3 real RBI corpus validation report](outputs/reports/phase_4a3_real_rbi_macro_validation/report.html)
- [Phase 4A.5 API-based ex-ante NLP monitoring report](outputs/reports/phase_4a5_api_sentiment_ingestion/report.html)
- [Phase 3F strategy-selection report](outputs/reports/phase_3f_strategy_selection/report.html)
- [Phase 3E replication report](outputs/reports/phase_3e_replication/report.html)
- [Post-P0 adaptive validation report](outputs/reports/post_p0_adaptive_validation/report.html)
- [Final project summary](outputs/final_project_pack/final_project_summary.md)
- [Presentation outline](outputs/final_project_pack/presentation_outline.md)
- [Viva questions and answers](outputs/final_project_pack/viva_questions_and_answers.md)
- [Resume bullets](outputs/final_project_pack/resume_bullets.md)

## 10. How to run

Python 3.10 or later is required.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run src/dashboard/app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run src/dashboard/app.py
```

The application opens in Manager View. Market-data retrieval uses Yahoo Finance, so network availability and source revisions can affect reruns.

## 11. Limitations

- Results are historical backtests and decision-support outputs, not live-trading claims.
- Yahoo Finance histories can change, and external data availability is not guaranteed.
- The primary validated universe and date windows are bounded; results may not generalize to other markets or long regimes.
- Adaptive CPCV has limited successful-fold coverage because warm-up and model-training requirements invalidate early folds.
- Current robustness ranking reports failed-fold counts but does not directly penalize missing-fold coverage.
- HMM fitting is probabilistic, computationally heavier, and sensitive to specification and sample history.
- Defensive-sleeve assumptions affect adaptive outcomes, although v1.0 centralizes and records their source.
- Transaction-cost estimates are simplified and do not constitute a full market-impact model.
- Historical VaR/ES and experiment VaR/CVaR use different sign presentations and must be interpreted by their documented API convention.
- The Phase 4A lexicon does not resolve context, negation, source credibility, or entity relevance.
- The bundled sentiment feed is synthetic and intended for pipeline demonstration; it is not empirical evidence.
- Sparse article coverage can produce stale or insufficient confirmation states.
- The bundled RBI-style documents are synthetic fixtures, not a live or comprehensive RBI archive.
- No valid real RBI documents are bundled; current Phase 4A.3 metrics validate the fallback pipeline and are not empirical RBI evidence.
- RBI stance lexicons can miss negation, speaker differences, policy nuance, and document-specific context.
- Optional transformer outputs require independent model, timestamp, and out-of-sample validation.
- External APIs are optional, may require credentials, and can revise or limit historical responses.
- Market-reaction language is flagged and is not treated as a pure ex-ante signal.
- Current Phase 4A.5 report data are offline fixtures and do not establish predictive or allocation value.

## 12. Future work

- Add explicit CPCV fold-coverage eligibility or penalties.
- Replicate across broader universes, regions, and longer market histories.
- Strengthen data versioning, model governance, and reproducibility metadata.
- Extend transaction-cost modeling with liquidity-aware market impact.
- Evaluate alternative probabilistic regime models under the same walk-forward safety contract.
- Phase 4B could test sentiment as a gated allocation feature after out-of-sample validation.
- Add governed real RBI source ingestion, publication-time rules, document provenance, and broader historical coverage before any Phase 4B test.
- Evaluate VADER or FinBERT only as optional scorers after establishing a reproducible timestamped corpus and baseline.
