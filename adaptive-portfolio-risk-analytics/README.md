# Regime-Aware Portfolio Risk Analytics Platform

**v1.3.0 — Final Integrated Portfolio Risk Analytics Release**

An evidence-gated portfolio research platform that combines hierarchical risk allocation, FRM risk diagnostics, regime-aware adaptive overlays, timestamped market-news and RBI macro-sentiment confirmation, CPCV-style robustness validation, and a simplified manager-facing decision interface.

## 1. Executive summary

This project is a Python research platform for portfolio construction, historical backtesting, risk diagnostics, regime analysis, adaptive risk control, sentiment confirmation, and evidence-gated strategy selection. It combines fixed allocation methods with lagged rule-based and HMM walk-forward regime decisions, evaluates net-of-cost outcomes, and exposes the results through a Streamlit decision-support dashboard.

The platform is designed to answer a practical question: should a regime-aware strategy replace a strong fixed portfolio, or should it be used selectively for risk control? The validated strategic conclusion remains role-based rather than winner-takes-all. Phase 4A through Phase 4A.13 add sentiment and NLP only as timestamped confirmation, monitoring, explanation, and shadow-analysis layers. v1.3.0 packages the full portfolio, regime, FRM analytics, Pain Ratio, real RBI/news NLP monitoring, and NLP shadow-impact work into a final integrated release pack.

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
- **Phase 4A.6 — Real Provider Data Collection and NLP Signal Validation:** YAML provider configuration, offline-safe collection and caching, real-versus-fixture provenance, source-quality scoring, coverage/freshness thresholds, and a conservative empirical-validation report.
- **Phase 4A.7 — Real NLP Data Acquisition Workflow:** governed RBI, earnings-call, and news intake templates; legal/provenance guidance; placeholder exclusion; corpus validation; and dashboard intake readiness.
- **Phase 4A.8 — Real RBI + News Multi-Source NLP Monitoring:** local real-RBI bootstrap/import/status tooling, RBI macro-index integration with real GDELT/news monitoring, explicit `news_only`, `rbi_only`, and `rbi_and_news` source-mix diagnostics, and monitoring-only multi-source reports.
- **Phase 4A.9–4A.11 — Official RBI Incremental + Targeted Policy Fetcher:** official RBI feed/page discovery, date-range filtering, local `.txt` caching, manifest updates, index/archive-page filtering, irrelevant-release exclusion, targeted MPC minutes / monetary-policy-statement discovery, diagnostics, manual-fallback warnings, and validation hooks for reproducible RBI corpus population.
- **Phase 4A.12 — v1.2.8 NLP Monitoring Finalization Pack:** final documentation, methodology, validation, limitations, reproducibility commands, dashboard guide, and summary snapshots for the real RBI + real news monitoring module.
- **Phase 4A.13 — v1.2.9 Pain Ratio and NLP Shadow Impact Analysis:** Pain Index and Pain Ratio are added alongside Calmar, and a decision-lagged RBI/news NLP shadow experiment compares HERC, HMM Conservative, Rule Conservative, and two NLP overlay variants without promoting NLP to production allocation.
- **v1.3.0 — Final Integrated Portfolio Risk Analytics Release:** faculty-facing release pack consolidating portfolio construction, FRM analytics, adaptive allocation, Pain Ratio, real RBI/news NLP monitoring, NLP shadow impact, dashboard guidance, reproducibility commands, limitations, evidence matrix, and viva answers.

### Phase 4A — Sentiment Regime Confirmation Layer

Sentiment is used only as a regime-confirmation and explanation layer. It does not directly change portfolio weights in v1.3.0.

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

### Phase 4A.6 — Real Provider Data Collection and NLP Signal Validation

Provider settings are defined in
[`config/nlp_providers.example.yaml`](config/nlp_providers.example.yaml). Copy
that file for local changes and keep credentials in environment variables; API
key values are never stored in configuration or diagnostics. Enabled local
providers validate their manifests, and Alpha Vantage requires its configured
environment variable only when that provider is enabled.

The collection command writes raw provider responses to the cache under
`data/sentiment/cache/` and normalized evidence to
`outputs/reports/phase_4a6_real_nlp_validation/`. `--no-live` hard-disables
network calls and allows only local manifests or existing cached responses.
Bundled synthetic transcripts and placeholder domains remain useful for
deterministic tests, but the provenance classifier excludes them from real
record counts and empirical claims.

GDELT live collection may rate-limit requests. Start with a small query set;
the default delay is six seconds between queries, and HTTP 429 responses use
bounded retries with query-level diagnostics. After a failed live run, use
`--no-cache` to ignore the existing cache and force a fresh provider request.
Failed, rate-limited, or non-JSON responses are not written as successful
cache entries.

```bash
python scripts/collect_real_nlp_data.py --config config/nlp_providers.example.yaml --start-date 2020-01-01 --end-date 2026-06-21 --no-live
python scripts/collect_real_nlp_data.py --config config/nlp_providers.local.yaml --start-date 2026-04-01 --end-date 2026-06-21 --no-cache
python scripts/validate_real_nlp_signal.py --input-records outputs/reports/phase_4a6_real_nlp_validation/deduped_sentiment_records.csv
```

Validation scores official/known sources, publication time, URL, entity/topic,
language support, duplicate risk, and reaction-data warnings. The default
eligibility thresholds are 50 real records, 20 publication dates, 20%
decision-label coverage, and at most a 25% reaction-warning rate. Sparse or
missing real data produces verdict **C. Insufficient real-data coverage** and
the exact Manager caveat: **NLP signal is monitoring-only due to insufficient
real-data coverage.**

The v1.2.8 validation pipeline converts real GDELT/news records into a
decision-lagged daily monitoring signal when record/date thresholds are met.
News-only data may produce `nlp_risk_on`, `nlp_neutral`, or `nlp_risk_off`
labels with `source_mix = news_only`; this is intentionally marked as limited
source diversity and is not allocation-ready evidence. The daily audit file is
`daily_nlp_signal.csv`, and record-level scoring remains in
`scored_records.csv`.

The composite remains decision-lagged and monitoring-only. Phase 4A.6 does not
change allocation, strategy scores, evidence gates, recommendation confidence,
or any backtest.

### Phase 4A.7 — Real NLP Data Acquisition Workflow

The [real-data acquisition guide](docs/nlp_real_data_acquisition_guide.md)
documents legally cautious intake for three channels:

- RBI documents under `data/sentiment/rbi_real/raw/`, with metadata in
  `data/sentiment/rbi_real/manifest.csv`.
- Private or legally available earnings transcripts under
  `data/sentiment/earnings_calls/raw/`, with metadata in
  `data/sentiment/earnings_calls/manifest.csv`.
- Permitted news summaries or exports under `data/sentiment/news_real/raw/`,
  with normalized metadata in `data/sentiment/news_real/manifest.csv`.

Each directory contains a `manifest_template.csv`, `intake_notes.md`, and an
explicit `DO_NOT_USE_PLACEHOLDER` example. Raw directories are ignored by Git
to reduce accidental commits of private, large, paid, or copyrighted material.
Placeholder and bundled fixture rows never count as real evidence.

Validate intake before collection:

```bash
python scripts/validate_nlp_corpus_intake.py
python scripts/collect_real_nlp_data.py --config config/nlp_providers.example.yaml --start-date 2020-01-01 --end-date 2026-06-21 --no-live
python scripts/validate_real_nlp_signal.py --input-records outputs/reports/phase_4a6_real_nlp_validation/deduped_sentiment_records.csv
```

The validator writes `intake_status.csv` plus RBI, earnings, and news row
diagnostics under `outputs/reports/nlp_corpus_intake_validation/`. Missing
manifests, invalid files, duplicate IDs, bad timestamps, or zero valid real
records are reported as manual action required rather than treated as failures.

The dashboard Research View exposes corpus readiness and valid-real counts;
Developer View exposes row-level intake diagnostics. Manager View states that
NLP monitoring is inactive or illustrative until real text coverage is
sufficient. This workflow adds no model and does not change allocation,
strategy scoring, evidence gates, recommendation confidence, or backtests.
GDELT and all other NLP inputs remain monitoring-only.

### Phase 4A.8 — Real RBI + News Multi-Source NLP Monitoring

Phase 4A.8 adds reproducible local RBI corpus population tools and combines
valid real RBI macro-policy documents with real GDELT/news monitoring when both
are present. The composite source mix is labeled as `news_only`, `rbi_only`,
`rbi_and_news`, or `none` by decision date. RBI coverage can improve monitoring
context, but it is not a hard allocation gate and does not make NLP
allocation-ready.

Bootstrap the governed local RBI corpus:

```bash
python scripts/bootstrap_rbi_real_corpus.py
```

Import one manually extracted public RBI text file:

```bash
python scripts/import_rbi_text_document.py --document-id RBI_MPC_MINUTES_2026_06 --publication-date 2026-06-06 --document-type mpc_minutes --title "Minutes of the Monetary Policy Committee Meeting June 2026" --source-url "https://..." --input-text-file path/to/local/extracted_text.txt --retrieval-date 2026-06-23
```

Check RBI sufficiency and rerun monitoring validation:

```bash
python scripts/check_rbi_corpus_status.py
python scripts/validate_real_nlp_signal.py --input-records outputs/reports/phase_4a6_real_nlp_validation/deduped_sentiment_records.csv --start-date 2026-04-01 --end-date 2026-06-21
```

The Phase 4A.8 report is written under
`outputs/reports/phase_4a8_multisource_nlp_monitoring/`. If no valid real RBI
documents are present, RBI manual action is required and the current signal
remains news-only monitoring.

**NLP remains monitoring-only and does not affect allocation.**

### Phase 4A.9 — Official RBI Incremental Fetcher

Phase 4A.9 adds a safe official-source fetcher for populating the governed
real-RBI corpus. The fetcher reads configured official RBI RSS/feed or page
URLs, filters by date and keyword relevance, saves extracted UTF-8 `.txt`
documents under `data/sentiment/rbi_real/raw/`, updates
`data/sentiment/rbi_real/manifest.csv`, and writes fetch diagnostics under
`outputs/reports/rbi_official_fetcher/`.
The v1.2.8 fetcher excludes RBI section landing pages, archive/navigation
pages, and low-substantive boilerplate pages from the manifest while retaining
skip diagnostics in `fetch_diagnostics.csv`.

Default official feed placeholders are configured for:

- `https://www.rbi.org.in/pressreleases_rss.xml`
- `https://www.rbi.org.in/Publication_rss.xml`
- `https://www.rbi.org.in/speeches_rss.xml`

Fetch and validate official RBI documents:

```bash
python scripts/fetch_rbi_documents.py --from-date 2020-01-01 --to-date 2026-06-24 --validate-after
```

Target MPC minutes and monetary-policy statements from official RBI sources:

```powershell
python scripts\fetch_rbi_documents.py `
  --from-date 2020-01-01 `
  --to-date 2026-06-24 `
  --max-documents 30 `
  --target-policy-docs `
  --target-document-types mpc_minutes,monetary_policy_statement `
  --min-policy-relevance medium `
  --request-delay-seconds 5 `
  --validate-after `
  --refresh
```

The generic RBI fetcher collects broad policy-relevant official documents,
including speeches, financial-stability documents, monetary-policy statements,
and relevant press releases. The targeted fetcher adds conservative official
RBI archive discovery and prioritizes MPC minutes and monetary-policy
statements, which are the highest-value RBI inputs for the Macro-Stance Index.
The same index-page and irrelevant-press-release filters remain active, skipped
candidates remain auditable in diagnostics, and NLP remains monitoring-only.

The fetcher skips already cached document IDs unless `--refresh` is passed,
supports `--dry-run`, respects `--request-delay-seconds`, and records manual
fallback diagnostics when RBI pages are inaccessible, CAPTCHA/JavaScript
protected, or PDF text extraction is unavailable.

Multi-source monitoring workflow:

```bash
python scripts/fetch_rbi_documents.py --from-date 2020-01-01 --to-date 2026-06-24 --validate-after
python scripts/collect_real_nlp_data.py --config config/nlp_providers.local.yaml --start-date 2026-04-01 --end-date 2026-06-21 --no-cache
python scripts/validate_real_nlp_signal.py --input-records outputs/reports/phase_4a6_real_nlp_validation/deduped_sentiment_records.csv --start-date 2026-04-01 --end-date 2026-06-21
```

If enough valid RBI documents are collected, the source mix can move from
`news_only` to `rbi_and_news`. This remains monitoring commentary only:
allocation, portfolio weights, strategy scoring, evidence gates,
recommendation confidence, and backtests are unchanged.

### Phase 4A.12 — v1.2.8 NLP Monitoring Finalization Pack

v1.2.8 adds official RBI + real news multi-source NLP monitoring. The module
fetches official RBI MPC minutes and monetary policy statements, combines them
with live GDELT/news records, applies timestamp and ex-ante checks, and produces
manager-facing monitoring labels.

The finalized pack is available at:

```text
outputs/reports/phase_4a12_nlp_monitoring_final_pack/
```

It documents data sources, methodology, validation results, limitations,
reproducibility commands, dashboard interpretation, source-mix summaries, RBI
corpus summaries, and daily NLP signal snapshots. NLP remains outside
allocation, strategy scoring, evidence gates, recommendation confidence, and
backtests.

### Phase 4A.13 — v1.2.9 Pain Ratio and NLP Shadow Impact Analysis

v1.2.9 adds Pain Index and Pain Ratio as drawdown-experience metrics alongside
Calmar Ratio. Pain Index is the mean absolute drawdown, and Pain Ratio is
annualized excess return divided by Pain Index. If the Pain Index is zero or
near zero, Pain Ratio is reported as unavailable rather than forcing a value.

The Phase 4A.13 shadow experiment evaluates whether decision-lagged RBI/news
NLP monitoring signals improve Pain Ratio, Pain Index, Calmar, max drawdown,
turnover, and transaction-cost drag. It compares:

- Fixed HERC
- HMM Conservative
- Rule Conservative
- HMM + NLP Confirmation Overlay
- HMM + NLP Early-Warning Overlay

The two NLP variants are explicitly shadow/experimental. NLP remains
monitoring-only in production and is not promoted to allocation unless future
evidence gates justify it. The experiment writes look-ahead diagnostics proving
that NLP signal dates do not exceed the allowed decision-lagged dates.

Run:

```powershell
python scripts\run_nlp_shadow_impact_experiment.py `
  --start-date 2026-04-01 `
  --end-date 2026-06-21 `
  --include-transaction-costs `
  --decision-lag-days 1
```

Artifacts are written to:

```text
outputs/reports/phase_4a13_nlp_shadow_impact/
```

### v1.3.0 — Final Integrated Portfolio Risk Analytics Release

v1.3.0 is the final submission/release pack. It does not add a new modeling
phase and does not change production allocation logic. It consolidates the core
portfolio construction engine, FRM risk analytics, regime-aware adaptive
allocation, Pain Ratio reporting, real RBI + news NLP monitoring, and NLP
shadow-impact analysis into one faculty-facing package.

Final strategy recommendation:

- HERC remains the strategic growth core.
- HMM Conservative remains the primary model-based risk-control overlay.
- Rule Conservative remains the robustness/fallback reference.
- Equal Weight remains the benchmark.

Pain Ratio is now a production reporting metric alongside Calmar Ratio. It
measures return per unit of average drawdown pain, while Calmar measures return
per unit of maximum drawdown.

The latest NLP shadow experiment shows positive shadow impact: the HMM + NLP
Confirmation Overlay improved Pain Ratio, CAGR, Calmar Ratio, turnover, and
transaction-cost drag versus HMM Conservative. It also worsened Pain Index and
maximum drawdown versus HMM Conservative, and the evidence window is short.
Therefore NLP remains monitoring/shadow only and is not production-active.

Final release pack:

```text
outputs/reports/v1_3_0_final_integrated_release/
```

The pack includes the executive summary, final report, technical methodology,
portfolio results, Pain Ratio analysis, NLP shadow-impact analysis, dashboard
guide, reproducibility commands, limitations, viva answers, final metrics
tables, and evidence matrix. It explicitly separates production allocation
logic from monitoring/reporting layers and shadow experiments.

## 5. Dashboard modes

- **Manager View:** a simplified recommendation interface showing the strategic core, optional overlay or fallback, confidence, net trade-offs, Calmar and Pain Ratio, compact market-news/RBI macro-confirmation cards, and NLP status as monitoring-only or positive shadow impact but not allocation-active.
- **Research View:** methodology controls for covariance, rebalancing, regimes, sentiment sources, RBI manifest/scorer/lag settings, adaptive policy, sensitivity, CPCV, attribution, and selection diagnostics.
- **Developer / Debug View:** raw HMM, market-news, RBI documents, sentence scores, alignment, fallback, look-ahead, CPCV, adaptive, gate, and reconciliation diagnostics.

The default manager workflow uses a Balanced objective, Moderate cost assumption, and HMM Walk-Forward Conservative as the risk-control candidate. The recommendation remains evidence-gated and can fall back to Rule-based Conservative.

## 6. Methodology

The fixed benchmark layer supports Equal Weight, Inverse Volatility, HRP, and HERC. Covariance estimation supports sample, Ledoit-Wolf, EWMA, and EWMA plus Ledoit-Wolf methods. Backtests apply weights out of sample, include transaction costs, and retain separate gross and net return series.

Performance reporting includes CAGR, volatility, Sharpe, Sortino, max
drawdown, Calmar Ratio, Pain Index, Pain Ratio, turnover, and transaction-cost
drag. Pain Ratio is reported alongside Calmar as an additional view of the
drawdown experience; it does not replace the existing Calmar logic.

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

Pain Ratio is available in regenerated strategy comparison tables, dashboard
performance tables, replication outputs, and the Phase 4A.13 shadow-impact
report. Existing historical summary values above are retained for continuity
until the full matched primary scenario pack is regenerated with Pain metrics.

See the [final results summary](outputs/final_project_pack/final_results_summary.md) for interpretation and source notes.

## 8. Validation and tests

The project includes unit and integration coverage across allocation, covariance, backtesting, costs, risk analytics, regimes, sentiment ingestion/scoring/alignment, adaptive policies, CPCV, strategy selection, dashboard modes, net labels, and reconciliation.

Final release verification:

```bash
python -m pytest -q
python scripts/final_smoke_test.py
```

The dashboard is also launched headlessly and checked through Streamlit's health endpoint. Exact freeze results are recorded in the [final validation checklist](outputs/final_project_pack/final_validation_checklist.md).

Verified on June 25, 2026: **502 passed, 1 skipped, final smoke test passed, Phase 4A.13 shadow-impact artifacts present, and v1.3.0 final integrated release-pack artifacts present**.

## 9. Reports and artifacts

- [Final project pack index](outputs/final_project_pack/INDEX.md)
- [Phase 4A sentiment-confirmation report](outputs/reports/phase_4a_sentiment_confirmation/report.html)
- [Phase 4A.2 RBI macro-sentiment report](outputs/reports/phase_4a2_rbi_macro_sentiment/report.html)
- [Phase 4A.3 real RBI corpus validation report](outputs/reports/phase_4a3_real_rbi_macro_validation/report.html)
- [Phase 4A.5 API-based ex-ante NLP monitoring report](outputs/reports/phase_4a5_api_sentiment_ingestion/report.html)
- [Phase 4A.6 real NLP signal validation report](outputs/reports/phase_4a6_real_nlp_validation/report.html)
- [Phase 4A.7 NLP corpus intake validation](outputs/reports/nlp_corpus_intake_validation/summary.md)
- [Phase 4A.13 NLP shadow impact report](outputs/reports/phase_4a13_nlp_shadow_impact/report.html)
- [v1.3.0 final integrated release report](outputs/reports/v1_3_0_final_integrated_release/final_report.html)
- [v1.3.0 final integrated release pack](outputs/reports/v1_3_0_final_integrated_release/executive_summary.md)
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
- The default Phase 4A.6 no-live run finds no real provider records because the bundled earnings transcripts are explicit synthetic fixtures; its insufficiency verdict is intentional.
- Phase 4A.7 reports manual action required until reviewed RBI, earnings-call, and news manifests contain valid real records; templates and placeholders are not empirical data.

## 12. Future work

- Add explicit CPCV fold-coverage eligibility or penalties.
- Replicate across broader universes, regions, and longer market histories.
- Strengthen data versioning, model governance, and reproducibility metadata.
- Extend transaction-cost modeling with liquidity-aware market impact.
- Evaluate alternative probabilistic regime models under the same walk-forward safety contract.
- Phase 4B could test sentiment as a gated allocation feature after out-of-sample validation.
- Add governed real RBI source ingestion, publication-time rules, document provenance, and broader historical coverage before any Phase 4B test.
- Evaluate VADER or FinBERT only as optional scorers after establishing a reproducible timestamped corpus and baseline.
