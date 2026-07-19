# Project Audit and Finalization Report

**Audit date:** 2026-07-19  
**Scope:** code, tests, dashboard, notebooks, reports, tracked artifacts, imports, paths, and root documentation  
**Research boundary:** historical portfolio analytics and decision support; no live trading or production execution claim

## Executive conclusion

The repository is a substantial portfolio research platform rather than the boilerplate project
described by several old root documents. Core portfolio construction, covariance, realistic
backtesting, risk analytics, lag-safe regimes, adaptive policies, experiments, CPCV-style
robustness, strategy selection, and sentiment monitoring are implemented and tested.

The audit also found real quality gaps:

- stale `src/nlp` and `src/regime_detection` scaffolds duplicated tested canonical packages;
- an unused equal-weight HERC scaffold contradicted the implemented HERC allocator;
- a direct `src.clustering` import could trigger a circular import through the HERC public export;
- one forward-fill test contained unreachable assertions after a fixture `return`;
- root bootstrap reports still described most implemented features as TODOs or claimed production
  readiness;
- the generic feature engineer, Alpha Vantage market-price adapter, dynamic allocator, and
  compatibility CPCV class remain partial/non-functional extension points;
- dashboard orchestration and plotting are materially less covered than core quantitative logic.

## Validation baseline

Before cleanup:

```text
python -m compileall -q src scripts main.py verify_setup.py   PASS
python scripts/final_smoke_test.py                            PASS
python -m pytest -q                                           504 passed, 2 skipped
coverage                                                      65%
python -m ruff check .                                        84 errors (Ruff absent from requirements initially)
python -m ruff format --check .                               170 files required formatting with default settings
```

The two skipped tests are optional HMM branches. The first test attempt hit a two-minute command
timeout; the rerun completed in about 2 minutes 20 seconds.

## Entrypoints

| Entrypoint | Purpose | Validation |
| --- | --- | --- |
| `src/dashboard/app.py` | Main Streamlit application | Import smoke; headless health smoke after finalization |
| `main.py` | Phase 1 Yahoo Finance and fixed-strategy pipeline | Compiled/imported; full run not used because it downloads current market data |
| `verify_setup.py` | Structure and core dependency check | Rewritten against current repository |
| `scripts/final_smoke_test.py` | Release imports and artifact presence | Passed baseline and rerun after finalization |
| `scripts/*.py` | NLP ingestion, validation, monitoring, and shadow studies | CLI/unit coverage varies by script |
| `notebooks/*` | Stage research companions | Inspected for cells/outputs; not all re-executed |

## Capability map

Status definitions:

1. Fully implemented and tested
2. Implemented but weakly tested
3. Partially implemented / scaffolded
4. Mentioned in docs but not implemented
5. Future work

| Feature / module | Evidence found | Status | Notes |
| --- | --- | --- | --- |
| Yahoo market data and inspection | `src/data_pipeline/ingest.py`, `tests/test_data_pipeline.py` | 1 | Adjusted-close fallback and volume retained |
| Centralized preprocessing/data quality | `src/data_pipeline/preprocess.py`, dashboard missingness report, tests | 1 | Downstream modules do not own missing-data policy |
| Generic technical/macro feature engineering | `src/data_pipeline/feature_engineering.py` | 3 | Rolling volatility exists; several methods empty/incomplete |
| Alpha Vantage market-price ingestion | `AlphaVantageProvider` | 3 | Raises `NotImplementedError` |
| Four covariance methods | `CovarianceFactory`, covariance tests | 1 | Sample, Ledoit-Wolf, EWMA, EWMA plus Ledoit-Wolf |
| Gerber covariance | older architecture/methodology only | 4 | Removed unused placeholder estimator file; no factory route |
| Correlation/distance/linkage/dendrogram | `src/clustering`, clustering tests | 1 | Shared by HRP/HERC |
| Equal Weight / Inverse Volatility | optimizer and benchmark tests | 1 | First-class benchmark strategies |
| HRP / HERC | canonical allocators, Phase 2A tests, dashboard factory | 1 | All covariance methods supported |
| Mean-Variance | `mean_variance.py`, optimization tests, `main.py` | 1 | Standalone; not benchmark/dashboard routed |
| Dynamic allocator | `dynamic_allocation.py` | 3 | Public extension point raises `NotImplementedError` |
| Rolling backtest | `rolling_backtester.py`, backtesting tests | 1 | Net/gross series and lagged application contract |
| Calendar/threshold rebalancing | rebalance rules, Phase 2B tests/report | 1 | Target updates separated from drift triggers |
| Turnover/transaction costs/slippage | backtesting modules/tests | 1 | Simplified linear/volatility-adjusted research model |
| Volatility targeting/defensive sleeve | overlay and defensive tests | 1 | Exposure bounds and no-look-ahead tests |
| Risk/performance metrics | analytics modules/tests | 1 | Includes Pain Ratio, VaR/ES, stress and active risk |
| Rule-based regimes | `src/regime/rule_based.py`, tests | 1 | Transparent thresholds and lagging |
| HMM regimes | `src/regime/hmm_regime.py`, tests | 1 | Walk-forward supported; full-sample historical only |
| Adaptive allocation | `src/adaptive`, adaptive tests | 1 | Explicit policy presets and lag-safe backtest |
| Sensitivity experiments | `src/experiments`, tests | 1 | Single selected objective controls ranking |
| Replication/reporting helpers | replication/reporting modules, limited tests | 2 | Core harness tested; export/report branches lighter |
| CPCV-style validation | `src/validation`, CPCV/robustness tests | 1 | Purge/embargo; not full independent-path CPCV |
| Strategy selection/evidence gates | `src/selection`, tests, saved artifacts | 1 | Role-based recommendation, not personalized advice |
| RBI/news sentiment monitoring | `src/sentiment`, provider/coverage tests | 1 | Monitoring only; fixtures excluded from real evidence |
| Optional FinBERT | `finbert_scoring.py`, one focused test | 2 | Local-files-only with deterministic fallback |
| NLP shadow overlay | overlay module/script/tests/artifacts | 1 | Shadow only; no production-active allocation |
| Streamlit dashboard | app/modes/plots/components, source/import tests | 2 | Large app; limited browser/plot coverage |
| Stage notebooks | 13 retained notebooks after empty Stage 7 removal | 3 | Several contain code but no executed cell counts |
| Live trading/broker/order management | no code | 5 | Explicitly outside scope |
| Full nonlinear market impact/capacity | liquidity diagnostics only | 5 | Future research/engineering |

## Saved evidence and result provenance

The matched primary result table in `outputs/final_project_pack/final_results_summary.md` uses:

- Core Diversified preset, 12 assets;
- January 1, 2020 through June 19, 2026;
- initial capital 1,000,000;
- 10 bps base cost plus 5 bps slippage;
- synthetic 4% annual adaptive defensive sleeve;
- net-of-cost metrics.

| Strategy | Net CAGR | Calmar | Max drawdown | Net final value |
| --- | ---: | ---: | ---: | ---: |
| HERC | 15.01% | 0.794 | -18.91% | 2,434,518 |
| HMM Walk-Forward Conservative | 11.84% | 1.521 | -7.78% | 2,037,180 |
| Rule-based Conservative | 10.80% | 1.114 | -9.69% | 1,920,070 |
| Equal Weight | 12.56% | 0.381 | -32.94% | 2,122,050 |

The handoff pack contains another fixed/adaptive table and a short Phase 4A.13 shadow table with
different values. Those are separate snapshots/windows and were not combined into one result
claim.

Current saved CPCV output reports 6/15 successful Rule-based Conservative folds and 3/15 HMM
Conservative folds. This limits confidence even though successful-fold metrics are favorable.

The June 25 Phase 4A.6 artifact reports 34 real RBI documents, 50 real GDELT/news records, and
98.3% decision-label coverage with a monitoring-only verdict. The June 21 Phase 4A.3 artifact is an
earlier synthetic-fallback snapshot. The apparent disagreement is temporal/provenance-related, not
silently reconciled.

## Notebook audit

- 14 notebook paths existed at baseline.
- Stage 7 contained zero cells and was removed as an empty placeholder.
- Stages 1, 3, 4, 5, 6, 8, and 12 had executed code cells at inspection time.
- Stages 2, 9, 10, 11, 13, and 14 retained code but had no execution counts.
- Notebooks were not batch re-executed because several depend on network data, can be expensive,
  and are not the canonical saved result artifacts.

## Safe restructuring

Before:

```text
root/
|-- STAGE_1_REPORT.md ... STAGE_14_REPORT.md
|-- PHASE_*.md and DASHBOARD_UI_REFACTOR_REPORT.md
|-- bootstrap/status/readiness documents
|-- Report.md
|-- src/ + tests/ + notebooks/ + outputs/ + docs/
```

After:

```text
root/
|-- README.md
|-- project_explainer.html
|-- pyproject.toml
|-- src/ + tests/ + notebooks/ + outputs/
`-- docs/
    |-- PROJECT_AUDIT.md
    |-- stage_reports/
    |-- audits/
    |-- archive/
    |-- architecture/
    `-- methodology/
```

Moved:

- `STAGE_1_REPORT.md` through `STAGE_14_REPORT.md` -> `docs/stage_reports/`
- Phase 2A/2B and dashboard refactor audits -> `docs/audits/`
- bootstrap/readiness/status documents -> `docs/archive/initial_scaffold/`
- old `Report.md` -> `docs/archive/stage14_project_report.md`

Removed as clearly stale/duplicate:

- unused `src/covariance/advanced_covariance.py` placeholder;
- unused equal-weight `src/clustering/herc.py` scaffold;
- superseded `src/regime_detection` scaffold;
- superseded `src/nlp` scaffold, empty files, sample duplicate, and overclaiming package summary;
- five assertion-free legacy regime tests tied to the removed scaffold;
- empty Stage 7 notebook.

Preserved intentionally:

- `DynamicAllocationAllocator` and `CPCVBacktester` public compatibility symbols, despite their
  documented non-functional boundary;
- historical reports and artifacts for traceability;
- ignored local raw corpora/caches and generated reports;
- current `src/dashboard/app.py` entrypoint.

## Paths and imports

- Added `src/paths.py` for repository-root and common directory paths.
- Updated config, selection, sentiment corpus/provider configuration, and dashboard constants.
- Replaced machine-specific links in Stage 1/2 reports with relative links.
- Added fresh-interpreter import tests after fixing the clustering/optimization circular import.
- Retained direct-script file-relative bootstraps because scripts must establish the repository
  root before importing `src` when launched from arbitrary working directories.

## Dashboard changes

- Preserved all three modes and the existing entrypoint.
- Added one collapsed, mode-aware “How to use this dashboard” guide.
- Added contextual help for benchmark, objective, costs, defensive sleeve, and volatility
  targeting.
- Added explicit warnings about overfitting, failed folds, full-sample HMM, and monitoring-only NLP.
- No dashboard redesign or portfolio algorithm change was made.

## Final documentation assets

- Root `README.md`: rewritten from current code/tests/artifacts.
- Root `project_explainer.html`: self-contained offline architecture, methods, results, dashboard,
  limitations, and interview-defense explainer.
- Current `docs/architecture/ARCHITECTURE.md`, `docs/methodology/METHODOLOGY.md`, and
  `docs/ROADMAP.md`: rewritten to remove stale scaffold claims.

## Post-cleanup validation

Final command results are recorded after the complete validation pass:

```text
python -m compileall -q src scripts main.py verify_setup.py   PASS
python -m pytest -q                                           503 passed, 2 skipped
coverage                                                      66%
python -m ruff check .                                        PASS
python -m ruff format --check .                               PASS (231 files)
python verify_setup.py                                        PASS (24 paths, 9 imports)
python scripts/final_smoke_test.py                            PASS
python scripts/validate_real_nlp_signal.py --help             PASS
README/HTML relative-link audit                              PASS
relocated-report Git blob audit                             PASS (24 exact, 2 intentional link edits)
Streamlit /_stcore/health                                    HTTP 200, ok
git diff --check                                              PASS (line-ending warnings only)
```

The final suite collected 505 tests. Five assertion-free legacy regime-detection tests were
removed, while two repository-path tests and two fresh-interpreter import-contract tests were
added. The two retained skips are optional HMM branches. The dashboard server was started
headlessly on a temporary local port, health-checked, and stopped explicitly.

## Intentionally not done

- No covariance, HRP/HERC, risk metric, or adaptive-policy algorithm was redesigned.
- No raw/private NLP corpus, cache, or historical report artifact was deleted.
- No network market-data pipeline or heavy experiment grid was rerun.
- No saved historical metric was overwritten.
- No production-readiness or live-trading claim was added.

## Recommended next work

1. Modularize the dashboard and add browser-level tests.
2. Add CPCV fold-coverage eligibility/penalties.
3. Execute/freeze deterministic notebooks and add an execution manifest.
4. Version market/corpus snapshots and experiment assumptions.
5. Expand independent-universe replication and calibrated market-impact research.
