# Technical Appendix for Report Writers

This appendix maps the implemented repository to report sections, tables, tests, and reproducibility commands.

## Module map

| Area | Paths | Purpose |
| --- | --- | --- |
| Data pipeline | `src/data_pipeline/` | ingest, preprocess, feature engineering, defensive asset helpers |
| Covariance | `src/covariance/` | sample, Ledoit-Wolf, EWMA, factory routing |
| Clustering | `src/clustering/` | correlation distances, dendrograms, HRP/HERC clustering logic |
| Optimization | `src/optimization/` | allocator interfaces and fixed allocation methods |
| Backtesting | `src/backtesting/` | rolling backtests, rebalancing, turnover, costs, volatility targeting |
| Analytics | `src/analytics/` | performance, drawdown, VaR/ES, stress, liquidity, active risk, risk contribution |
| Regime | `src/regime/` | features, rule-based labels, HMM walk-forward, regime analytics |
| Adaptive | `src/adaptive/` | regime controller, policies, adaptive backtest, defensive returns |
| Experiments | `src/experiments/` | fixed/adaptive experiments, sensitivity, replication |
| Validation | `src/validation/` | CPCV splits, purge/embargo, robustness ranking |
| Selection | `src/selection/` | evidence gates, profile config, scoring, final recommendation |
| Sentiment/NLP | `src/sentiment/` | ingestion, providers, scoring, RBI fetcher, NLP shadow overlays |
| Dashboard | `src/dashboard/` | Manager, Research, Developer Streamlit views |

## Script map

| Script | Purpose |
| --- | --- |
| `scripts/final_smoke_test.py` | verifies imports and required report artifacts |
| `scripts/fetch_rbi_documents.py` | official RBI document fetcher and manifest updater |
| `scripts/check_rbi_corpus_status.py` | RBI corpus sufficiency/status check |
| `scripts/collect_real_nlp_data.py` | provider-based NLP data collection |
| `scripts/validate_real_nlp_signal.py` | real NLP signal validation and reporting |
| `scripts/run_nlp_shadow_impact_experiment.py` | Phase 4A.13 NLP shadow-impact experiment |
| `scripts/run_rbi_empirical_validation.py` | RBI empirical validation report runner |
| `scripts/validate_nlp_corpus_intake.py` | validates governed NLP corpus intake |

## Report artifact map

| Artifact | Use in final report |
| --- | --- |
| `outputs/final_project_pack/` | final architecture, methodology, summary, validation, viva prep |
| `outputs/reports/post_p0_adaptive_validation/` | adaptive vs fixed results, CPCV, stress findings |
| `outputs/reports/phase_3e_replication/` | replication and defensive-return consistency |
| `outputs/reports/phase_3f_strategy_selection/` | selection gates, final strategy recommendation |
| `outputs/reports/phase_4a6_real_nlp_validation/` | real NLP validation diagnostics |
| `outputs/reports/phase_4a12_nlp_monitoring_final_pack/` | RBI + news monitoring final pack |
| `outputs/reports/phase_4a13_nlp_shadow_impact/` | Pain Ratio and NLP shadow-impact metrics |
| `outputs/reports/v1_3_0_final_integrated_release/` | integrated final release pack |
| `outputs/reports/team_report_handoff_pack/` | this handoff pack |

## Important tests

- `tests/test_cpcv_validation.py`
- `tests/test_adaptive_cpcv_integration.py`
- `tests/test_adaptive_controller.py`
- `tests/test_strategy_selection_engine.py`
- `tests/test_pain_ratio_metrics.py`
- `tests/test_nlp_shadow_impact_experiment.py`
- `tests/test_rbi_targeted_policy_fetch.py`
- `tests/test_active_risk_metrics.py`
- `tests/test_var_es.py`
- `tests/test_stress_testing_frm.py`
- `tests/test_liquidity_diagnostics.py`
- `tests/test_dashboard_modes.py`

## Reproducibility commands

```powershell
python -m pytest -q
python scripts\final_smoke_test.py
```

RBI/NLP workflow:

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

python scripts\collect_real_nlp_data.py `
  --config config\nlp_providers.local.yaml `
  --start-date 2026-04-01 `
  --end-date 2026-06-21 `
  --no-cache

python scripts\validate_real_nlp_signal.py `
  --input-records outputs\reports\phase_4a6_real_nlp_validation\deduped_sentiment_records.csv `
  --start-date 2026-04-01 `
  --end-date 2026-06-21

python scripts\run_nlp_shadow_impact_experiment.py `
  --start-date 2026-04-01 `
  --end-date 2026-06-21 `
  --include-transaction-costs `
  --decision-lag-days 1
```

## Metrics definitions

| Metric | Definition / interpretation |
| --- | --- |
| CAGR | annualized compounded return |
| Volatility | annualized return standard deviation |
| Sharpe | annualized excess return per unit volatility |
| Sortino | annualized excess return per downside deviation |
| Max Drawdown | worst portfolio value decline from running peak |
| Calmar Ratio | CAGR divided by absolute max drawdown |
| Pain Index | average absolute drawdown |
| Pain Ratio | annualized excess return divided by Pain Index |
| VaR | return-tail quantile or positive loss depending on API |
| ES/CVaR | average tail loss/return beyond VaR threshold |
| Tracking Error | annualized active return volatility |
| Information Ratio | active return divided by tracking error |
| Beta | CAPM sensitivity to benchmark |
| Jensen's Alpha | regression intercept annualized from daily alpha |

## Strategy definitions

- Equal Weight: equal capital allocation baseline.
- Inverse Volatility: lower-volatility assets receive higher weights.
- HRP: hierarchical risk parity.
- HERC: hierarchical equal risk contribution; final strategic growth core.
- Rule Conservative: explainable regime-aware fallback/reference.
- HMM Conservative: model-based regime-aware risk-control overlay.
- HMM + NLP Confirmation Overlay: shadow-only NLP overlay, not production-active.

## Validation design

Validation combines unit tests, integration tests, smoke tests, sensitivity experiments, stress windows, replication harnesses, and CPCV-style robustness. CPCV uses ordered time blocks, test-block combinations, purge, embargo, fold reruns, summary statistics, stability, and robustness ranking.

## Dashboard guide

- Manager View: strategy recommendation and compact tradeoffs.
- Research View: methodology and diagnostics.
- Developer View: raw audit tables, look-ahead checks, source-mix diagnostics, and gate traces.

## Known limitations

- Historical data and external providers can revise or fail.
- CPCV successful-fold coverage is limited for adaptive strategies.
- HMM fitting is sample-sensitive.
- Defensive-return assumptions affect adaptive outcomes.
- NLP evidence is short-window and monitoring-only.
- NLP production allocation is not approved.

