# README for Final Report Team

This handoff is a repo-grounded guide for teammates writing the final academic/project report. It is based on the implemented repository structure, source modules, scripts, tests, and report artifacts in `adaptive-portfolio-risk-analytics`.

Do not describe this project as a live-trading system. It is a historical portfolio research, risk analytics, dashboard, and decision-support platform.

## 1. Project objective

The project builds an evidence-gated portfolio risk analytics platform that compares fixed allocation strategies with regime-aware adaptive overlays. It evaluates performance net of transaction costs, adds FRM diagnostics, tests robustness with CPCV-style validation, incorporates Pain Ratio as a drawdown-experience metric, and uses real RBI + GDELT/news NLP only as monitoring and shadow-confirmation evidence.

## 2. Business/financial motivation

The business question is whether a strong fixed portfolio should be replaced by a regime-aware strategy, or whether regime-aware strategies are better used as risk-control overlays. The final answer is role-based:

- HERC is the strategic growth core.
- HMM Conservative is the primary model-based risk-control overlay.
- Rule Conservative is the robustness/fallback reference.
- NLP remains monitoring and shadow-confirmation only.

## 3. Final system architecture

Implemented layers:

```text
Data Layer → Portfolio Strategy Layer → Backtesting Layer → FRM Risk Analytics
          → Regime Detection → Adaptive Allocation → Robustness Validation
          → Strategy Selection → Sentiment/NLP Monitoring → Dashboard/Reports
```

Main code locations:

- `src/data_pipeline/`
- `src/covariance/`
- `src/clustering/`
- `src/optimization/`
- `src/backtesting/`
- `src/analytics/`
- `src/regime/`
- `src/adaptive/`
- `src/experiments/`
- `src/validation/`
- `src/selection/`
- `src/sentiment/`
- `src/dashboard/`

## 4. Data sources

Portfolio data are market price/return data processed by the data pipeline and dashboard workflows. NLP data are from governed local/official RBI document ingestion and GDELT/news records. RBI and news records are timestamped, validated, deduplicated, assigned to market dates, and decision-lagged before being used for monitoring or shadow experiments.

Key paths:

- `data/sentiment/rbi_real/manifest.csv`
- `data/sentiment/rbi_real/raw/`
- `outputs/reports/phase_4a6_real_nlp_validation/`
- `outputs/reports/phase_4a12_nlp_monitoring_final_pack/`

## 5. Data pipeline

The data layer prepares aligned return matrices and handles missing/invalid observations before portfolio construction. Important modules include:

- `src/data_pipeline/ingest.py`
- `src/data_pipeline/preprocess.py`
- `src/data_pipeline/feature_engineering.py`
- `src/data_pipeline/defensive_assets.py`

For report writing, describe this as the source of clean, aligned daily return inputs for fixed and adaptive strategies.

## 6. Portfolio construction methods

Implemented fixed strategies:

- Equal Weight
- Inverse Volatility
- HRP
- HERC

Key modules:

- `src/benchmarks/benchmark_factory.py`
- `src/benchmarks/strategy_comparison.py`
- `src/optimization/equal_weight.py`
- `src/optimization/inverse_volatility.py`
- `src/optimization/hrp_allocator.py`
- `src/clustering/herc_allocator.py`

## 7. Covariance estimators

Implemented covariance options:

- sample covariance
- Ledoit-Wolf covariance
- EWMA covariance
- EWMA + Ledoit-Wolf

Key modules:

- `src/covariance/sample_covariance.py`
- `src/covariance/ledoit_wolf.py`
- `src/covariance/ewma_covariance.py`
- `src/covariance/covariance_factory.py`

## 8. Hierarchical clustering

Hierarchical clustering is used to derive portfolio structure from asset relationships. Distance and dendrogram utilities are implemented in:

- `src/clustering/distance_metrics.py`
- `src/clustering/hierarchical.py`
- `src/clustering/dendrograms.py`

## 9. HRP and HERC

HRP and HERC are hierarchical risk allocation methods. HRP allocates using hierarchical clustering and recursive bisection. HERC extends the approach toward equal risk contribution across clusters.

Key modules:

- `src/clustering/hrp.py`
- `src/optimization/hrp_allocator.py`
- `src/clustering/herc.py`
- `src/clustering/herc_allocator.py`

Final interpretation: HERC remains the strategic growth core.

## 10. Backtesting engine

The backtesting engine applies weights out of sample, tracks portfolio value, retains gross and net return series, and applies transaction costs. Important modules:

- `src/backtesting/rolling_backtester.py`
- `src/backtesting/base.py`
- `src/backtesting/backtest_diagnostics.py`
- `src/backtesting/volatility_targeting.py`

Tests:

- `tests/test_backtesting.py`
- `tests/test_backtest_diagnostics.py`
- `tests/test_volatility_targeting.py`

## 11. Transaction-cost modeling

Transaction costs are included in fixed/adaptive comparisons where applicable. Costs include base transaction cost and slippage assumptions. Key modules:

- `src/backtesting/transaction_costs.py`
- `src/backtesting/turnover.py`

Report language: results are net-of-cost where the corresponding backtest path includes transaction costs.

## 12. Rebalancing logic

The project supports calendar and threshold-style rebalancing. Rebalance decisions, turnover, number of rebalances, and transaction-cost drag are surfaced in outputs and dashboard views.

Key module:

- `src/backtesting/rebalance_rules.py`

## 13. Defensive assets / defensive policy design

Adaptive strategies use defensive sleeves/floors and risky caps by regime. The repo includes defensive-return handling and consistency checks.

Key modules:

- `src/adaptive/defensive.py`
- `src/data_pipeline/defensive_assets.py`
- `src/adaptive/policies.py`

Relevant artifact:

- `outputs/reports/phase_3e_replication/defensive_return_consistency.md`

## 14. Regime detection

Regime detection produces market-state labels used by adaptive allocation. Implemented regime methods:

- rule-based regimes;
- HMM walk-forward regimes;
- full-sample HMM for historical visualization only.

Key modules:

- `src/regime/features.py`
- `src/regime/rule_based.py`
- `src/regime/hmm_regime.py`
- `src/regime/analytics.py`

## 15. Rule-based regime detection

Rule-based regimes use transparent thresholds on rolling volatility, drawdown, trend, momentum, return shock, and correlation. Labels include Calm, Normal, Stress, Crisis, and Unknown. Observed labels are lagged before decision use.

Tests:

- `tests/test_rule_based_regime.py`
- `tests/test_regime_features.py`
- `tests/test_regime_analytics.py`

## 16. HMM regime detection

HMM walk-forward fits on expanding prior data and produces lagged decision regimes. Full-sample HMM is excluded from trading-safe allocation claims.

Key tests:

- `tests/test_hmm_regime.py`
- `tests/test_adaptive_backtest.py`

Report boundary: never claim full-sample HMM was used for production/trading-safe decisions.

## 17. Adaptive allocation controller

The adaptive controller maps regimes to policies: allocator choice, covariance method, target volatility, rebalance parameters, defensive floor, and risky cap.

Key modules:

- `src/adaptive/controller.py`
- `src/adaptive/policies.py`
- `src/adaptive/backtest.py`

Important test:

- `tests/test_adaptive_controller.py`

## 18. CPCV / combinatorial purged cross-validation style robustness

CPCV-style validation is implemented to reduce overfitting risk by evaluating strategy behavior across multiple time-block train/test partitions. It applies purge and embargo logic around test periods and summarizes fold behavior with median, worst fold, stability score, and robustness score.

Key modules:

- `src/validation/cpcv.py`
- `src/validation/robustness.py`

Key tests:

- `tests/test_cpcv_validation.py`
- `tests/test_adaptive_cpcv_integration.py`
- `tests/test_robustness.py`

Important artifacts:

- `outputs/reports/post_p0_adaptive_validation/cpcv_summary.csv`
- `outputs/reports/post_p0_adaptive_validation/summary.md`

Interpretation: CPCV supports conservative confidence. It does not guarantee future performance, and the current adaptive CPCV results have limited successful-fold coverage.

## 19. Adaptive strategy experiments

Adaptive experiments compare rule-based and HMM walk-forward regimes under Conservative, Balanced, and Aggressive policy presets.

Key modules:

- `src/experiments/adaptive.py`
- `src/experiments/adaptive_evaluation.py`
- `src/experiments/runner.py`
- `src/experiments/sensitivity.py`

Relevant reports:

- `outputs/reports/post_p0_adaptive_validation/report.html`
- `outputs/reports/adaptive_strategy_readiness/report.html`

## 20. Defensive-return consistency checks

Defensive-return conventions are tracked because adaptive outcomes depend on the defensive sleeve. The project centralizes and documents the convention used by experiments and CPCV.

Artifacts:

- `outputs/reports/phase_3e_replication/defensive_return_consistency.md`
- `outputs/reports/phase_3e_replication/summary.md`

## 21. Replication harness

The replication harness reruns matched scenarios, cost/sleeve sensitivities, and policy-tuning checks.

Key module:

- `src/experiments/replication.py`

Artifacts:

- `outputs/reports/phase_3e_replication/replication_results.csv`
- `outputs/reports/phase_3e_replication/policy_tuning_results.csv`
- `outputs/reports/phase_3e_replication/report.html`

## 22. Strategy-selection engine and evidence gates

The selection engine converts metrics and robustness evidence into role-based recommendations. It uses evidence gates and investor-profile mappings.

Key modules:

- `src/selection/selector.py`
- `src/selection/gates.py`
- `src/selection/scoring.py`
- `src/selection/config.py`
- `src/selection/playbook.py`

Key tests:

- `tests/test_strategy_selection_engine.py`
- `tests/test_selection_gates.py`
- `tests/test_strategy_playbook.py`

Strategy selection artifact:

- `outputs/reports/phase_3f_strategy_selection/summary.md`

## 23. FRM risk analytics layer

The FRM analytics layer includes return, volatility, tail risk, drawdown, stress testing, liquidity, risk contribution, active risk, and concentration diagnostics.

Key modules:

- `src/analytics/performance_metrics.py`
- `src/analytics/risk_metrics.py`
- `src/analytics/var_es.py`
- `src/analytics/stress_testing.py`
- `src/analytics/liquidity_diagnostics.py`
- `src/analytics/active_risk_metrics.py`
- `src/analytics/risk_contribution.py`

## 24. VaR and Expected Shortfall

VaR and ES/CVaR are implemented in `src/analytics/var_es.py` and tested in `tests/test_var_es.py`. Note the sign convention: experiment VaR/CVaR uses signed return-tail values, while dashboard historical VaR/ES can show positive-loss values.

## 25. Stress testing

Stress testing evaluates portfolio behavior under historical and hypothetical stress windows. Key module and test:

- `src/analytics/stress_testing.py`
- `tests/test_stress_testing_frm.py`

Relevant artifact:

- `outputs/reports/post_p0_adaptive_validation/stress_period_comparison.csv`

## 26. Liquidity diagnostics

Liquidity diagnostics use price, volume, average daily traded value, weights, and participation-rate logic.

Key module and test:

- `src/analytics/liquidity_diagnostics.py`
- `tests/test_liquidity_diagnostics.py`

## 27. Active risk / tracking error / information ratio

Active risk metrics are implemented in `src/analytics/active_risk_metrics.py`. They include tracking error, information ratio, hit ratio, benchmark-relative return, drawdown duration, and concentration diagnostics.

Key test:

- `tests/test_active_risk_metrics.py`

## 28. CAPM alpha/beta diagnostics

CAPM-style diagnostics include beta and Jensen's alpha in `src/analytics/active_risk_metrics.py`.

Tests verify:

- known beta behavior;
- zero benchmark variance handling;
- Jensen's alpha estimation.

See `tests/test_active_risk_metrics.py`.

## 29. Pain Index and Pain Ratio

Pain metrics were added in Phase 4A.13.

Definitions:

```text
Drawdown_t = PortfolioValue_t / RunningPeak_t - 1
Pain Index = mean(abs(Drawdown_t))
Pain Ratio = Annualized Excess Return / Pain Index
```

Key modules:

- `src/analytics/risk_metrics.py`
- `src/analytics/performance_metrics.py`

Key test:

- `tests/test_pain_ratio_metrics.py`

Interpretation: Calmar measures return per unit of maximum drawdown; Pain Ratio measures return per unit of average drawdown pain.

## 30. Real RBI + GDELT/news NLP monitoring

Real RBI + news monitoring is implemented as a monitoring layer. It does not affect allocation, strategy scoring, evidence gates, recommendation confidence, backtests, or portfolio weights.

Key modules:

- `src/sentiment/providers/rbi_provider.py`
- `src/sentiment/providers/gdelt_provider.py`
- `src/sentiment/api_ingestion.py`
- `src/sentiment/composite_index.py`
- `src/sentiment/source_quality.py`
- `src/sentiment/ex_ante_filters.py`

Important reports:

- `outputs/reports/phase_4a6_real_nlp_validation/`
- `outputs/reports/phase_4a8_multisource_nlp_monitoring/`
- `outputs/reports/phase_4a12_nlp_monitoring_final_pack/`

## 31. Official RBI targeted policy-document fetcher

The RBI fetcher populates the governed real-RBI corpus from official RBI sources and writes diagnostics.

Scripts/modules:

- `scripts/fetch_rbi_documents.py`
- `src/sentiment/rbi_official_fetcher.py`

Key tests:

- `tests/test_rbi_targeted_policy_fetch.py`
- `tests/test_rbi_official_fetcher.py`
- `tests/test_rbi_exclude_keyword_filter.py`
- `tests/test_fetch_rbi_documents_cli.py`

Fetcher artifacts:

- `outputs/reports/rbi_official_fetcher/fetch_summary.md`
- `outputs/reports/rbi_official_fetcher/fetch_diagnostics.csv`
- `outputs/reports/rbi_official_fetcher/downloaded_documents.csv`

## 32. NLP source mix and ex-ante validation

NLP source mix labels include `none`, `news_only`, `rbi_only`, and `rbi_and_news`. Ex-ante checks validate timestamps, publication lags, source quality, and possible reaction-data warnings.

Latest monitoring corpus status supplied for final reporting:

- Valid RBI documents: 34
- Policy core documents: 26
- MPC minutes: 13
- Monetary policy statements: 13
- Governor speeches: 6
- Financial stability reports: 1
- Press releases: 1
- Source mix: `{"none": 1, "rbi_and_news": 54, "rbi_only": 3}`
- Verdict: B. Useful for monitoring only
- Allocation impact: None

## 33. NLP shadow-impact experiment

The NLP shadow experiment compares HERC, HMM Conservative, Rule Conservative, HMM + NLP Confirmation Overlay, and HMM + NLP Early-Warning Overlay.

Script/module:

- `scripts/run_nlp_shadow_impact_experiment.py`
- `src/sentiment/nlp_shadow_overlay.py`

Key test:

- `tests/test_nlp_shadow_impact_experiment.py`
- `tests/test_nlp_shadow_overlay.py`

Latest result:

- Look-ahead diagnostics passed: 56 / 56
- Production allocation active: False
- NLP Confirmation Overlay improved Pain Ratio versus HMM Conservative.
- NLP Confirmation Overlay worsened Pain Index and maximum drawdown versus HMM Conservative.

## 34. Dashboard modes: Manager, Research, Developer

Dashboard code:

- `src/dashboard/app.py`
- `src/dashboard/modes.py`
- `src/dashboard/plots.py`
- `src/dashboard/components/`

Modes:

- Manager View: simplified recommendation and tradeoff display.
- Research View: methodology, comparison, CPCV, regime, risk, and NLP diagnostics.
- Developer View: raw diagnostics, look-ahead checks, source-mix tables, gate traces, and artifact paths.

## 35. Final strategy recommendation

Final recommendation:

- HERC = strategic growth core.
- HMM Conservative = primary model-based risk-control overlay.
- Rule Conservative = robustness/fallback reference.
- Pain Ratio = production reporting metric.
- RBI + news NLP = monitoring and shadow-confirmation layer.
- NLP Confirmation Overlay = positive shadow impact but not production-active.
- Production NLP allocation = not approved.

## 36. Final results and interpretation

Earlier full-period strategy results:

| Strategy | CAGR | Max Drawdown | Calmar Ratio | Interpretation |
| --- | ---: | ---: | ---: | --- |
| HMM Conservative | 10.02% | -8.38% | 1.195 | Stronger drawdown/risk-control overlay |
| Fixed HERC | 15.01% | -18.91% | 0.794 | Stronger growth core |

Final NLP/Pain shadow metrics:

| Strategy | CAGR | Pain Index | Pain Ratio | Calmar | Max DD | Turnover | Cost drag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed HERC | 0.71% | 3.28% | -0.39 | 0.09 | -7.84% | 0.54 | 0.12% |
| HMM Conservative | -1.25% | 1.52% | -2.14 | -0.56 | -2.23% | 1.04 | 0.20% |
| Rule Conservative | 7.70% | 0.32% | 17.80 | 7.34 | -1.05% | 1.49 | 0.27% |
| HMM + NLP Confirmation Overlay | 11.42% | 1.66% | 5.69 | 2.54 | -4.50% | 0.43 | 0.09% |
| HMM + NLP Early-Warning Overlay | -1.25% | 1.52% | -2.14 | -0.56 | -2.23% | 1.04 | 0.20% |

Interpretation: HERC has stronger long-run growth. HMM Conservative improves risk control. Rule Conservative remains robust in short/limited folds. NLP improves Pain Ratio in a shadow test, but it is not production-ready.

## 37. Limitations

- Historical research output, not live trading.
- Yahoo/market data can revise.
- CPCV has limited successful-fold coverage for adaptive models.
- HMM fitting is probabilistic and sample-sensitive.
- Defensive-sleeve assumptions affect adaptive results.
- Transaction-cost model is simplified and not full market impact.
- Pain Ratio is window-dependent.
- NLP evidence window is short.
- NLP improves one shadow metric set but worsens Pain Index and max drawdown versus HMM.
- NLP production allocation is not approved.

## 38. Future work

- Add CPCV fold-coverage penalties or eligibility thresholds.
- Replicate across broader universes and longer histories.
- Strengthen defensive-return governance.
- Add liquidity-aware transaction-cost modeling.
- Extend real RBI/news history before any NLP allocation test.
- Define explicit future gates for NLP promotion.

## 39. Suggested final report structure

1. Introduction and motivation
2. Data and preprocessing
3. Fixed portfolio construction
4. FRM risk analytics
5. Regime detection
6. Adaptive allocation
7. Robustness validation / CPCV
8. Strategy selection and evidence gates
9. Pain Ratio and drawdown experience
10. Real RBI + news NLP monitoring
11. NLP shadow-impact experiment
12. Dashboard and implementation
13. Final recommendation
14. Limitations and future work

## 40. Viva preparation questions

- Why is HERC the growth core?
- Why is HMM Conservative an overlay instead of a replacement?
- How does Rule Conservative differ from HMM Conservative?
- What does CPCV test?
- Why is successful-fold coverage important?
- How are transaction costs included?
- What is the difference between Calmar and Pain Ratio?
- What prevents look-ahead bias?
- Why is NLP not production-active?
- What evidence would be required to promote NLP?

## Adaptive Allocation and Market-Regime Responsiveness

Fixed strategies can underperform across regimes because the same risk exposures are held during calm, stress, and crisis periods. This system detects regime changes using lagged rule-based features and HMM walk-forward inference. HMM Conservative and Rule Conservative adjust exposure through regime-dependent target volatility, defensive floors, risky caps, allocator choice, covariance method, and rebalance behavior.

Adaptive allocation differs from static HERC because HERC keeps a fixed strategic risk-allocation logic, while adaptive strategies reduce or reshape risk exposure when regimes deteriorate. Adaptiveness is evaluated through drawdown, Calmar Ratio, Pain Ratio, turnover, transaction costs, and terminal wealth. The project found that adaptive allocation helps risk control but is not automatically a growth replacement for HERC.

## CPCV and Robustness Validation

CPCV-style validation is used to reduce overfitting risk. It evaluates strategy behavior across multiple train/test market partitions, applies purge and embargo rules, and checks whether adaptive improvements are robust or window-specific. It supports final confidence by showing fold medians, worst folds, stability scores, failed folds, and robustness ranks.

Available artifacts:

- `src/validation/cpcv.py`
- `src/validation/robustness.py`
- `tests/test_cpcv_validation.py`
- `tests/test_adaptive_cpcv_integration.py`
- `outputs/reports/post_p0_adaptive_validation/cpcv_summary.csv`
- `outputs/reports/post_p0_adaptive_validation/summary.md`

CPCV does not guarantee future performance. It was important in deciding that adaptive strategies are useful for risk control but should be interpreted conservatively.

## Verification status

Latest known verification:

- Tests: 502 passed, 1 skipped
- Final smoke test: passed

