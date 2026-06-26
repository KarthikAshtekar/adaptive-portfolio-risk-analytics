# Report Writing Guide

Use this guide to turn the repository into the final academic/report document.

## Suggested final report chapter structure

1. Executive summary
2. Problem statement and financial motivation
3. Data sources and preprocessing
4. Fixed portfolio construction
5. Covariance estimation and hierarchical clustering
6. HRP and HERC methodology
7. Backtesting, rebalancing, and transaction costs
8. FRM risk analytics
9. Regime detection
10. Adaptive allocation
11. CPCV and robustness validation
12. Replication and defensive-return consistency
13. Strategy-selection engine and evidence gates
14. Pain Index and Pain Ratio
15. RBI + GDELT/news NLP monitoring
16. NLP shadow-impact experiment
17. Dashboard implementation
18. Final recommendation
19. Limitations and future work
20. Viva appendix

## What to write in each chapter

- Executive summary: state the final role-based recommendation.
- Problem statement: motivate regime instability, drawdown risk, and cost-aware allocation.
- Data: describe market returns, RBI documents, news records, and timestamp governance.
- Fixed portfolios: compare Equal Weight, Inverse Volatility, HRP, and HERC.
- Covariance/clustering: explain sample, Ledoit-Wolf, EWMA, HRP, HERC.
- Backtesting: explain out-of-sample weights, rebalance rules, turnover, and transaction costs.
- Risk analytics: describe VaR, ES, stress testing, liquidity, active risk, alpha/beta, Calmar, Pain Ratio.
- Regimes: explain rule-based regimes and HMM walk-forward.
- Adaptive allocation: explain regime-dependent exposure control.
- CPCV: explain why robustness validation matters and why results are interpreted conservatively.
- Selection: explain evidence gates and role-based recommendation.
- Pain Ratio: explain average drawdown experience.
- NLP: explain monitoring-only status and ex-ante validation.
- Shadow impact: explain positive shadow result without production promotion.
- Dashboard: explain Manager, Research, Developer modes.

## Project artifacts to use

- `README.md`
- `outputs/final_project_pack/`
- `outputs/reports/post_p0_adaptive_validation/summary.md`
- `outputs/reports/phase_3f_strategy_selection/summary.md`
- `outputs/reports/phase_4a12_nlp_monitoring_final_pack/summary.md`
- `outputs/reports/phase_4a13_nlp_shadow_impact/summary.md`
- `outputs/reports/v1_3_0_final_integrated_release/`
- `outputs/reports/team_report_handoff_pack/README_FOR_TEAM.md`
- `outputs/reports/team_report_handoff_pack/README_TECHNICAL_APPENDIX.md`

## Tables to include

Use the aggregated files under:

- `outputs/reports/team_report_handoff_pack/ready_to_use_tables/`
- `outputs/reports/team_report_handoff_pack/ready_to_use_figures/`
- `outputs/reports/team_report_handoff_pack/TABLES_FIGURES_RESULTS_INSIGHTS.md`

Recommended tables:

1. Fixed/adaptive strategy performance table: `ready_to_use_tables/fixed_vs_adaptive_metrics_comparison.csv`.
2. Adaptive vs fixed summary table: `ready_to_use_tables/adaptive_vs_fixed_summary.csv`.
3. CPCV robustness ranking table: `ready_to_use_tables/cpcv_robustness_summary.csv`.
4. Strategy-selection evidence matrix: `ready_to_use_tables/evidence_matrix.csv`.
5. Final strategy ranking table: `ready_to_use_tables/strategy_ranking_table.csv`.
6. Pain Ratio comparison table: `ready_to_use_tables/pain_ratio_comparison.csv`.
7. NLP shadow-impact metrics table: `ready_to_use_tables/nlp_shadow_impact_table.csv`.
8. RBI/NLP corpus status table: `ready_to_use_tables/rbi_corpus_summary.csv`.
9. NLP source-mix table: `ready_to_use_tables/source_mix_summary.csv`.
10. Look-ahead diagnostics table: `ready_to_use_tables/phase_4a13_lookahead_diagnostics.csv`.
11. Selection gate table: `ready_to_use_tables/selection_gate_results.csv`.
12. Stress-period comparison table: `ready_to_use_tables/stress_period_comparison.csv`.

Recommended figures:

1. Calmar ranking: `ready_to_use_figures/calmar_ranking.png`.
2. Stress drawdown comparison: `ready_to_use_figures/stress_drawdown_comparison.png`.
3. Replication win rates: `ready_to_use_figures/replication_win_rates.png`.
4. Cost sensitivity heatmap: `ready_to_use_figures/cost_sensitivity_heatmap.png`.
5. Profile candidate scores: `ready_to_use_figures/profile_candidate_scores.png`.
6. NLP coverage threshold attainment: `ready_to_use_figures/nlp_coverage_threshold_attainment.png`.

## Allowed claims

- The platform supports adaptive regime-aware portfolio analytics.
- HERC remains the strategic growth core.
- HMM Conservative is useful as a model-based risk-control overlay.
- Rule Conservative remains a robustness/fallback reference.
- Pain Ratio was added to capture average drawdown experience.
- Real RBI + news NLP supports monitoring and shadow confirmation.
- NLP shows positive shadow impact versus HMM Conservative.
- NLP remains monitoring-only and not production-active.

## Claims to avoid

- NLP improves the production strategy.
- NLP predicts market crashes.
- NLP is allocation-ready.
- Adaptive allocation always beats fixed HERC.
- HMM full-sample labels are trading-safe.
- The system is a live-trading engine.
- Backtest results guarantee future performance.

## Recommended wording for final conclusion

The final platform supports regime-aware portfolio analytics and evidence-gated strategy selection. HERC remains the growth core because it provides stronger long-run growth. HMM Conservative remains the primary model-based risk-control overlay because it materially improves drawdown behavior. Rule Conservative remains the robustness/fallback reference because it is transparent and robust in short windows. Pain Ratio adds a useful average-drawdown view alongside Calmar. RBI + news NLP is useful for monitoring and shows positive shadow impact, but it is not production-active because the evidence window is short and drawdown results are mixed.
