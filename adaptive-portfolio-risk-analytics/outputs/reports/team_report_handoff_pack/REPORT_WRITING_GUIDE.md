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

1. Fixed strategy performance table.
2. Adaptive strategy performance table.
3. Adaptive vs fixed comparison table.
4. CPCV robustness ranking table.
5. Strategy-selection evidence matrix.
6. Pain Ratio comparison table.
7. NLP shadow-impact metrics table.
8. RBI/NLP corpus status table.
9. Dashboard mode summary table.

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

