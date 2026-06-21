# Resume Bullets and Project Descriptions

**v1.0 — Regime-Aware Portfolio Risk Analytics Platform**

## Three short resume bullets

- Built a Python portfolio research platform combining hierarchical allocation, FRM risk analytics, historical backtesting, and a three-mode Streamlit dashboard.
- Developed lagged rule-based and HMM walk-forward regime overlays with defensive allocation, transaction costs, stress testing, and net/gross reconciliation.
- Implemented CPCV-style robustness validation and an evidence-gated selector that classifies HERC as the growth core and adaptive strategies as risk-control overlays.

## Three technical resume bullets

- Engineered an end-to-end platform using hierarchical risk allocation, regime-aware adaptive overlays, HMM walk-forward regime inference, CPCV-style robustness validation, evidence-gated strategy selection, and a Streamlit dashboard.
- Implemented Equal Weight, Inverse Volatility, HRP, and HERC backtests with sample, Ledoit-Wolf, EWMA, and EWMA-shrinkage covariance; added turnover, transaction costs, volatility targeting, VaR/ES, stress, liquidity, concentration, and active-risk diagnostics.
- Built walk-forward-safe regime research with lagged decisions, centralized defensive-return metadata, matched scenario replication, objective-consistent CPCV ranking, strategy-role guardrails, and Manager/Research/Developer dashboard modes.

## Three interview-style project descriptions

### 30-second version

I built a portfolio research platform that compares Equal Weight, Inverse Volatility, HRP, and HERC, then adds rule-based and HMM walk-forward adaptive overlays. It evaluates net performance, drawdown, VaR/ES, stress, costs, and CPCV-style robustness, and converts that evidence into a Streamlit recommendation. The final conclusion is that HERC is the growth core, while HMM Conservative is a drawdown-control overlay.

### 60-second version

The project started as a portfolio-construction and backtesting engine and expanded into an FRM risk layer with tail risk, drawdown, stress, liquidity, concentration, and active-risk diagnostics. I then added lagged rule-based regimes and an expanding-window HMM so adaptive decisions did not use full-sample information. Conservative, Balanced, and Aggressive policies change allocation method, covariance estimator, volatility target, risky cap, and defensive floor. Finally, I added CPCV-style validation, matched replication, evidence gates, investor profiles, and a three-mode Streamlit dashboard. The key finding was role separation: HERC led growth, while conservative adaptive strategies improved downside behavior.

### Technical deep-dive version

The system uses aligned daily return matrices and modular allocators for Equal Weight, Inverse Volatility, HRP, and HERC. Backtests apply weights out of sample, include turnover-based base cost and slippage, and retain gross and net series for reconciliation. Regime features include volatility percentile, drawdown, trend, momentum, and correlation. Rule labels are lagged, while HMM states are estimated through expanding-window walk-forward fits and decision lagging; full-sample HMM is visualization only. Validation combines sensitivity, stress windows, purged and embargoed CPCV-style folds, and matched replication. A PASS/WARN/FAIL/NOT_AVAILABLE gate system then produces profile-aware recommendations while enforcing strategy-role guardrails.

## LinkedIn-style project description

Built **v1.0 — Regime-Aware Portfolio Risk Analytics Platform**, a Python research project combining hierarchical portfolio allocation, FRM risk diagnostics, lagged rule-based regimes, HMM walk-forward inference, adaptive defensive overlays, transaction-cost-aware backtesting, stress testing, matched replication, and CPCV-style robustness validation. The Streamlit interface separates Manager, Research, and Developer views and uses evidence gates to communicate strategy roles and confidence. The final result is deliberately role-based: HERC is the strategic growth core, HMM Conservative is the drawdown-control overlay, Rule-based Conservative is the robustness reference and HMM fallback, and Equal Weight is the benchmark. The platform is intended for historical research and decision support, not live execution.

