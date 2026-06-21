# Architecture Summary

**v1.0 — Regime-Aware Portfolio Risk Analytics Platform**

## Architecture diagram

```text
Data Layer
  → preprocessing
  → return/risk matrix

Strategy Layer
  → EW / IV / HRP / HERC

Risk Analytics Layer
  → Sharpe / Sortino / Calmar / VaR / ES / Drawdown / Stress / Active Risk

Regime Layer
  → Rule-based regimes
  → HMM walk-forward regimes

Adaptive Layer
  → Conservative / Balanced / Aggressive overlays
  → Defensive sleeve handling

Validation Layer
  → Sensitivity
  → Stress testing
  → CPCV-style robustness
  → Replication checks

Selection Layer
  → evidence gates
  → investor profile mapping
  → strategy role classification

Dashboard Layer
  → Manager View
  → Research View
  → Developer View
```

## Data Layer

The data layer retrieves adjusted market prices, validates observations, handles missing values, and builds aligned daily simple-return matrices. It also produces the return and risk inputs consumed by all later layers, so date alignment and data quality are checked before portfolio analysis.

## Strategy Layer

The strategy layer implements Equal Weight, Inverse Volatility, HRP, and HERC as first-class benchmark strategies. It converts historical return and covariance information into portfolio weights while preserving a consistent allocator interface for backtesting and comparison.

## Risk Analytics Layer

The risk layer calculates net performance, volatility, downside efficiency, tail risk, drawdown, stress, concentration, liquidity, and active-risk diagnostics. It supports both portfolio-level decision summaries and detailed FRM-style analysis, while keeping documented distinctions between signed return-quantile VaR/CVaR and positive-loss dashboard VaR/ES.

## Regime Layer

The regime layer creates market-state features and classifies market conditions. Rule-based regimes are explainable and lagged for decisions; HMM walk-forward regimes use expanding historical training and lagging. Full-sample HMM output is restricted to historical visualization.

## Adaptive Layer

The adaptive layer maps a decision regime to an allocation policy. Policies can change the allocator, covariance estimator, volatility target, risky-exposure cap, defensive floor, and rebalance rule. Defensive returns are resolved centrally and carry source and fallback metadata.

## Validation Layer

The validation layer checks whether findings survive parameter changes, stress windows, time-block splits, cost assumptions, defensive sleeves, universes, and date windows. CPCV-style validation reports successful and failed folds separately, and replication tests whether downside improvements persist under matched scenarios.

## Selection Layer

The selection layer translates research evidence into explicit strategy roles. Safety and evidence gates assess net-return basis, HMM validity, data sufficiency, CPCV coverage, adverse folds, costs, turnover, stress evidence, defensive metadata, and replication classification. Profile-aware scoring then recommends a core and optional overlay without overriding role guardrails.

## Dashboard Layer

The Streamlit dashboard presents the same underlying analysis at three levels. Manager View focuses on the decision, Research View exposes methodology and validation controls, and Developer / Debug View exposes raw diagnostics and reconciliation evidence.

## Cross-layer timing and evidence contracts

- Daily returns are aligned by `DatetimeIndex`.
- Weights selected at date `t` apply to returns at `t+1`.
- Net metrics include transaction-cost drag; gross metrics are retained separately.
- Rule-based decisions are lagged by at least one period.
- HMM recommendations require walk-forward decisions with lagging.
- The selected objective propagates into sensitivity and CPCV ranking.
- Adaptive strategies are classified as overlays or references unless repeated net evidence supports a different role.

