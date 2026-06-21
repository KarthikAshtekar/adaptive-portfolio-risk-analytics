# Presentation Outline

**v1.0 — Regime-Aware Portfolio Risk Analytics Platform**

## Slide 1 — Title and objective

**Main message:** The project connects portfolio construction, FRM risk analytics, regime-aware risk control, robustness validation, and manager-facing decisions.

**Key points:**

- Evidence-gated portfolio research platform
- Historical backtesting with net-of-cost metrics
- Hierarchical core plus adaptive risk-control overlays
- Streamlit dashboard for decision support

**Suggested visual:** One-line architecture ribbon from data to dashboard.

**Speaker notes:** Introduce the central question: should adaptive allocation replace a strong fixed strategy, or should it be used as a risk-control overlay? State that the final answer is role-based.

## Slide 2 — Problem statement

**Main message:** A single full-sample backtest is insufficient for selecting a portfolio strategy.

**Key points:**

- Return alone ignores drawdown and tail risk
- Correlations and volatility change across market regimes
- Turnover and transaction costs can erase apparent gains
- Model complexity creates look-ahead and overfitting risks
- A recommendation needs evidence quality, not only point estimates

**Suggested visual:** Four risk boxes around a central “single backtest” warning: drawdown, regime change, cost, overfitting.

**Speaker notes:** Explain that the project evaluates both performance and the reliability of the evidence. This motivates stress testing, walk-forward regimes, CPCV-style validation, and explicit recommendation gates.

## Slide 3 — System architecture

**Main message:** The system is modular, with evidence flowing from cleaned data to an explainable strategy recommendation.

**Key points:**

- Data and return/risk matrix
- Strategy and FRM analytics
- Regime and adaptive-policy layers
- Validation and evidence gates
- Manager, Research, and Developer interfaces

**Suggested visual:** Layered architecture diagram from `architecture_summary.md`.

**Speaker notes:** Emphasize the separation of concerns. The selection layer does not invent performance; it consumes stored strategy, stress, replication, and CPCV evidence.

## Slide 4 — Portfolio construction methods

**Main message:** The fixed strategy layer provides transparent benchmarks and hierarchical risk allocation.

**Key points:**

- Equal Weight as the baseline
- Inverse Volatility for marginal volatility scaling
- HRP for cluster-based recursive risk allocation
- HERC for cluster-level equal risk contribution
- Multiple covariance estimators improve sensitivity analysis

**Suggested visual:** Comparison table of method, information used, and main advantage.

**Speaker notes:** Explain why HERC and HRP are attractive alternatives to expected-return-sensitive Markowitz optimization. They use hierarchical structure and avoid relying on unstable return forecasts.

## Slide 5 — FRM risk analytics layer

**Main message:** Strategy comparison includes return, downside, tail, stress, liquidity, and active-risk dimensions.

**Key points:**

- Sharpe, Sortino, Calmar, and drawdown duration
- Historical VaR and ES/CVaR
- Historical, hypothetical, and correlation stress
- Turnover and transaction-cost drag
- Concentration, liquidity, beta, tracking error, and information ratio

**Suggested visual:** Risk dashboard mock-up with five metric families.

**Speaker notes:** Clarify the two VaR/ES sign conventions: experiment outputs are signed tail returns, while dashboard historical VaR/ES are positive losses.

## Slide 6 — Regime detection

**Main message:** The platform combines an explainable rule system with a probabilistic HMM, while enforcing trading-safe timing.

**Key points:**

- Features: volatility, drawdown, trend, momentum, correlation
- Rule-based Calm / Normal / Stress / Crisis states
- HMM latent-state inference and financial state mapping
- Observed signals are lagged before portfolio use
- Full-sample HMM is visualization only

**Suggested visual:** Regime timeline with rule-based and HMM bands.

**Speaker notes:** State the key safety rule exactly: trading-safe recommendations use HMM walk-forward decisions with lagging. Full-sample HMM never drives adaptive trading claims.

## Slide 7 — Adaptive allocation overlay

**Main message:** Adaptive policies change risk exposure and implementation choices by regime, not merely asset weights.

**Key points:**

- Conservative, Balanced, and Aggressive presets
- Regime-dependent HERC, HRP, or Equal Weight allocator
- Regime-dependent covariance and volatility target
- Risky caps and defensive floors
- Centralized defensive-sleeve handling

**Suggested visual:** Policy matrix showing Calm, Normal, Stress, and Crisis settings.

**Speaker notes:** Explain that Conservative reduces target volatility, raises defensive floors, and lowers risky caps. The overlay is designed to control drawdown, so lower terminal wealth can be an expected trade-off.

## Slide 8 — Validation framework

**Main message:** Results are challenged through multiple forms of out-of-sample and scenario evidence.

**Key points:**

- Rolling historical backtests
- Parameter sensitivity and stress windows
- Purged and embargoed CPCV-style folds
- Matched replication across costs, sleeves, universes, and dates
- PASS/WARN/FAIL/NOT_AVAILABLE selection gates

**Suggested visual:** Funnel from many configurations to a gated recommendation.

**Speaker notes:** Describe CPCV as pragmatic time-block robustness testing, not complete independent-path CPCV. Mention that adaptive warm-up requirements reduce fold coverage.

## Slide 9 — Key findings

**Main message:** HERC leads growth; conservative adaptive strategies improve downside behavior.

**Key points:**

- HERC: 15.01% net CAGR and 2.43 million final value
- HMM Conservative: -7.78% drawdown and 1.521 Calmar
- Rule Conservative: first in current adaptive CPCV ranking
- Equal Weight: benchmark with -32.94% drawdown
- Confidence: Moderate because adaptive fold coverage is limited

**Suggested visual:** Two-axis trade-off chart: net CAGR versus maximum drawdown, with bubble size for final value.

**Speaker notes:** Avoid declaring one overall winner. HMM Conservative gives up growth for protection. Rule Conservative is the robustness fallback. The evidence supports role separation.

## Slide 10 — Dashboard and selection engine

**Main message:** The same evidence is presented at the level needed by managers, researchers, and developers.

**Key points:**

- Manager View: core, overlay, confidence, trade-offs
- Research View: settings, regimes, sensitivity, CPCV, attribution
- Developer View: folds, HMM internals, gates, reconciliation
- Investor profiles map to explicit scoring priorities
- Safety gates prevent unsupported recommendations

**Suggested visual:** Three-column screenshot layout for the three dashboard modes.

**Speaker notes:** Demonstrate the Balanced profile. Show HERC as core, HMM Conservative as overlay, Rule Conservative as fallback, and the CPCV coverage warning.

## Slide 11 — Limitations

**Main message:** The platform is defensible because its evidence boundaries are explicit.

**Key points:**

- Historical and bounded data sample
- Mutable Yahoo Finance source data
- Limited adaptive CPCV successful-fold coverage
- HMM specification and fit uncertainty
- Simplified transaction-cost and market-impact assumptions

**Suggested visual:** Limitations table with current mitigation and next action.

**Speaker notes:** The most important limitation is coverage-aware robustness. Strong successful folds do not erase failed folds, and the current ranking does not directly penalize missing coverage.

## Slide 12 — Final conclusion and future work

**Main message:** v1.0 is frozen as a role-based portfolio decision-support platform.

**Key points:**

- HERC: strategic growth core
- HMM Conservative: drawdown-control overlay
- Rule Conservative: robustness reference and HMM fallback
- Equal Weight: benchmark
- Next: coverage-aware CPCV, broader replication, governance, and market impact

**Suggested visual:** Final role map with arrows from investor objective to core and overlay.

**Speaker notes:** Close with the precise conclusion: adaptive is useful, but not as a universal replacement for HERC. NLP and macro sentiment remain future research and are not part of v1.0.

