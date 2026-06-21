# Final Project Summary

**v1.0 — Regime-Aware Portfolio Risk Analytics Platform**

An evidence-gated portfolio research platform that combines hierarchical risk allocation, FRM risk diagnostics, regime-aware adaptive overlays, CPCV-style robustness validation, and a simplified manager-facing decision interface.

## Project objective

Build an end-to-end research platform that constructs portfolios, measures their risk, detects changing market conditions, tests adaptive risk-control policies, validates results across time partitions, and converts technical evidence into a manager-readable strategy recommendation.

## Problem statement

Portfolio strategies are often compared through a single full-sample backtest. That approach can hide drawdown risk, transaction-cost drag, regime dependence, and unstable results across time. This project asks whether hierarchical allocation and regime-aware overlays can create a more defensible decision process without claiming that one method is universally best.

## Why this project matters

The project connects portfolio construction with FRM concepts such as market risk, tail risk, stress loss, concentration, liquidity, active risk, and model risk. It also separates growth and risk-control roles: a strategy that protects capital during stress may still be unsuitable as the main growth portfolio if it sacrifices too much terminal wealth.

## Data and universe

The primary validated scenario uses a 12-asset Indian diversified universe spanning banks, technology, diversified industry, consumer, health care, infrastructure, telecom, and gold. Yahoo Finance adjusted prices are cleaned and converted to aligned daily simple returns. The latest primary evaluation covers January 1, 2020 through June 19, 2026, with earlier data used for warm-up where required.

Because the data source is external and mutable, exact rerun values can change. The stored reports and CSV artifacts are the evidence base for v1.0.

## Strategies implemented

- **Equal Weight:** allocates the same weight to each asset and serves as the benchmark.
- **Inverse Volatility:** assigns larger weights to lower-volatility assets.
- **HRP:** uses hierarchical clustering and recursive allocation to diversify risk without inverting the covariance matrix.
- **HERC:** allocates risk across hierarchical clusters and is the validated strategic growth core.
- **Regime-adaptive policies:** Conservative, Balanced, and Aggressive overlays that change allocator, covariance method, volatility target, risky exposure, defensive floor, and rebalance behavior by decision regime.

## Risk analytics implemented

The analytics layer includes CAGR, volatility, Sharpe, Sortino, Calmar, maximum drawdown, drawdown duration, historical VaR and ES, signed return-quantile VaR/CVaR for experiments, stress-period performance, hypothetical and correlation stress, turnover, transaction costs, concentration, liquidity diagnostics, beta, alpha, tracking error, information ratio, and benchmark-relative comparisons.

## Regime detection methods

The explainable rule-based model classifies Calm, Normal, Stress, Crisis, and Unknown states from rolling volatility, drawdown, trend, momentum, correlation, and benchmark features. Observed labels are lagged before they are used in adaptive decisions.

The HMM method estimates latent states from historical features. Full-sample HMM is used only for historical visualization. Trading-safe adaptive analysis uses expanding-window HMM walk-forward inference with a decision lag.

## Adaptive allocation logic

Each regime maps to a policy specifying an allocator, covariance estimator, target volatility, rebalance rule, defensive floor, and risky-exposure cap. Conservative policies lower target volatility, raise defensive floors, and reduce risky caps relative to Balanced. The applied portfolio weight is chosen before the subsequent return, preserving the repository's out-of-sample timing contract.

## CPCV-style validation

The validation layer divides time into ordered blocks, forms test-block combinations, removes test observations from training, applies purge and embargo controls, and reruns selected configurations within each split. Fold results are summarized through median objective, adverse worst fold, dispersion, sign consistency, stability, and a robustness score.

This is CPCV-style robustness validation rather than full independent-path CPCV. Adaptive warm-up requirements cause early folds to fail, so successful-fold coverage is reported separately and reduces recommendation confidence.

## Strategy selection engine

The Phase 3F engine combines net performance, stress evidence, replication results, CPCV coverage, turnover, cost, defensive-sleeve metadata, sufficient-history checks, and HMM walk-forward validity. Gates return PASS, WARN, FAIL, or NOT_AVAILABLE. Candidate scores are mapped to investor profiles, but role guardrails prevent an adaptive strategy from being promoted to the growth core without repeated net growth evidence.

Manager-facing output identifies:

- strategic core,
- risk-control overlay or robustness fallback,
- confidence,
- key trade-offs,
- warnings and assumptions.

## Final conclusion

This project does not claim one universal best strategy.

- **HERC is the strategic growth core.**
- **HMM Conservative is the drawdown-control overlay.**
- **Rule-based Conservative is the robustness reference and HMM fallback.**
- **Equal Weight is the benchmark.**

In the latest matched primary scenario, HERC produced 15.01% net CAGR and a 2.43 million final value from 1 million, while HMM Conservative reduced maximum drawdown from -18.91% to -7.78% and raised Calmar from 0.794 to 1.521. HMM still finished with lower CAGR and terminal value, so its value is risk control rather than growth replacement. Confidence is Moderate because adaptive CPCV coverage remains limited.

## Known limitations

- Historical backtests do not establish future performance.
- Yahoo Finance histories can be revised.
- The primary universe and replication grid are bounded.
- Adaptive CPCV successful-fold coverage is low.
- CPCV ranking does not yet directly penalize missing folds.
- HMM results depend on sample, features, state count, and fitting stability.
- Transaction costs do not model complete market impact.
- Defensive-sleeve assumptions affect adaptive returns.
- No NLP or macro-sentiment signal is implemented.

## Future work

The highest-priority improvement is coverage-aware CPCV ranking. Further work includes broader market replication, stronger data and model governance, liquidity-aware execution costs, alternative walk-forward regime models, and separately validated NLP or macro-sentiment research.

