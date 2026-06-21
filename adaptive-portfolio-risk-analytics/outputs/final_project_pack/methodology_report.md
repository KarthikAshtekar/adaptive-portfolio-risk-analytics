# Methodology Report

**v1.0 — Regime-Aware Portfolio Risk Analytics Platform**

## Equal Weight

Equal Weight assigns `1/N` to each asset. It requires no return forecast or covariance inversion, has low model complexity, and serves as the baseline benchmark.

## Inverse Volatility

Inverse Volatility assigns weights in proportion to the reciprocal of each asset's estimated volatility, then normalizes the weights. It reduces exposure to volatile assets but does not explicitly account for cross-asset correlation.

## Hierarchical Risk Parity

HRP converts correlations into distances, applies hierarchical clustering, quasi-diagonalizes the covariance structure, and recursively allocates capital between clusters. It avoids direct covariance-matrix inversion and is generally less sensitive to unstable expected-return estimates than Markowitz optimization.

## Hierarchical Equal Risk Contribution

HERC uses the hierarchical tree to form clusters and distribute risk across them. In this project it is the strategic growth core because the latest matched scenario produced the strongest net CAGR and terminal value among the main role candidates.

## Covariance estimation

The platform supports:

- **Sample covariance:** direct historical estimate; simple but noise-sensitive.
- **Ledoit-Wolf:** shrinkage toward a structured target to improve conditioning.
- **EWMA:** gives greater weight to recent observations.
- **EWMA plus Ledoit-Wolf:** combines recency weighting with shrinkage.

Covariance outputs are validated for shape, symmetry, finite values, and positive diagonal entries.

## Transaction costs

At each rebalance, turnover is calculated from portfolio-weight changes and converted into cost using base transaction cost plus slippage assumptions. Net portfolio returns and values include this drag; gross series remain separate for reconciliation. Cost assumptions affect both final value and whether an active overlay remains attractive.

## Volatility targeting

Volatility targeting scales risky exposure using target volatility divided by realized volatility, subject to exposure bounds. Unallocated risky capital moves to a defensive sleeve. In adaptive policies, target volatility and exposure limits depend on the lagged decision regime.

## Value at Risk and Expected Shortfall

Historical VaR estimates a loss threshold from the empirical return distribution at a selected confidence level. Expected Shortfall, also called CVaR, averages losses beyond that threshold and therefore captures tail severity.

The repository has two display conventions:

- experiment `var_95` and `cvar_95` are signed return-tail values, usually negative;
- dashboard historical VaR and ES are positive loss values.

## Stress testing

Historical stress tests evaluate portfolio behavior during identified market windows and benchmark-defined worst rolling periods. Hypothetical and correlation stress tools estimate the impact of specified shocks. Stress evidence is diagnostic and complements, rather than replaces, full-period performance.

## Rule-based regime detection

The rule-based method uses historical volatility percentile, drawdown, trend, momentum, average correlation, and benchmark features to classify Calm, Normal, Stress, Crisis, or Unknown. Priority rules identify severe conditions first. Observed labels are shifted by at least one period before adaptive use to reduce look-ahead bias.

## HMM walk-forward regime detection

A Gaussian Hidden Markov Model estimates latent market states from regime features. Hidden state numbers have no inherent economic meaning, so states are mapped to readable regimes using risk characteristics.

**Full-sample HMM is historical visualization only. Trading-safe recommendations use HMM walk-forward decisions with lagging.**

Walk-forward inference trains on expanding prior history, refits at a configured frequency, and produces a lagged decision regime. If the dependency, history, or fit is inadequate, HMM adaptive evidence is skipped or rejected rather than replaced with full-sample output.

## Adaptive allocation policies

Conservative, Balanced, and Aggressive presets map each decision regime to:

- allocator,
- covariance method,
- target volatility,
- rebalance rule and threshold,
- defensive floor,
- risky-exposure cap.

Conservative lowers target volatility, raises defensive floors, and lowers risky caps. Aggressive does the reverse within bounded limits. Two-state HMM Risk-On and Risk-Off labels map to Calm-style and Stress-style policies.

## CPCV-style robustness validation

Ordered time observations are divided into blocks. Combinations of blocks become test sets, while test observations are removed from training and nearby observations are purged or embargoed. Each selected configuration is rerun inside each fold.

The validation summary reports median objective, adverse worst fold, dispersion, sign consistency, stability, successful folds, and failed folds. The robustness score combines median rank, worst-fold rank, and stability for the selected objective. Adaptive warm-up requirements can invalidate early folds, so fold coverage must be interpreted alongside rank.

## Evidence-gated strategy selection

The selector evaluates net-return basis, full-sample HMM exclusion, data sufficiency, CPCV coverage, worst-fold sign, turnover, cost, stress evidence, defensive metadata, HMM validity, and replication classification. Gates are combined with investor-profile weights for growth, drawdown control, robustness, and implementation cost.

Role guardrails remain explicit:

- HERC: strategic growth core
- HMM Conservative: risk-control overlay
- Rule-based Conservative: robustness reference and HMM fallback
- Equal Weight: benchmark

The engine outputs a recommendation and confidence level, but does not convert a historical backtest into a live-trading claim.

