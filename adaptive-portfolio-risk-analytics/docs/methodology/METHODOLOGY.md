# Methodology

## Research scope

The project compares fixed and regime-adaptive portfolio policies on aligned daily return data.
Outputs are historical research evidence, not forecasts, investment advice, or live execution.

## Data and preprocessing

`YahooFinanceProvider` requests adjusted close when available, falls back to close, and retains
volume for inspection. `DataQualityProcessor` centralizes missingness, anomaly, interpolation,
return, and winsorization rules. Downstream optimizers and dashboards consume cleaned data rather
than silently applying their own missing-data policy.

Simple and log returns are both available in preprocessing. Portfolio backtests use simple daily
returns because they compound directly through `1 + r`.

## Covariance and clustering

`CovarianceFactory.compute` supports:

- `sample`: direct historical covariance;
- `ledoit_wolf`: shrinkage toward a structured target;
- `ewma`: exponentially weighted observations;
- `ewma_ledoit_wolf`: recency weighting plus shrinkage.

All outputs are labeled, finite, symmetric covariance DataFrames with positive diagonals. The
hierarchical path is correlation -> distance -> linkage -> cluster tree. Gerber covariance is a
documented future extension, not a current capability.

## Portfolio construction

- Equal Weight assigns `1/N` and is the transparent baseline.
- Inverse Volatility scales each asset inversely to standalone volatility.
- Mean-Variance maximizes an estimated long-only Sharpe objective, with equal weight as a solver
  fallback. It is implemented but not routed through the benchmark factory or dashboard.
- HRP orders assets through the linkage tree and recursively allocates according to cluster
  variance.
- HERC recursively gives equal risk budgets to the left and right branches of the actual cluster
  tree, then propagates those budgets to leaves. This differs from HRP's quasi-diagonal
  bisection and can therefore produce different weights on the same covariance estimate.

## Backtesting, rebalancing, and costs

`RollingBacktester` uses a rolling training window and applies selected weights one observation
later. Calendar, threshold, and calendar-or-threshold modes are supported. Target weights update
at `target_update_frequency` (monthly by default); current weights drift with realized asset
returns. Threshold triggers compare that drift with the latest stored target.

Turnover is half the absolute weight change. Base transaction cost and slippage are applied at
rebalance events. The engine exposes net and gross returns/values, rebalance reason, turnover,
transaction cost, and maximum weight drift.

Volatility targeting estimates lagged realized volatility, applies fixed or regime-adaptive
targets, clips exposure to configured bounds, and places residual exposure in a documented
defensive sleeve.

## Risk and performance analytics

The analytics layer includes CAGR, volatility, Sharpe, Sortino, Calmar, maximum drawdown, Pain
Index/Pain Ratio, risk contribution, concentration, tracking error, information ratio, beta,
Jensen's alpha, liquidity diagnostics, and historical/hypothetical stress tests.

Two VaR/ES conventions exist and must not be mixed:

- experiment `var_95`/`cvar_95` are signed return-tail statistics;
- dashboard historical VaR/ES are displayed as positive losses.

## Regimes and adaptive allocation

Rule-based labels use rolling volatility, drawdown, trend, momentum, correlation, and return-shock
features. Observed labels are lagged before decision use. HMM walk-forward inference trains only
on prior expanding history and emits lagged decision states. Full-sample HMM output is restricted
to historical visualization.

Adaptive policies map decision regimes to allocator, covariance method, target volatility,
rebalance settings, defensive floor, and risky-exposure cap. Conservative, Balanced, and
Aggressive presets are explicit policy transformations rather than learned labels.

## Experiments and robustness

Sensitivity grids vary implemented strategy, covariance, rebalance, threshold, and cost controls.
Ranking uses exactly one selected objective; turnover and costs affect ranking only through their
effect on that objective.

The CPCV-style validator creates ordered blocks, selects test-block combinations, and applies
purge and embargo around test intervals. It reports fold medians, worst folds, dispersion,
stability, failed folds, and a robustness score. It is a pragmatic robustness diagnostic and does
not guarantee future performance.

## Sentiment and NLP

The sentiment package validates provenance and timestamps, deduplicates records, applies
publication lags, scores text with deterministic lexicons or optional local FinBERT, and combines
RBI/news evidence into monitoring signals. Fixtures and placeholders are excluded from real-data
claims.

The latest saved monitoring artifact reports real RBI and news coverage, but older Phase 4A.3
artifacts record an earlier synthetic-fallback run. Each report must be interpreted using its own
generation date and corpus metadata. NLP remains monitoring/shadow evidence and does not enter
production-active portfolio weights or strategy gates.

## Reproducibility and interpretation

- Keep the asset universe, dates, costs, defensive source, objective, and estimator fixed when
  comparing methods.
- Prefer net metrics for decisions and use gross metrics to explain cost drag.
- Report failed CPCV folds and warm-up exclusions, not only successful-fold ranks.
- Do not present the best in-sample configuration as a universal strategy winner.
