# Final Project Report: Adaptive Portfolio Risk Analytics

## Technical Summary

This project progressed from raw market-data understanding to a modular portfolio research platform covering data quality, returns and risk estimation, covariance research, hierarchical portfolio construction, rolling backtesting, realistic trading frictions, risk attribution, adaptive volatility targeting, benchmark comparison, and experiment sensitivity analysis.

The core analytical chain is:

```text
Market prices
  -> data validation and cleaning
  -> simple/log returns
  -> volatility, covariance, correlation, and distance matrices
  -> hierarchical clustering
  -> allocation strategies: Equal Weight, Inverse Volatility, Mean-Variance, HRP, HERC
  -> rolling walk-forward backtests
  -> performance, drawdown, risk contribution, turnover, and cost analytics
  -> dashboard and experiment sensitivity workflows
```

The current implementation is strongest as a research and teaching platform for long-only portfolio construction. It can compare strategies, covariance estimators, rebalance rules, transaction-cost assumptions, and volatility-targeting overlays. The most important future work is to complete true regime detection, NLP sentiment signals, CPCV/walk-forward robustness validation, liquidity-aware execution modeling, and production-grade data governance.

## Current Completion Boundary

Implemented and tested:

- Yahoo Finance data acquisition, inspection, and preprocessing.
- Return, volatility, covariance, correlation, and distance calculations.
- Hierarchical clustering and dendrogram support.
- Equal Weight, Inverse Volatility, Mean-Variance, HRP, and HERC allocation paths.
- Covariance estimator factory with sample, Ledoit-Wolf, EWMA, and EWMA plus Ledoit-Wolf estimators.
- Rolling backtesting with calendar, threshold, and calendar-or-threshold rebalancing.
- Turnover, transaction-cost, cost-drag, and rebalance diagnostics.
- Performance metrics, risk metrics, and risk-contribution analytics.
- Benchmark-relative strategy comparison.
- Adaptive volatility-targeting overlay with a separate defensive sleeve.
- Experiment-grid orchestration, sensitivity summaries, and optional local MLflow logging.
- Streamlit dashboard integration for the main portfolio workflow.

Still mostly extension-point or partial:

- Full Markov-switching regime detection.
- NLP sentiment analysis for RBI policy text, earnings-call transcripts, and uncertainty scoring.
- Dynamic allocation driven by macro/sentiment/regime signals.
- Combinatorial Purged Cross-Validation (CPCV).
- Asset-specific liquidity, market impact, taxes, and defensive-sleeve trading-cost accounting.
- Production data warehouse, persistent data catalog, deployment, and monitoring.

## Logical Project Progression

### Stage 1: Data Understanding

The project started with runtime market-data acquisition rather than a stored dataset. A provider-style interface was added for Yahoo Finance downloads, returning adjusted prices, volume, raw payload metadata, and an inspection table.

Key assumptions:

- Adjusted close is preferred over raw close because it incorporates corporate actions.
- Data is fetched on demand and kept in memory.
- Volume is collected for quality and tradability inspection, even though early stages use prices only.
- Network availability and Yahoo Finance reliability are external dependencies.

This stage established the first invariant: all later analytics depend on a clean, labeled, date-indexed price panel.

### Stage 2: Returns and Data Quality

Stage 2 converted prices into simple and log returns, added anomaly detection, and stabilized return inputs before risk estimation.

Key formulas:

```text
simple_return_t = P_t / P_{t-1} - 1

log_return_t = log(P_t / P_{t-1})

annualized_volatility = std(daily_returns) * sqrt(252)
```

Data-quality rules:

- Drop assets with more than 5 percent missing observations, then forward-fill/back-fill retained assets.
- Flag suspicious price moves using absolute log-return thresholding; default threshold is 0.50.
- Repair flagged price anomalies through interpolation.
- Detect return outliers using MAD modified z-scores or z-scores.
- Stabilize returns by winsorizing to the default range [-20 percent, 20 percent].

Logical role:

The project deliberately fixed data-quality issues before estimating covariance. This matters because covariance, clustering, and optimization are highly sensitive to extreme observations.

### Stage 3: Covariance, Correlation, and Distance

Stage 3 built the diversification geometry. Covariance measures co-movement in return units; correlation standardizes that co-movement; distance converts correlation into a clustering input.

Key formulas:

```text
Sigma_ij = cov(r_i, r_j)

rho_ij = Sigma_ij / (sigma_i * sigma_j)

d_ij = sqrt((1 - rho_ij) / 2)
```

The Stage 3 sample universe showed low average correlation, with gold behaving differently from equity assets. This justified hierarchical clustering rather than treating diversification as only an asset-count problem.

### Stage 4: Hierarchical Clustering

Stage 4 transformed the distance matrix into a tree. It added linkage matrix creation, cluster assignment, member extraction, and dendrogram plotting.

Concepts covered:

- A linkage matrix stores the sequence of merges in hierarchical clustering.
- Ward linkage favors compact clusters by minimizing within-cluster variance.
- Average linkage uses average pairwise distance and may create more elongated clusters.
- Cluster membership depends on the full distance geometry, not just one pairwise correlation.

Logical role:

This stage did not allocate weights. It produced the structural input required for HRP and later HERC.

### Stage 5: Hierarchical Risk Parity

Stage 5 implemented the first hierarchy-aware portfolio allocator: HRP.

Core HRP steps:

1. Estimate covariance.
2. Convert covariance to correlation and distance.
3. Build a hierarchical linkage tree.
4. Extract quasi-diagonal asset order.
5. Recursively split the ordered list.
6. Allocate capital inversely to cluster variance.

Key formulas:

```text
cluster_variance_C = w_ivp,C' * Sigma_C * w_ivp,C

allocation_to_left = variance_right / (variance_left + variance_right)

allocation_to_right = variance_left / (variance_left + variance_right)
```

where `w_ivp,C` is the inverse-variance portfolio inside cluster `C`.

Interpretation:

- HRP avoids direct mean-return forecasting.
- It uses covariance structure and hierarchy to reduce concentration in correlated clusters.
- It is usually more stable than unconstrained mean-variance optimization when expected returns are noisy.

### Stage 6: Rolling Backtesting

Stage 6 added walk-forward evaluation. Strategies were no longer judged only by static weights; they were tested through a rolling training window and periodic rebalancing.

Default design choices:

- Training window: 252 trading days.
- Rebalance frequency: monthly.
- Initial capital: 1,000,000.
- Strategies compared: Equal Weight, Inverse Volatility, and HRP.

Core simulation logic:

```text
For each backtest date t after the training window:
  train allocator on returns up to t
  compute target weights when rebalance rules require it
  apply portfolio weights to next-period returns
  update portfolio value
  record returns, values, drawdowns, and weights
```

Portfolio return:

```text
r_p,t = w_{t-1}' * r_t
```

Logical role:

Backtesting changed the project from a portfolio-construction exercise into a strategy-evaluation framework. It introduced out-of-sample thinking, rebalancing effects, and performance measurement.

### Stage 7: Analytics and Dashboard

Stage 7 completed the first end-to-end application with performance metrics, risk metrics, Plotly visualizations, and a Streamlit dashboard.

Performance formulas:

```text
cumulative_return = product(1 + r_t) - 1

CAGR = (product(1 + r_t))^(252 / n) - 1

annualized_volatility = std(r_t) * sqrt(252)

Sharpe = mean(r_t - rf / 252) / std(r_t - rf / 252) * sqrt(252)

Sortino = mean(r_t - target / 252) / downside_deviation * sqrt(252)

Calmar = CAGR / abs(max_drawdown)
```

Risk formulas:

```text
portfolio_value_t = product_{i <= t}(1 + r_i)

drawdown_t = portfolio_value_t / running_max(portfolio_value)_t - 1

max_drawdown = min(drawdown_t)

VaR_95 = 5th percentile of returns

CVaR_95 = mean(returns <= VaR_95)

rolling_volatility_t = std(r_{t-window:t}) * sqrt(252)
```

Dashboard capabilities:

- Select assets and date ranges.
- Choose strategies.
- Run construction and backtesting.
- Compare growth curves, drawdowns, metrics, weights, correlation heatmaps, and dendrograms.

Logical role:

This was the first point where the full pipeline became usable interactively.

### Stage 8: Covariance Research Engine

Stage 8 generalized covariance estimation through `CovarianceFactory`.

Supported methods:

| Method | Purpose | Main assumption |
| --- | --- | --- |
| `sample` | Direct historical covariance | All observations have equal weight |
| `ledoit_wolf` | Shrunk covariance | A structured target can reduce estimation noise |
| `ewma` | Exponentially weighted covariance | Recent observations are more relevant |
| `ewma_ledoit_wolf` | EWMA plus shrinkage | Combine recency sensitivity with regularization |

Key formulas:

```text
sample_covariance = cov(R)

Ledoit-Wolf: Sigma_LW = (1 - delta) * S + delta * F

EWMA alpha = 2 / (span + 1)
```

where `S` is sample covariance, `F` is the shrinkage target, and `delta` is the learned shrinkage intensity.

Validation invariants:

- Output is a labeled `pd.DataFrame`.
- Matrix is square.
- Matrix is symmetric.
- Diagonal values are positive.
- No NaNs.
- Estimator metadata is attached.

Logical role:

This stage made covariance choice an explicit research variable rather than a hidden implementation detail.

### Phase 2A Audit and Patch

The Phase 2A audit found two important research-interface gaps:

- HERC initially returned unlabeled arrays, losing asset identity.
- HRP did not yet expose covariance-method selection through the allocator interface.

The patch fixed both:

- HRP and HERC now return labeled weight `pd.Series` objects.
- HRP and HERC both route covariance estimation through `CovarianceFactory`.
- Monthly frequency handling was updated to avoid the pandas `"M"` deprecation warning while preserving user-facing compatibility.

Logical role:

This patch improved comparability. HRP versus HERC can now be studied under the same covariance-method assumptions at the allocator level.

### Stage 9: Hierarchical Equal Risk Contribution

Stage 9 added HERC as a distinct hierarchical risk-budgeting allocator.

HERC differs from HRP:

- HRP splits the quasi-diagonal ordered list by midpoint.
- HERC traverses the explicit linkage tree.
- HRP allocates using cluster variance.
- HERC allocates sibling branch capital to equalize branch-level risk.

HERC formulas:

```text
cluster_risk_C = sqrt(w_iv,C' * Sigma_C * w_iv,C)

weight_left = parent_weight * risk_right / (risk_left + risk_right)

weight_right = parent_weight * risk_left / (risk_left + risk_right)
```

where `w_iv,C` is the local inverse-volatility proxy inside a cluster.

Logical role:

HERC turned the project from "HRP versus simple baselines" into a broader comparison of hierarchical allocation philosophies.

### Stage 10: Risk Contribution Analytics

Stage 10 added risk attribution, which separates capital allocation from actual volatility contribution.

Key formulas:

```text
portfolio_volatility = sqrt(w' * Sigma * w)

MRC = Sigma * w / portfolio_volatility

TRC_i = w_i * MRC_i

PRC_i = TRC_i / portfolio_volatility
```

where:

- `MRC` is marginal risk contribution.
- `TRC` is total risk contribution.
- `PRC` is percentage risk contribution.

Interpretation:

- A high-weight asset can contribute modest risk if it has low volatility or diversifying covariance.
- A low-weight asset can dominate portfolio risk if it is volatile or strongly correlated with the portfolio.
- HRP and HERC can be compared by their risk budgets, not just by their weights.

Logical role:

This stage gave the platform an explanation layer: why a strategy behaves defensively or aggressively.

### Stage 11: Benchmark Framework

Stage 11 added explicit benchmark-relative strategy comparison.

Supported strategies:

- Equal Weight.
- Inverse Volatility.
- HRP.
- HERC.

Comparison outputs:

```text
excess_cagr = strategy_cagr - benchmark_cagr

excess_sharpe = strategy_sharpe - benchmark_sharpe

drawdown_difference = strategy_max_drawdown - benchmark_max_drawdown

volatility_difference = strategy_volatility - benchmark_volatility

final_value_difference = strategy_final_value - benchmark_final_value
```

Logical role:

The project stopped asking only "how did this strategy perform?" and started asking "better than what?"

### Stage 12: Realistic Backtesting

Stage 12 upgraded the backtester from a clean theoretical simulator into a friction-aware engine.

Turnover:

```text
turnover = 0.5 * sum(abs(target_weights - current_weights))
```

Transaction cost:

```text
cost_rate = (base_bps + slippage_bps) / 10000
cost_rate += volatility_multiplier * portfolio_volatility

transaction_cost = turnover * portfolio_value * cost_rate
```

Natural weight drift:

```text
new_weight_i = old_weight_i * (1 + asset_return_i) / (1 + portfolio_return)
```

Rebalance modes:

- `calendar`: rebalance on a time schedule.
- `threshold`: rebalance when live weights drift too far from target.
- `calendar_or_threshold`: rebalance when either condition is met.

Threshold rule:

```text
max(abs(current_weights - target_weights)) >= threshold
```

Diagnostics added:

- Rebalance log.
- Turnover summary.
- Rebalance reason counts.
- Gross versus net portfolio value.
- Cost-drag summary.

Logical role:

This stage made the backtest answer a more practical question: does a strategy survive trading friction?

### Phase 2B Rebalance Audit

The Phase 2B audit found that threshold rebalancing originally mixed two effects:

- natural live-weight drift from asset returns;
- daily optimizer target changes.

The fix separated:

- target update cadence;
- rebalance trigger cadence.

Default design:

```text
target_update_frequency = monthly
rebalance trigger = calendar, threshold, or calendar_or_threshold
```

This makes threshold rebalancing interpretable: it now reacts to drift against the latest stored target rather than daily optimizer noise.

### Stage 13: Adaptive Volatility Targeting

Stage 13 added a rule-based overlay on top of an existing risky strategy return stream.

Targeted return:

```text
r_targeted,t =
  exposure_t * risky_strategy_return_t
  + (1 - exposure_t) * defensive_asset_return_t
```

Exposure:

```text
exposure_t = clip(target_vol_t / realized_vol_{t-1}, exposure_floor, exposure_cap)
```

Realized volatility:

```text
realized_vol_t = std(risky_returns_{t-window:t}) * sqrt(252)
```

No-look-ahead safeguards:

- Realized volatility is shifted by one day.
- Regime classification uses lagged realized volatility.
- Exposure for day `t` is based only on information available through `t-1`.

Regime-specific target volatility:

| Regime | Percentile rule | Target volatility |
| --- | --- | ---: |
| Calm | percentile <= 40 percent | 12 percent |
| Normal | 40 percent < percentile <= 80 percent | 10 percent |
| Stress | 80 percent < percentile <= 95 percent | 6 percent |
| Crisis | percentile > 95 percent | 3 percent |

Defensive sleeve assumptions:

- The defensive asset is separate from the risky universe.
- It is not included in covariance estimation, clustering, HRP, or HERC.
- Default preferred tickers are `LIQUIDBEES.NS` and `LIQUIDETF.NS`.
- If live defensive tickers are unavailable, a synthetic risk-free series is used.
- Default synthetic annual rate is 4 percent, converted to `0.04 / 252` daily.

Logical role:

This stage added adaptive risk scaling without changing the base allocator. The allocator controls risky-asset composition; the overlay controls how much risky exposure to hold.

### Stage 14: Experiment Sensitivity Framework

Stage 14 added orchestration for parameter-grid research.

The experiment framework coordinates existing modules rather than adding a new allocation method.

Supported experiment parameters:

- Strategy.
- Covariance method.
- Rebalance mode.
- Threshold.
- Transaction cost bps.
- Slippage bps.
- Volatility targeting enabled or disabled.
- Target volatility.
- Defensive asset.
- Training window.
- Initial capital.

Ranking objectives:

- CAGR.
- Sharpe.
- Sortino.
- Calmar.
- Maximum drawdown.
- Final value.

Sensitivity methods:

```text
rank_experiments(results, objective)
summarize_by_parameter(results, parameter, metric)
compute_parameter_sensitivity(results, metric)
```

Logical role:

The project shifted from single-run analysis to controlled research over strategy assumptions.

### Dashboard UI Refactor

The dashboard was later reorganized to make the growing workflow usable:

- Sidebar inputs were grouped into expanders.
- Main outputs were moved into tabs.
- Indian asset universe presets were added.
- Manual ticker override was preserved.
- Validation was added for asset count, dates, exposure bounds, defensive-sleeve separation, and sensitivity thresholds.

Logical role:

The UI refactor reduced operational friction after the analytics scope expanded.

## Portfolio Construction Methods

### Equal Weight

```text
w_i = 1 / N
```

Assumptions:

- No return estimation.
- No covariance estimation.
- Long-only and fully invested.
- Useful as a naive but strong baseline.

### Inverse Volatility

```text
w_i = (1 / sigma_i) / sum_j(1 / sigma_j)
```

Assumptions:

- Lower-volatility assets deserve higher capital allocation.
- Correlation is not directly modeled.
- Works best when volatility is the dominant risk driver.

### Mean-Variance / Max-Sharpe

Objective:

```text
maximize (w' * mu - rf) / sqrt(w' * Sigma * w)

subject to:
  sum(w_i) = 1
  min_weight <= w_i <= max_weight
  optional target return constraint
```

Assumptions:

- Expected returns and covariance are estimated accurately enough to optimize.
- The default implementation is long-only and fully invested.
- A small diagonal jitter is added to stabilize covariance numerics.

Risk:

- Sensitive to estimation error.
- Can produce unstable or concentrated weights without constraints.

### HRP

HRP is most useful when the covariance structure is meaningful but expected-return forecasts are unreliable. It uses hierarchical clustering and recursive variance allocation to avoid concentrating capital in highly related clusters.

### HERC

HERC is most useful when the research objective is explicit hierarchical risk budgeting. It equalizes sibling branch risk recursively and can produce different risk-contribution behavior from HRP even under the same covariance estimate.

## Key Assumptions Across the Project

Data assumptions:

- Price data is adjusted close when available.
- Daily observations are treated as the base frequency.
- 252 trading days per year is the annualization convention.
- Most workflows assume a clean `DatetimeIndex`.
- At least two risky assets are required for portfolio analytics.

Return and risk assumptions:

- Default risk-free rate for Sharpe is 2 percent annually.
- Default Sortino target return is 0 percent annually.
- VaR and CVaR are historical quantile metrics, not parametric normal estimates.
- Volatility is standard deviation based and annualized by `sqrt(252)`.

Portfolio assumptions:

- Allocators are long-only.
- Portfolios are fully invested in the risky universe unless a volatility-targeting overlay is applied.
- HRP and HERC covariance methods are configurable through `CovarianceFactory`.
- Defensive assets are excluded from risky-universe covariance and clustering.

Backtesting assumptions:

- Rolling backtests skip the initial training window.
- The allocator is trained only on historical window data.
- Target weights update on a configured cadence.
- Transaction costs are linear in turnover.
- Threshold rebalancing compares live drifted weights to the latest stored target.
- Gross and net values are both tracked after Stage 12.

Dashboard assumptions:

- YFinance downloads are cached in Streamlit where appropriate.
- The dashboard prioritizes research usability over production deployment.
- The default Indian asset universe is currently hardcoded in `src/dashboard/app.py`.

## Validation and Test Evidence

The project used staged unit and integration testing rather than one final monolithic validation pass.

Important validation milestones:

- Stage 1 data pipeline tests: provider outputs, adjusted-close fallback, volume handling, inspection table.
- Stage 2 preprocessing tests: anomaly detection, repair, MAD/z-score outliers, winsorization, integration.
- Stage 3 covariance tests: covariance, correlation, distance outputs.
- Stage 4 clustering tests: linkage matrix, cluster labels, dendrogram figure.
- Stage 5 HRP tests: HRP weights, cluster variance, recursive bisection.
- Stage 6 backtesting tests: output structure, weight normalization, drawdown, rebalance dates, reproducibility.
- Stage 7 analytics tests: performance metrics, risk metrics, edge cases, reproducibility.
- Stage 8 covariance tests: estimator routing, shape, symmetry, positive diagonal, labels, metadata.
- Stage 9 HERC tests: weights, covariance methods, backtester integration, dashboard exposure.
- Phase 2A patch tests: HRP/HERC covariance comparability and labeled outputs.
- Stage 10 risk-contribution tests: MRC/TRC/PRC properties and HRP/HERC comparisons.
- Stage 11 benchmark tests: strategy factory, aliases, comparison result structure, relative performance.
- Stage 12 realistic backtesting tests: turnover, rebalance rules, diagnostics, transaction-cost behavior.
- Phase 2B audit tests: threshold semantics, target update separation, rebalance reason logging.
- Stage 13 tests: defensive sleeve fallback, volatility-targeting no-look-ahead behavior.
- Stage 14 tests: experiment configuration, grid runner, sensitivity helpers.

The report was generated from the existing stage reports, audit reports, methodology notes, and source implementation files. No market-data-dependent live validation was rerun while creating this final document.

## Main Limitations

### Data Limitations

- Yahoo Finance is convenient but not production-grade.
- There is no persistent raw data catalog or reproducible snapshot layer.
- Corporate-action handling depends on adjusted close quality from the provider.
- Volume is inspected but not yet used for liquidity-aware execution constraints.
- The dashboard asset universe is hardcoded.

### Statistical Limitations

- Sample covariance and EWMA can remain noisy in small samples.
- Ledoit-Wolf shrinkage improves conditioning but can smooth away genuine recent structure.
- Correlation distance assumes correlation is an adequate similarity measure.
- Risk metrics are mostly descriptive and historical.
- VaR/CVaR are not stress-scenario or forward-looking risk estimates.

### Backtesting Limitations

- Transaction costs are linear in turnover.
- Market impact, spread dynamics, taxes, and capacity are not modeled.
- Defensive-sleeve trading costs are not separately modeled.
- Rebalance settings are global rather than strategy-specific in some dashboard flows.
- CPCV is reserved for a future phase.

### Model Limitations

- Full regime detection is not implemented.
- NLP sentiment modules are currently placeholders or simple scaffolds.
- Dynamic allocation is reserved for a future phase.
- Volatility targeting uses realized-volatility rules rather than forecast models.
- Experiment sensitivity is descriptive; it is not yet an optimizer.

### Product Limitations

- The dashboard is still a single-file Streamlit app with supporting plot/components modules.
- There are no dedicated UI regression tests.
- Export/reporting workflows are limited.
- No deployment, authentication, user permissions, or scheduled refresh workflow is included.

## Most Necessary Future Extensions

### 1. Complete Regime Detection and Dynamic Allocation

This is the highest-value next extension because the project title and roadmap emphasize dynamic correlation and sentiment regimes.

Required work:

- Implement Markov-switching or hidden Markov regime detection with filtered and smoothed probabilities.
- Add volatility, correlation, trend, drawdown, and macro features.
- Validate regime labels out of sample.
- Connect regime states to allocator choice, covariance method, target volatility, and defensive allocation.
- Ensure all regime features are lagged to avoid look-ahead bias.

Expected output:

- A regime-aware allocator or strategy controller that decides when to prefer HRP, HERC, inverse volatility, defensive exposure, or lower target volatility.

### 2. Implement NLP and Macro Sentiment Signals

The NLP package currently contains scaffolding for RBI policy sentiment, earnings-call analysis, and uncertainty scoring. These should become actual lagged signals only after a careful data pipeline is defined.

Required work:

- Build document ingestion for RBI statements, policy minutes, speeches, and earnings-call transcripts.
- Use finance-specific language models or lexicons, with domain validation.
- Store document timestamps and release dates.
- Convert sentiment into lagged features.
- Test whether sentiment improves drawdown control, volatility targeting, or allocation decisions.

Expected output:

- A sentiment feature table that can be used in regime detection and dynamic allocation.

### 3. Add CPCV and Walk-Forward Robustness Validation

Current rolling backtests are useful but can still overfit strategy parameters. CPCV and embargo logic are necessary for research-grade parameter selection.

Required work:

- Implement combinatorial purged cross-validation for time-series data.
- Add embargo periods around test folds.
- Evaluate parameter grids across multiple train/test partitions.
- Report stability of Sharpe, Calmar, drawdown, turnover, and cost drag.

Expected output:

- Robustness scores that distinguish genuinely stable strategies from lucky backtest winners.

### 4. Build Production-Grade Data Governance

The platform should move from ad hoc runtime downloads to reproducible datasets.

Required work:

- Add a local or cloud data cache with versioned raw and cleaned data.
- Store provider, download timestamp, adjusted-price field, and cleaning reports.
- Add benchmark, risk-free rate, sector, liquidity, and corporate-action reference data.
- Add data-quality dashboards and failure alerts.

Expected output:

- Reproducible experiments where the same data snapshot can be audited later.

### 5. Add Liquidity-Aware Transaction Cost and Market Impact Modeling

Stage 12 added a strong first transaction-cost layer, but production realism requires asset-specific costs.

Required work:

- Use bid-ask estimates, volume, average daily value traded, and volatility.
- Model market impact as a nonlinear function of participation rate.
- Separate base portfolio trading costs from defensive-sleeve overlay costs.
- Add capacity analysis: how much capital can the strategy manage before returns degrade?

Expected output:

- Strategy rankings that account for liquidity and scalable execution.

### 6. Extend Risk Attribution and Stress Testing

Risk contribution is currently point-in-time. The platform needs rolling and scenario-aware risk attribution.

Required work:

- Add rolling risk-contribution history.
- Add sector, factor, and asset-class attribution.
- Implement historical scenarios and correlation-stress scenarios in dashboard flows.
- Add reverse stress tests to find market moves that breach loss limits.

Expected output:

- A risk explanation layer that can answer what is driving risk today and what breaks the portfolio.

### 7. Add Multi-Objective Optimization and Experiment Governance

Stage 14 coordinates experiments but does not optimize them.

Required work:

- Add Optuna or another optimizer for parameter search.
- Support multi-objective ranking: maximize Calmar and Sharpe while minimizing drawdown, turnover, and cost drag.
- Promote MLflow from optional local logging to a structured experiment registry.
- Record dataset snapshot, code version, configuration, and results for every run.

Expected output:

- A governed research loop where experiments are reproducible, comparable, and not manually cherry-picked.

### 8. Modularize and Test the Dashboard

The Streamlit dashboard has grown with the project. It should be decomposed before more features are added.

Required work:

- Move sidebar controls, result tabs, validation, and workflow orchestration into separate modules.
- Add lightweight UI tests for validation logic and strategy routing.
- Add report export for benchmark tables, risk attribution, and experiment results.
- Move the hardcoded asset universe into configuration.

Expected output:

- A dashboard that remains maintainable as regime, NLP, and production data features are added.

## Recommended Next Project Phase

The next phase should not start with a new dashboard feature. It should first close the research-validity gap:

1. Implement CPCV and walk-forward robustness scoring.
2. Use that validation layer to compare existing HRP, HERC, covariance, rebalance, cost, and volatility-targeting choices.
3. Only then add regime and sentiment signals, because those signals need the same validation framework to avoid overfitting.

This ordering keeps the project scientifically defensible: first make strategy evaluation robust, then add more predictive inputs.

## Conclusion

The project has evolved into a coherent adaptive portfolio analytics platform. The strongest completed areas are data quality, covariance research, hierarchical allocation, realistic backtesting, benchmark comparison, risk attribution, volatility targeting, and sensitivity analysis.

The remaining work is not minor polish; it is the set of extensions needed to move from a strong research prototype to a robust adaptive portfolio system. The highest-priority gaps are regime detection, sentiment integration, CPCV robustness, production data governance, and liquidity-aware execution modeling.
