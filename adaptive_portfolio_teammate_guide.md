# Adaptive Portfolio Risk Analytics: Hierarchical Allocation, Backtesting, and Volatility Targeting

**Project name:** Adaptive Portfolio Risk Analytics  
**Project type:** Risk-aware adaptive allocation and portfolio analytics platform  
**Audience:** Intermediate teammates who understand basic Python/finance but need the full project story  
**Purpose:** Viva preparation + technical documentation support  
**Current status:** Strong research prototype; not a production trading system

\---

## 1\. One-Paragraph Project Summary

This project builds a portfolio analytics platform that takes historical market prices, cleans and validates them, converts them into returns, estimates risk through volatility/covariance/correlation, constructs portfolios using multiple allocation methods, and tests those strategies through rolling backtests with realistic rebalancing and transaction-cost assumptions. The project compares Equal Weight, Inverse Volatility, Mean-Variance/Max-Sharpe, Hierarchical Risk Parity (HRP), and Hierarchical Equal Risk Contribution (HERC), then adds risk analytics, benchmark comparison, volatility targeting, and experiment sensitivity analysis. The current system is best described as a **risk-aware adaptive allocation and portfolio analytics platform**. It is not yet a full regime-detection or NLP-driven dynamic allocation engine; those are future extensions.

\---

## 2\. What Problem Are We Solving?

A normal investor may ask: “Which stocks should I buy?”  
This project asks a deeper question:

> Given a universe of assets, how can we construct and test portfolios that balance return, risk, diversification, drawdown control, rebalancing cost, and robustness?

The project does not simply pick the asset with the highest past return. Instead, it studies how assets behave together. In portfolio construction, the risk of the whole portfolio depends not only on the risk of each asset, but also on how assets move together.

A portfolio may contain 30 assets, but if all of them crash together, diversification is weak. Therefore, the project focuses on:

* estimating returns and risk,
* estimating covariance and correlation,
* understanding asset similarity through hierarchical clustering,
* constructing portfolios using different allocation philosophies,
* testing strategies out of sample through rolling backtests,
* including turnover and transaction costs,
* comparing results against benchmarks,
* explaining risk contribution, and
* running sensitivity experiments over strategy assumptions.

\---

## 3\. Final Presentation Boundary

### 3.1 Implemented and Safe to Present

The following parts are implemented and can be presented confidently:

1. Yahoo Finance data acquisition.
2. Data inspection and preprocessing.
3. Simple and log returns.
4. Volatility, covariance, correlation, and distance matrices.
5. Hierarchical clustering and dendrogram support.
6. Equal Weight allocation.
7. Inverse Volatility allocation.
8. Mean-Variance / Max-Sharpe allocation.
9. HRP allocation.
10. HERC allocation.
11. Covariance estimator factory:

    * sample covariance,
    * Ledoit-Wolf,
    * EWMA,
    * EWMA + Ledoit-Wolf.
12. Rolling walk-forward backtesting.
13. Rebalance modes:

    * `calendar`,
    * `threshold`,
    * `calendar\\\\\\\_or\\\\\\\_threshold`.
14. Turnover and transaction-cost diagnostics.
15. Gross and net portfolio value tracking.
16. Performance metrics and drawdown metrics.
17. Risk-contribution analytics.
18. Benchmark comparison.
19. Volatility-targeting overlay with defensive sleeve.
20. Experiment sensitivity analysis.
21. Streamlit dashboard.

### 3.2 Future Work Only — Do Not Overclaim

The following should be presented only as future work or partial scaffolding:

1. Full Markov-switching regime detection.
2. Real NLP sentiment pipeline for RBI policy text, earnings calls, and uncertainty scoring.
3. Dynamic allocation driven by macro/sentiment/regime signals.
4. Combinatorial Purged Cross-Validation (CPCV).
5. Production-grade data governance and persistent data catalog.
6. Liquidity-aware execution, market impact, taxes, and capacity modeling.
7. Separate defensive-sleeve trading-cost accounting.
8. Full deployment, monitoring, authentication, and scheduled refresh.
9. Alpha Vantage as a working data source; it is only a placeholder.

**Correct framing:**  
“We have built the core portfolio construction, analytics, backtesting, and volatility-targeting platform. Regime detection and NLP sentiment are planned extensions.”

**Incorrect framing:**  
“We have already built a fully AI-driven regime-aware sentiment allocation engine.”

\---

## 4\. Asset Universe and Data Setup

### 4.1 Default Dashboard Universe

The default dashboard universe is the `Core Diversified` Indian asset universe:

```text
HDFCBANK.NS, ICICIBANK.NS, SBIN.NS, KOTAKBANK.NS, AXISBANK.NS,
TCS.NS, INFY.NS, WIPRO.NS, HCLTECH.NS, TECHM.NS,
RELIANCE.NS, ONGC.NS, NTPC.NS, POWERGRID.NS,
HINDUNILVR.NS, ITC.NS, NESTLEIND.NS, TATACONSUM.NS,
SUNPHARMA.NS, DRREDDY.NS, CIPLA.NS,
TATAMOTORS.NS, MARUTI.NS, M\\\\\\\&M.NS,
LT.NS, ULTRACEMCO.NS, ASIANPAINT.NS, BHARTIARTL.NS,
GOLDBEES.NS, SILVERBEES.NS
```

This universe combines:

* Indian banking stocks,
* Indian IT stocks,
* energy and utilities,
* FMCG/consumer names,
* pharma names,
* automobile names,
* infrastructure/materials/telecom names,
* gold ETF exposure, and
* silver ETF exposure.

The gold and silver ETFs are important because they may behave differently from equities and can improve diversification.

### 4.2 Data Source

The main data source is **Yahoo Finance**.

Alpha Vantage is only a placeholder and should not be presented as a completed integration.

The project prefers **adjusted close prices** because adjusted close accounts for corporate actions such as dividends, splits, and bonuses. If we used raw close prices, return calculations may be distorted around corporate-action dates.

### 4.3 Default Dates

Default experiment range:

```text
2020-01-01 to 2025-01-01
```

Dashboard default:

```text
Start date: 2020-01-01
End date: run date
```

The dashboard can be changed manually over a broad range, roughly from 2010 to 2026 depending on data availability and interface settings.

\---

## 5\. Full Sequential Project Pipeline

The project pipeline is:

```text
Market prices
  -> data validation and cleaning
  -> simple/log returns
  -> volatility, covariance, correlation, distance matrices
  -> hierarchical clustering
  -> allocation strategies: EW, IV, MV/Max-Sharpe, HRP, HERC
  -> rolling walk-forward backtesting
  -> rebalancing and transaction-cost handling
  -> performance and risk metrics
  -> risk contribution analytics
  -> benchmark comparison
  -> volatility targeting overlay
  -> experiment sensitivity analysis
  -> dashboard visualization
```

The logic is sequential. Each stage depends on the previous stage being correct.

For example, if prices are wrong, returns are wrong. If returns are wrong, covariance is wrong. If covariance is wrong, HRP/HERC allocations are wrong. If allocations are wrong, backtest results are misleading.

\---

## 6\. Stage 1 — Data Acquisition and Data Understanding

### 6.1 What Happens in This Stage?

The system downloads historical market data for the selected tickers. The main input is a list of Yahoo Finance tickers and a date range. The output is a clean date-indexed price panel.

A price panel is a table where:

* rows are dates,
* columns are assets,
* values are prices.

Example:

|Date|HDFCBANK.NS|INFY.NS|GOLDBEES.NS|
|-|-:|-:|-:|
|2020-01-01|1000|720|36|
|2020-01-02|1010|725|36.2|
|2020-01-03|1005|730|36.5|

### 6.2 Why Adjusted Close?

Suppose a stock splits 1:2. The raw price may suddenly fall from ₹1000 to ₹500, but the investor did not lose 50%. The number of shares doubled. Adjusted close handles such corporate-action effects.

So we prefer:

```text
adjusted\\\\\\\_close\\\\\\\_price
```

instead of raw close.

### 6.3 Code Reference

Typical module responsibility:

```text
src/data/
```

Dashboard reference:

```text
src/dashboard/app.py
```

The dashboard allows asset selection, date selection, and ticker override.

\---

## 7\. Stage 2 — Returns and Data Cleaning

### 7.1 Why Convert Prices to Returns?

Portfolio models do not work directly with prices. A ₹10 movement in a ₹1000 stock is very different from a ₹10 movement in a ₹50 stock. Returns standardize price changes.

### 7.2 Simple Return

Formula:

```text
simple\\\\\\\_return\\\\\\\_t = P\\\\\\\_t / P\\\\\\\_{t-1} - 1
```

Mathematical notation:

```text
r\\\\\\\_t = (P\\\\\\\_t - P\\\\\\\_{t-1}) / P\\\\\\\_{t-1}
```

Example:

```text
Yesterday price = 100
Today price = 105
simple return = 105 / 100 - 1 = 0.05 = 5%
```

### 7.3 Log Return

Formula:

```text
log\\\\\\\_return\\\\\\\_t = log(P\\\\\\\_t / P\\\\\\\_{t-1})
```

Example:

```text
Yesterday price = 100
Today price = 105
log return = log(105 / 100) = log(1.05) ≈ 0.04879 = 4.879%
```

Simple returns are intuitive. Log returns are useful mathematically because they add over time.

### 7.4 Annualized Volatility

Daily volatility is scaled to annual volatility using 252 trading days:

```text
annualized\\\\\\\_volatility = std(daily\\\\\\\_returns) \\\\\\\* sqrt(252)
```

Example:

```text
daily volatility = 1%
annualized volatility = 1% \\\\\\\* sqrt(252) ≈ 15.87%
```

### 7.5 Data Quality Rules

The project applies rules such as:

1. Drop assets with too many missing observations.
2. Forward-fill/back-fill retained assets where appropriate.
3. Detect suspicious price moves using log-return thresholds.
4. Repair price anomalies through interpolation.
5. Detect return outliers using z-score or MAD-based methods.
6. Winsorize returns to reduce extreme outlier influence.

Default examples:

```text
missing observation tolerance: 5%
price anomaly threshold: absolute log return > 0.50
winsorization range: \\\\\\\[-20%, +20%]
```

### 7.6 Why Cleaning Matters

Covariance and optimization are very sensitive to outliers.

Example:

If one data error shows a stock moving +500% in one day, then the model may think that stock is extremely volatile. This can distort allocation weights and clustering structure.

\---

## 8\. Stage 3 — Risk, Volatility, Covariance, Correlation, and Distance

This is one of the most important parts of the project.

### 8.1 Volatility

Volatility measures how much an asset’s returns fluctuate.

Formula:

```text
sigma\\\\\\\_i = std(r\\\\\\\_i)
```

Annualized:

```text
sigma\\\\\\\_i\\\\\\\_annual = std(daily\\\\\\\_returns\\\\\\\_i) \\\\\\\* sqrt(252)
```

Interpretation:

* Higher volatility means returns fluctuate more.
* Lower volatility means returns are more stable.

But volatility alone is not enough. Two volatile assets can still diversify each other if they do not move together.

### 8.2 Covariance

Covariance measures whether two assets move together in return units.

Formula:

```text
Sigma\\\\\\\_ij = cov(r\\\\\\\_i, r\\\\\\\_j)
```

Interpretation:

* Positive covariance: assets tend to move in the same direction.
* Negative covariance: assets tend to move in opposite directions.
* Near-zero covariance: weak linear co-movement.

Example:

If HDFCBANK and ICICIBANK usually rise and fall together, their covariance is likely positive.

### 8.3 Correlation

Correlation standardizes covariance so it lies between -1 and +1.

Formula:

```text
rho\\\\\\\_ij = Sigma\\\\\\\_ij / (sigma\\\\\\\_i \\\\\\\* sigma\\\\\\\_j)
```

Interpretation:

```text
rho = +1  -> perfectly same direction
rho = 0   -> no linear relationship
rho = -1  -> perfectly opposite direction
```

Example:

If two banking stocks have correlation 0.80, they are very similar from a diversification viewpoint.

If equity and gold have correlation 0.10 or negative, gold may improve diversification.

### 8.4 Distance Matrix

Hierarchical clustering needs a distance measure. The project converts correlation into distance:

```text
d\\\\\\\_ij = sqrt((1 - rho\\\\\\\_ij) / 2)
```

Interpretation:

* If correlation is high, distance is low.
* If correlation is low or negative, distance is high.

Example:

```text
rho = 1
Distance = sqrt((1 - 1) / 2) = 0
```

Perfectly correlated assets have zero distance.

```text
rho = -1
Distance = sqrt((1 - (-1)) / 2) = sqrt(1) = 1
```

Perfectly opposite assets have maximum distance.

### 8.5 Code Reference

Typical module responsibility:

```text
src/risk/
src/portfolio/
src/analytics/
```

\---

## 9\. Stage 4 — Covariance Estimation Engine

The project does not use only one covariance estimator. It supports multiple estimators so we can test how strategy performance changes under different risk assumptions.

### 9.1 Why Covariance Estimation Matters

Portfolio risk is:

```text
portfolio\\\\\\\_variance = w' \\\\\\\* Sigma \\\\\\\* w
```

where:

```text
w      = vector of portfolio weights
Sigma  = covariance matrix
```

If the covariance matrix is unstable, optimized portfolios become unstable.

### 9.2 Sample Covariance

Formula:

```text
sample\\\\\\\_covariance = cov(R)
```

where `R` is the returns matrix.

Assumption:

```text
All historical observations are equally important.
```

Pros:

* Simple.
* Easy to understand.

Cons:

* Noisy in small samples.
* Sensitive to extreme periods.

### 9.3 Ledoit-Wolf Shrinkage

Formula:

```text
Sigma\\\\\\\_LW = (1 - delta) \\\\\\\* S + delta \\\\\\\* F
```

where:

```text
S      = sample covariance matrix
F      = structured shrinkage target
delta  = shrinkage intensity
```

Meaning:

Ledoit-Wolf pulls the noisy sample covariance matrix toward a more stable target. This often improves numerical stability.

Simple example:

Imagine one correlation estimate is unusually high only because of a short crisis window. Shrinkage reduces the chance that this unstable estimate dominates the portfolio allocation.

### 9.4 EWMA Covariance

EWMA means Exponentially Weighted Moving Average.

The idea:

```text
Recent observations should matter more than older observations.
```

Default span:

```text
span = 252
```

Equivalent alpha:

```text
alpha = 2 / (span + 1)
alpha = 2 / 253 ≈ 0.0079
```

Interpretation:

The model reacts more to recent market behavior than a simple covariance estimator.

### 9.5 EWMA + Ledoit-Wolf

This combines:

1. EWMA recency sensitivity, and
2. Ledoit-Wolf regularization.

Purpose:

```text
Capture recent structure while reducing covariance noise.
```

### 9.6 Code Reference

```text
src/risk/covariance.py
CovarianceFactory
```

The covariance factory supports:

```text
sample
ledoit\\\\\\\_wolf
ewma
ewma\\\\\\\_ledoit\\\\\\\_wolf
```

\---

## 10\. Stage 5 — Hierarchical Clustering

### 10.1 What Is Clustering?

Clustering groups similar assets together.

In this project, similarity is based on correlation distance. Highly correlated assets are close. Less correlated assets are far apart.

Example:

Banking stocks may cluster together:

```text
HDFCBANK.NS, ICICIBANK.NS, SBIN.NS, AXISBANK.NS
```

IT stocks may form another cluster:

```text
TCS.NS, INFY.NS, WIPRO.NS, HCLTECH.NS
```

Gold and silver ETFs may sit away from equity clusters.

### 10.2 Linkage Matrix

A linkage matrix records the order in which assets or clusters are merged.

It answers:

```text
Which two assets/clusters are closest?
Which clusters merge next?
At what distance do they merge?
```

### 10.3 Dendrogram

A dendrogram is a tree diagram showing how assets are grouped.

Interpretation:

* Assets connected at a low height are very similar.
* Assets connected at a high height are less similar.

### 10.4 Linkage Method

In this repo:

```text
Allocator default linkage: single
Clustering helper/YAML may mention: ward
```

So the safe statement is:

> Linkage is configurable, but HRP/HERC allocator default is `single`.

Avoid saying the whole project uses only Ward linkage.

### 10.5 Why Clustering Matters

Traditional portfolio methods may allocate too much to many similar assets. HRP and HERC try to avoid this by using the asset hierarchy.

Example:

If five banking stocks are highly correlated, holding all five is not the same as holding five independent assets. HRP/HERC use clustering to recognize this concentration.

\---

## 11\. Stage 6 — Portfolio Construction Methods

The project implements six strategy concepts:

1. Equal Weight.
2. Inverse Volatility.
3. Mean-Variance / Max-Sharpe.
4. HRP.
5. HERC.
6. Volatility Targeting overlay.

The first five decide composition inside the risky asset universe. Volatility targeting controls how much exposure is allocated to the risky portfolio versus defensive sleeve.

\---

## 12\. Strategy 1 — Equal Weight

### 12.1 Formula

```text
w\\\\\\\_i = 1 / N
```

where:

```text
w\\\\\\\_i = weight of asset i
N   = number of assets
```

Example:

If there are 5 assets:

```text
weight of each asset = 1 / 5 = 20%
```

### 12.2 Intuition

Equal Weight says:

```text
Do not estimate returns, do not estimate risk, just divide capital equally.
```

### 12.3 Pros

* Very simple.
* Hard to overfit.
* Strong benchmark.

### 12.4 Cons

* Ignores volatility.
* Ignores correlation.
* Can allocate too much to risky assets.

### 12.5 Role in Project

Equal Weight is the default benchmark. Other strategies are compared against it.

\---

## 13\. Strategy 2 — Inverse Volatility

### 13.1 Formula

```text
w\\\\\\\_i = (1 / sigma\\\\\\\_i) / sum\\\\\\\_j(1 / sigma\\\\\\\_j)
```

where:

```text
sigma\\\\\\\_i = volatility of asset i
```

### 13.2 Example

Assume two assets:

```text
Asset A volatility = 10%
Asset B volatility = 20%
```

Inverse vol scores:

```text
A score = 1 / 0.10 = 10
B score = 1 / 0.20 = 5
Total score = 15
```

Weights:

```text
A weight = 10 / 15 = 66.67%
B weight = 5 / 15 = 33.33%
```

### 13.3 Intuition

Lower-volatility assets get more weight. Higher-volatility assets get less weight.

### 13.4 Pros

* Simple risk-aware allocation.
* Does not need expected returns.
* Usually more defensive than Equal Weight.

### 13.5 Cons

* Does not directly use correlation.
* Can over-allocate to low-volatility assets even if they are highly correlated.

\---

## 14\. Strategy 3 — Mean-Variance / Max-Sharpe

### 14.1 Background

Mean-Variance Optimization comes from Markowitz portfolio theory. It uses expected returns and covariance to construct an efficient portfolio.

The Max-Sharpe version tries to maximize return per unit of risk.

### 14.2 Portfolio Return

```text
portfolio\\\\\\\_return = w' \\\\\\\* mu
```

where:

```text
w  = portfolio weights
mu = expected returns
```

### 14.3 Portfolio Risk

```text
portfolio\\\\\\\_volatility = sqrt(w' \\\\\\\* Sigma \\\\\\\* w)
```

### 14.4 Sharpe Ratio

```text
Sharpe = (portfolio\\\\\\\_return - risk\\\\\\\_free\\\\\\\_rate) / portfolio\\\\\\\_volatility
```

### 14.5 Max-Sharpe Objective

```text
maximize (w' \\\\\\\* mu - rf) / sqrt(w' \\\\\\\* Sigma \\\\\\\* w)
```

subject to:

```text
sum(w\\\\\\\_i) = 1
min\\\\\\\_weight <= w\\\\\\\_i <= max\\\\\\\_weight
optional target return constraint
```

### 14.6 Intuition

Mean-Variance asks:

```text
Which combination gives the best expected return for the risk taken?
```

### 14.7 Pros

* Theoretically powerful.
* Directly uses expected return and covariance.
* Can create efficient frontier portfolios.

### 14.8 Cons

* Highly sensitive to expected-return estimation.
* Can produce unstable or concentrated weights.
* Small changes in input assumptions can lead to large allocation changes.

### 14.9 Role in This Project

Mean-Variance / Max-Sharpe is implemented, but it is not central to the dashboard/sensitivity experiments. The main comparative emphasis is on Equal Weight, Inverse Volatility, HRP, HERC, covariance methods, rebalancing rules, transaction costs, and volatility targeting.

\---

## 15\. Strategy 4 — Hierarchical Risk Parity (HRP)

### 15.1 What HRP Tries to Solve

Traditional optimization can become unstable when expected returns are noisy or covariance matrices are ill-conditioned.

HRP avoids direct expected-return forecasting. It uses the covariance/correlation structure to group assets and allocate risk more robustly.

### 15.2 HRP Steps

```text
1. Estimate covariance matrix.
2. Convert covariance to correlation.
3. Convert correlation to distance.
4. Build hierarchical clustering tree.
5. Reorder assets using quasi-diagonalization.
6. Recursively split assets into clusters.
7. Allocate capital inversely to cluster variance.
```

### 15.3 Cluster Variance

Formula:

```text
cluster\\\\\\\_variance\\\\\\\_C = w\\\\\\\_ivp,C' \\\\\\\* Sigma\\\\\\\_C \\\\\\\* w\\\\\\\_ivp,C
```

where:

```text
C          = cluster
Sigma\\\\\\\_C    = covariance matrix inside cluster
w\\\\\\\_ivp,C    = inverse-variance portfolio weights inside cluster
```

### 15.4 Recursive Allocation Formula

When HRP splits a cluster into left and right sub-clusters:

```text
allocation\\\\\\\_to\\\\\\\_left = variance\\\\\\\_right / (variance\\\\\\\_left + variance\\\\\\\_right)
allocation\\\\\\\_to\\\\\\\_right = variance\\\\\\\_left / (variance\\\\\\\_left + variance\\\\\\\_right)
```

### 15.5 Intuition

If the left cluster is riskier, it gets less capital. If the right cluster is less risky, it gets more capital.

Example:

```text
variance\\\\\\\_left = 0.04
variance\\\\\\\_right = 0.01
```

Then:

```text
allocation\\\\\\\_to\\\\\\\_left = 0.01 / (0.04 + 0.01) = 20%
allocation\\\\\\\_to\\\\\\\_right = 0.04 / (0.04 + 0.01) = 80%
```

The lower-risk cluster receives higher allocation.

### 15.6 Why HRP Is Useful

HRP is useful when:

* expected returns are unreliable,
* covariance structure is meaningful,
* we want more stable allocations,
* we want to avoid concentration in correlated clusters.

\---

## 16\. Strategy 5 — Hierarchical Equal Risk Contribution (HERC)

### 16.1 What HERC Adds

HERC is another hierarchical allocation strategy. Like HRP, it uses clustering. But HERC explicitly tries to equalize risk contribution across hierarchical branches.

### 16.2 Difference Between HRP and HERC

|Aspect|HRP|HERC|
|-|-|-|
|Tree use|Uses quasi-diagonal ordering|Traverses explicit linkage tree|
|Split logic|Recursive bisection|Sibling branch risk allocation|
|Allocation basis|Cluster variance|Cluster risk|
|Main idea|Reduce concentration by cluster variance|Equalize risk contribution across branches|

### 16.3 HERC Cluster Risk

Formula:

```text
cluster\\\\\\\_risk\\\\\\\_C = sqrt(w\\\\\\\_iv,C' \\\\\\\* Sigma\\\\\\\_C \\\\\\\* w\\\\\\\_iv,C)
```

### 16.4 HERC Branch Allocation

```text
weight\\\\\\\_left = parent\\\\\\\_weight \\\\\\\* risk\\\\\\\_right / (risk\\\\\\\_left + risk\\\\\\\_right)
weight\\\\\\\_right = parent\\\\\\\_weight \\\\\\\* risk\\\\\\\_left / (risk\\\\\\\_left + risk\\\\\\\_right)
```

### 16.5 Intuition

If one branch is riskier, it receives less capital. The goal is to balance the risk contribution of branches, not merely capital weights.

### 16.6 Why HERC Is Useful

HERC is useful when the project wants to compare different hierarchical risk-budgeting philosophies.

HRP and HERC can produce different weights even with the same covariance estimator and same asset universe.

\---

## 17\. Stage 7 — Rolling Walk-Forward Backtesting

### 17.1 Why Backtesting?

A strategy can look good in static weights but fail over time. Backtesting asks:

```text
If we had used this strategy historically, how would it have performed?
```

### 17.2 Rolling Walk-Forward Logic

Default training window:

```text
252 trading days
```

Core simulation:

```text
For each date t after the training window:
    Use past 252 trading days as training data
    Estimate covariance and strategy weights
    Apply weights to next-period returns
    Update portfolio value
    Record returns, drawdowns, weights, turnover, and costs
```

### 17.3 Portfolio Return Formula

```text
r\\\\\\\_p,t = w\\\\\\\_{t-1}' \\\\\\\* r\\\\\\\_t
```

where:

```text
w\\\\\\\_{t-1} = weights decided before return at time t
r\\\\\\\_t     = asset returns at time t
```

### 17.4 Why This Avoids Look-Ahead Bias

The strategy must only use information available before the return occurs.

Correct:

```text
Use data up to yesterday to decide today’s weights.
```

Wrong:

```text
Use today’s return to decide today’s weights.
```

The second method creates look-ahead bias and gives unrealistically good results.

\---

## 18\. Stage 8 — Rebalancing Logic

### 18.1 Why Rebalancing Is Needed

Portfolio weights drift over time because assets earn different returns.

Example:

Initial weights:

```text
Stock A = 50%
Stock B = 50%
```

If Stock A rises sharply and Stock B falls, the portfolio may become:

```text
Stock A = 60%
Stock B = 40%
```

Rebalancing brings the portfolio back toward target weights.

### 18.2 Rebalance Modes

The project supports:

```text
calendar
threshold
calendar\\\\\\\_or\\\\\\\_threshold
```

### 18.3 Calendar Rebalancing

Rebalance at fixed dates, such as monthly.

Logic:

```text
If date is rebalance date:
    rebalance
```

Pros:

* Simple.
* Predictable.

Cons:

* May trade even when weights have barely changed.
* May miss large drift between rebalance dates.

### 18.4 Threshold Rebalancing

Rebalance only when weights drift too far from target.

Formula:

```text
max(abs(current\\\\\\\_weights - target\\\\\\\_weights)) >= threshold
```

Example:

```text
target HDFCBANK weight = 10%
current HDFCBANK weight = 16%
drift = 6%
threshold = 5%
```

Since 6% > 5%, rebalance is triggered.

### 18.5 Calendar-or-Threshold Rebalancing

Rebalance if either condition is satisfied:

```text
calendar date reached OR threshold breached
```

### 18.6 Threshold Values

Main tested/default thresholds:

```text
0.03, 0.05, 0.10
```

Dashboard slider allows values up to:

```text
0.20
```

### 18.7 Important Patch: Target Update vs Trigger

Earlier, threshold rebalancing could mix two effects:

1. natural weight drift due to returns, and
2. optimizer target changes.

The current design separates:

```text
target\\\\\\\_update\\\\\\\_frequency = monthly
rebalance\\\\\\\_trigger = calendar / threshold / calendar\\\\\\\_or\\\\\\\_threshold
```

This makes threshold rebalancing more interpretable.

\---

## 19\. Stage 9 — Turnover and Transaction Costs

### 19.1 Turnover

Turnover measures how much of the portfolio is traded during rebalancing.

Formula:

```text
turnover = 0.5 \\\\\\\* sum(abs(target\\\\\\\_weights - current\\\\\\\_weights))
```

Why multiply by 0.5?

Because buying one asset usually means selling another. Without 0.5, the same reallocation can be double-counted.

Example:

Current weights:

```text
A = 60%, B = 40%
```

Target weights:

```text
A = 50%, B = 50%
```

Absolute differences:

```text
A = 10%, B = 10%
Sum = 20%
Turnover = 0.5 \\\\\\\* 20% = 10%
```

### 19.2 Transaction Cost

Default cost assumptions:

```text
base cost = 10 bps
slippage = 5 bps
total = 15 bps
```

Conversion:

```text
1 bps = 0.01%
15 bps = 0.15% = 0.0015
```

Formula:

```text
cost\\\\\\\_rate = (base\\\\\\\_bps + slippage\\\\\\\_bps) / 10000
transaction\\\\\\\_cost = turnover \\\\\\\* portfolio\\\\\\\_value \\\\\\\* cost\\\\\\\_rate
```

Example:

```text
portfolio value = ₹10,00,000
turnover = 10% = 0.10
cost rate = 15 bps = 0.0015
transaction cost = 0.10 \\\\\\\* 10,00,000 \\\\\\\* 0.0015 = ₹150
```

### 19.3 Gross vs Net Portfolio Value

The backtester tracks both:

```text
gross portfolio value = before transaction costs
net portfolio value   = after transaction costs
```

Important precision:

Transaction costs reduce net portfolio value/final value. However, return-stream metrics are not fully cost-adjusted daily returns in every part of the code. So while final value and cost diagnostics reflect costs, be careful when interpreting every metric as fully net-of-cost daily performance.

Correct statement:

> The system tracks gross and net values, and transaction costs reduce net portfolio value. Cost diagnostics are included, but return-stream metrics should be interpreted carefully because they are not always fully reconstructed as daily net returns.

\---

## 20\. Stage 10 — Performance Metrics

### 20.1 Cumulative Return

Formula:

```text
cumulative\\\\\\\_return = product(1 + r\\\\\\\_t) - 1
```

Example:

Returns:

```text
10%, -5%, 8%
```

Cumulative return:

```text
(1.10 \\\\\\\* 0.95 \\\\\\\* 1.08) - 1 = 0.1286 = 12.86%
```

### 20.2 CAGR

CAGR means Compound Annual Growth Rate.

Formula:

```text
CAGR = (final\\\\\\\_value / initial\\\\\\\_value)^(1 / years) - 1
```

Code-style daily formula:

```text
CAGR = product(1 + r\\\\\\\_t)^(252 / n) - 1
```

Interpretation:

CAGR answers:

```text
What annual growth rate would produce the same final value?
```

### 20.3 Annualized Volatility

```text
annualized\\\\\\\_volatility = std(r\\\\\\\_t) \\\\\\\* sqrt(252)
```

Interpretation:

Higher volatility means greater fluctuation in portfolio returns.

### 20.4 Sharpe Ratio

```text
Sharpe = mean(r\\\\\\\_t - rf / 252) / std(r\\\\\\\_t - rf / 252) \\\\\\\* sqrt(252)
```

Interpretation:

Sharpe measures excess return per unit of total volatility.

High Sharpe means the strategy generated good return relative to its volatility.

### 20.5 Sortino Ratio

```text
Sortino = mean(r\\\\\\\_t - target / 252) / downside\\\\\\\_deviation \\\\\\\* sqrt(252)
```

Sortino penalizes downside volatility only.

Interpretation:

If two strategies have the same volatility, but one has more upside volatility and less downside volatility, Sortino may prefer that strategy.

### 20.6 Drawdown

Drawdown measures the fall from a previous peak.

Formula:

```text
drawdown\\\\\\\_t = portfolio\\\\\\\_value\\\\\\\_t / running\\\\\\\_max(portfolio\\\\\\\_value)\\\\\\\_t - 1
```

Maximum drawdown:

```text
max\\\\\\\_drawdown = min(drawdown\\\\\\\_t)
```

Example:

Portfolio rises to ₹12,00,000 and then falls to ₹9,00,000:

```text
drawdown = 9,00,000 / 12,00,000 - 1 = -25%
```

### 20.7 Calmar Ratio

```text
Calmar = CAGR / abs(Max Drawdown)
```

Interpretation:

Calmar measures growth relative to worst drawdown.

Example:

```text
CAGR = 15%
Max Drawdown = -10%
Calmar = 15% / 10% = 1.5
```

A strategy with high CAGR but very deep drawdown may have a lower Calmar ratio.

### 20.8 VaR and CVaR

Historical VaR:

```text
VaR\\\\\\\_95 = 5th percentile of returns
```

CVaR:

```text
CVaR\\\\\\\_95 = mean(returns <= VaR\\\\\\\_95)
```

Interpretation:

VaR estimates a bad threshold loss. CVaR estimates the average loss conditional on being in the bad tail.

These are historical descriptive metrics, not fully forward-looking stress tests.

\---

## 21\. Stage 11 — Risk Contribution Analytics

### 21.1 Why Risk Contribution?

Portfolio weights do not tell the full risk story.

Example:

An asset may have only 5% capital weight but contribute 20% of portfolio risk if it is very volatile and highly correlated with the portfolio.

Risk contribution analytics answer:

```text
Which assets are actually driving portfolio risk?
```

### 21.2 Portfolio Volatility

```text
portfolio\\\\\\\_volatility = sqrt(w' \\\\\\\* Sigma \\\\\\\* w)
```

### 21.3 Marginal Risk Contribution (MRC)

```text
MRC = Sigma \\\\\\\* w / portfolio\\\\\\\_volatility
```

MRC means how much portfolio risk changes if we increase an asset’s weight slightly.

### 21.4 Total Risk Contribution (TRC)

```text
TRC\\\\\\\_i = w\\\\\\\_i \\\\\\\* MRC\\\\\\\_i
```

TRC is the actual risk contribution of asset `i`.

### 21.5 Percentage Risk Contribution (PRC)

```text
PRC\\\\\\\_i = TRC\\\\\\\_i / portfolio\\\\\\\_volatility
```

PRC shows each asset’s share of total portfolio risk.

### 21.6 Example

If an asset has:

```text
capital weight = 10%
percentage risk contribution = 25%
```

then the asset is risk-heavy relative to its capital allocation.

### 21.7 Why This Helps in Viva

This is an explanation layer. It lets us say not only which strategy performed better, but why its risk behaved differently.

\---

## 22\. Stage 12 — Benchmark Comparison

### 22.1 Default Benchmark

The default benchmark is:

```text
Equal Weight
```

Other compared strategies include:

```text
Inverse Volatility
HRP
HERC
```

### 22.2 Benchmark-Relative Metrics

The framework can compare strategy metrics against benchmark metrics.

Examples:

```text
excess\\\\\\\_cagr = strategy\\\\\\\_cagr - benchmark\\\\\\\_cagr
excess\\\\\\\_sharpe = strategy\\\\\\\_sharpe - benchmark\\\\\\\_sharpe
drawdown\\\\\\\_difference = strategy\\\\\\\_max\\\\\\\_drawdown - benchmark\\\\\\\_max\\\\\\\_drawdown
volatility\\\\\\\_difference = strategy\\\\\\\_volatility - benchmark\\\\\\\_volatility
final\\\\\\\_value\\\\\\\_difference = strategy\\\\\\\_final\\\\\\\_value - benchmark\\\\\\\_final\\\\\\\_value
```

### 22.3 Why Benchmarking Is Important

Without a benchmark, we only know whether a strategy made money. With a benchmark, we know whether the complexity was useful.

Good viva line:

> We use Equal Weight as a simple but strong benchmark because any sophisticated strategy should justify its complexity by improving risk-adjusted performance, drawdown, turnover, or robustness.

\---

## 23\. Stage 13 — Volatility Targeting Overlay

### 23.1 What Volatility Targeting Does

The base allocator decides the composition of the risky portfolio.

Volatility targeting decides:

```text
How much of the portfolio should be exposed to the risky strategy?
How much should be shifted into a defensive sleeve?
```

### 23.2 Targeted Return Formula

```text
r\\\\\\\_targeted,t = exposure\\\\\\\_t \\\\\\\* risky\\\\\\\_strategy\\\\\\\_return\\\\\\\_t
               + (1 - exposure\\\\\\\_t) \\\\\\\* defensive\\\\\\\_asset\\\\\\\_return\\\\\\\_t
```

Interpretation:

If exposure is 70%, then:

```text
70% goes into risky strategy
30% goes into defensive asset
```

### 23.3 Realized Volatility

```text
realized\\\\\\\_vol\\\\\\\_t = std(risky\\\\\\\_returns\\\\\\\_{t-window:t}) \\\\\\\* sqrt(252)
```

### 23.4 Exposure Formula

```text
exposure\\\\\\\_t = clip(target\\\\\\\_vol\\\\\\\_t / realized\\\\\\\_vol\\\\\\\_{t-1}, exposure\\\\\\\_floor, exposure\\\\\\\_cap)
```

Interpretation:

* If realized volatility is higher than target, reduce exposure.
* If realized volatility is lower than target, increase exposure.

Example:

```text
target volatility = 10%
realized volatility = 20%
exposure = 10% / 20% = 50%
```

So the model holds only 50% risky exposure.

### 23.5 Adaptive Regime Targets

Default base target volatility:

```text
10%
```

Adaptive regime targets:

|Regime|Percentile rule|Target volatility|
|-|-|-:|
|Calm|percentile <= 40%|12%|
|Normal|40% < percentile <= 80%|10%|
|Stress|80% < percentile <= 95%|6%|
|Crisis|percentile > 95%|3%|

Important: These are rule-based realized-volatility regimes, not full Markov-switching macro regimes.

### 23.6 Defensive Asset

Default experiment defensive sleeve:

```text
Synthetic Risk-Free
```

Dashboard preferred defensive ticker:

```text
LIQUIDBEES.NS
```

Fallback ticker:

```text
LIQUIDETF.NS
```

If live defensive ticker data is unavailable, the system uses synthetic risk-free returns.

### 23.7 No-Look-Ahead Logic

This is critical.

The system must not use future information to set today’s exposure.

Safeguards:

```text
realized volatility is shifted by one day
regime classification uses lagged realized volatility
exposure for day t uses information only through t-1
```

Correct:

```text
Use volatility observed until yesterday to decide today’s exposure.
```

Wrong:

```text
Use today’s realized return to decide today’s exposure.
```

### 23.8 Role in Project

Volatility targeting adds adaptive risk scaling without changing the underlying allocation method.

Good explanation:

> HRP/HERC decide what risky basket to hold. Volatility targeting decides how much of that risky basket to hold.

\---

## 24\. Stage 14 — Experiment Sensitivity Framework

### 24.1 What the Experiment Framework Does

The experiment framework runs multiple strategy configurations and ranks them by a selected objective metric.

It can vary:

```text
strategy
covariance method
rebalance mode
threshold
transaction cost bps
slippage bps
volatility targeting enabled/disabled
target volatility
defensive asset
training window
initial capital
```

### 24.2 Ranking Objectives

Dashboard objectives:

```text
CAGR
Sharpe
Sortino
Calmar
Max Drawdown
Final Value
```

Default objective:

```text
Calmar
```

Reason:

The dashboard selectbox uses `index=3`, and `Calmar` is the fourth option.

Code reference:

```text
src/dashboard/app.py
```

### 24.3 Default Ranking Logic

The ranking is **single-objective**, not multi-objective.

That means the code does not apply weights such as:

```text
40% Sharpe + 30% CAGR + 20% drawdown + 10% turnover
```

Instead, it ranks by one selected metric.

Default:

```text
best strategy = highest Calmar
```

Formula:

```text
Calmar = CAGR / abs(Max Drawdown)
```

Interpretation:

The default ranking favors strategies that produce good growth while keeping drawdowns small.

### 24.4 Important Detail About Max Drawdown Sorting

The code sorts objectives descending, including `max\\\\\\\_drawdown`.

Because max drawdown is negative, less negative is better.

Example:

```text
-0.08 is better than -0.25
```

Descending sort correctly ranks:

```text
-0.08 above -0.25
```

Code reference:

```text
src/experiments/sensitivity.py
```

### 24.5 Diagnostic Metrics Recorded

Experiment records may include:

```text
total\\\\\\\_turnover
average\\\\\\\_turnover
total\\\\\\\_transaction\\\\\\\_cost
number\\\\\\\_of\\\\\\\_rebalances
```

These are not directly used in ranking unless they affect the selected objective through net portfolio value or returns.

Correct explanation:

> Turnover and transaction cost are recorded as diagnostics. The current ranking does not explicitly penalize turnover by a fixed weight. A future extension can implement multi-objective ranking.

### 24.6 Result Table Template

Use this table after running verified experiments. Do not fill it with unverified results.

|Strategy|Covariance|Rebalance Mode|Threshold|Vol Target?|CAGR|Volatility|Sharpe|Sortino|Calmar|Max Drawdown|Final Value|Turnover|Transaction Cost|Rebalances|
|-|-|-|-:|-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|Example|Example|Example|Example|Example|—|—|—|—|—|—|—|—|—|—|

### 24.7 Future Enhancement: Multi-Objective Decision Model

A stronger version could compute:

```text
score = 0.30 \\\\\\\* normalized\\\\\\\_Calmar
      + 0.25 \\\\\\\* normalized\\\\\\\_Sharpe
      + 0.20 \\\\\\\* normalized\\\\\\\_CAGR
      + 0.15 \\\\\\\* normalized\\\\\\\_drawdown\\\\\\\_score
      + 0.10 \\\\\\\* normalized\\\\\\\_cost\\\\\\\_score
```

But this is not current behavior.

\---

## 25\. Dashboard Workflow

### 25.1 What the Dashboard Allows

The Streamlit dashboard allows users to:

1. Select an asset universe.
2. Override tickers manually.
3. Select date range.
4. Choose strategy.
5. Choose covariance method.
6. Choose rebalance rule.
7. Configure transaction costs and slippage.
8. Enable/disable volatility targeting.
9. Select defensive sleeve.
10. Run backtest.
11. View plots and metrics.
12. Run sensitivity experiments.

### 25.2 Main Outputs

Expected dashboard outputs include:

```text
growth curves
drawdown curves
performance metric tables
portfolio weights
correlation heatmap
dendrogram
turnover diagnostics
transaction-cost summary
benchmark comparison
sensitivity ranking table
```

### 25.3 Code Reference

```text
src/dashboard/app.py
```

Relevant dashboard concepts:

```text
asset presets
manual ticker override
objective selectbox
sensitivity experiment panel
validation for dates/assets/exposure/thresholds
```

\---

## 26\. Module-Wise Understanding

### 26.1 Data Module

Responsibility:

```text
Download prices and volume
Return adjusted close price panel
Inspect missingness and metadata
```

Possible location:

```text
src/data/
```

### 26.2 Preprocessing Module

Responsibility:

```text
Clean prices
Handle missing values
Detect anomalies
Compute simple/log returns
Winsorize extreme returns
```

Possible location:

```text
src/preprocessing/
```

### 26.3 Risk/Covariance Module

Responsibility:

```text
Compute volatility
Compute covariance
Compute correlation
Compute distance matrix
Support multiple covariance estimators
```

Possible location:

```text
src/risk/
```

### 26.4 Clustering Module

Responsibility:

```text
Build linkage matrix
Assign clusters
Create dendrograms
Provide hierarchy for HRP/HERC
```

Possible location:

```text
src/clustering/
```

### 26.5 Allocators Module

Responsibility:

```text
Equal Weight
Inverse Volatility
Mean-Variance / Max-Sharpe
HRP
HERC
```

Possible location:

```text
src/portfolio/
src/allocators/
```

### 26.6 Backtesting Module

Responsibility:

```text
Run rolling walk-forward simulations
Apply rebalance rules
Track weights and returns
Track gross/net values
Track transaction costs and turnover
```

Possible location:

```text
src/backtesting/
```

### 26.7 Analytics Module

Responsibility:

```text
Calculate CAGR, Sharpe, Sortino, Calmar
Calculate drawdown, VaR, CVaR
Calculate risk contribution
Compare benchmarks
Generate plots
```

Possible location:

```text
src/analytics/
```

### 26.8 Experiments Module

Responsibility:

```text
Run parameter grids
Rank experiments by selected objective
Summarize sensitivity by parameter
```

Important reference:

```text
src/experiments/sensitivity.py
```

### 26.9 Dashboard Module

Responsibility:

```text
Streamlit app
User controls
Validation
Plots and tables
Sensitivity UI
```

Reference:

```text
src/dashboard/app.py
```

\---

## 27\. Key Assumptions Across the Project

### 27.1 Data Assumptions

```text
Adjusted close is preferred.
Daily data is the base frequency.
Yahoo Finance is the main source.
Data is downloaded at runtime.
Manual ticker override is allowed.
```

### 27.2 Annualization Assumption

```text
252 trading days per year
```

Used for:

```text
annualized volatility
CAGR from daily returns
Sharpe ratio
Sortino ratio
rolling volatility
```

### 27.3 Portfolio Assumptions

```text
Long-only
Fully invested in risky universe unless volatility targeting is enabled
No short selling
No leverage assumption unless exposure cap permits it in overlay configuration
Defensive sleeve excluded from risky covariance/clustering
```

### 27.4 Risk Metric Assumptions

```text
Volatility = standard deviation of returns
VaR/CVaR = historical, not parametric normal
Drawdown = peak-to-trough portfolio decline
```

### 27.5 Backtesting Assumptions

```text
Training window = 252 trading days
Rebalancing can be calendar, threshold, or both
Transaction cost model is linear in turnover
Default cost = 10 bps base + 5 bps slippage
Gross and net values tracked
```

\---

## 28\. What Makes the Project Quantitative?

This project is quantitative because decisions are made using measurable variables, formulas, and repeatable rules rather than subjective opinion.

Examples:

|Project Component|Quantitative Basis|
|-|-|
|Returns|Price ratios/log returns|
|Volatility|Standard deviation|
|Covariance|Co-movement of asset returns|
|Correlation|Standardized covariance|
|Clustering|Distance matrix from correlations|
|HRP|Recursive allocation by cluster variance|
|HERC|Recursive allocation by branch risk|
|Backtesting|Historical simulation using rules|
|Transaction costs|Turnover × cost rate|
|Vol targeting|Target vol / realized vol|
|Ranking|Objective metric such as Calmar|

Good viva line:

> The project is quantitative because every allocation and ranking decision is generated by explicit mathematical rules applied to market data, not by discretionary stock picking.

\---

## 29\. How to Explain the Project in 60 Seconds

Use this in viva or presentation:

> Our project, Adaptive Portfolio Risk Analytics, is a risk-aware portfolio analytics platform for Indian assets. It starts by downloading adjusted price data from Yahoo Finance, cleaning it, and converting prices into simple and log returns. Then it estimates risk through volatility, covariance, correlation, and correlation distance. Using this structure, it builds portfolios through Equal Weight, Inverse Volatility, Mean-Variance, HRP, and HERC. The strategies are evaluated using rolling walk-forward backtests with rebalancing rules, turnover, transaction costs, drawdowns, and benchmark-relative metrics. We also added risk-contribution analytics to explain which assets drive portfolio risk, and a volatility-targeting overlay that dynamically shifts exposure between the risky strategy and a defensive sleeve based on lagged realized volatility. Finally, the dashboard supports experiment sensitivity analysis, where strategies can be ranked by a selected objective such as Calmar. The current system is a strong research prototype; full Markov regimes, NLP sentiment, CPCV, and production deployment are future work.

\---

## 30\. Likely Viva Questions and Strong Answers

### Q1. What is the main objective of your project?

**Answer:**  
The objective is to build a risk-aware portfolio analytics platform that compares portfolio construction strategies under different covariance estimators, rebalancing rules, transaction-cost assumptions, and volatility-targeting overlays. It helps evaluate not only returns but also drawdowns, risk contribution, turnover, and robustness.

### Q2. Why did you use adjusted close prices?

**Answer:**  
Adjusted close accounts for corporate actions like dividends, splits, and bonuses. If we use raw close prices, returns may show artificial jumps around corporate actions, which would distort volatility, covariance, and portfolio weights.

### Q3. Why do you calculate returns instead of using prices directly?

**Answer:**  
Prices are not comparable across assets because each asset has a different price scale. Returns measure percentage change, making assets comparable for risk and performance analysis.

### Q4. What is the difference between covariance and correlation?

**Answer:**  
Covariance measures co-movement in return units, while correlation standardizes covariance between -1 and +1. Correlation is easier to interpret and is used to convert asset relationships into distances for clustering.

### Q5. Why is covariance important for portfolio risk?

**Answer:**  
Portfolio risk depends not only on individual asset volatility but also on how assets move together. Even if individual assets are risky, the portfolio can be diversified if their correlations are low or negative.

### Q6. Why did you include multiple covariance estimators?

**Answer:**  
Sample covariance can be noisy. Ledoit-Wolf improves stability through shrinkage. EWMA gives more importance to recent observations. EWMA + Ledoit-Wolf combines recency sensitivity with regularization. Testing all four helps us understand sensitivity to risk-estimation assumptions.

### Q7. What is HRP?

**Answer:**  
HRP stands for Hierarchical Risk Parity. It clusters assets based on correlation distance and allocates capital recursively based on cluster variance. It avoids direct expected-return forecasting and is usually more stable than unconstrained mean-variance optimization.

### Q8. What is HERC?

**Answer:**  
HERC stands for Hierarchical Equal Risk Contribution. It also uses hierarchical clustering, but instead of only recursive bisection by cluster variance, it traverses the tree and allocates capital to equalize risk contribution across sibling branches.

### Q9. Difference between HRP and HERC?

**Answer:**  
HRP allocates recursively using cluster variance after quasi-diagonal ordering. HERC uses the explicit hierarchy and allocates sibling branch weights based on branch risk, aiming for more direct hierarchical risk budgeting.

### Q10. Why is Equal Weight used as benchmark?

**Answer:**  
Equal Weight is simple, transparent, and hard to overfit. Any sophisticated strategy should justify its complexity by outperforming or improving risk-adjusted metrics relative to Equal Weight.

### Q11. What does volatility targeting do?

**Answer:**  
Volatility targeting adjusts exposure to the risky strategy based on lagged realized volatility. When realized volatility is high, exposure is reduced and more capital moves into a defensive sleeve. When realized volatility is low, exposure can increase.

### Q12. How do you avoid look-ahead bias in volatility targeting?

**Answer:**  
The realized volatility signal is shifted by one day. Exposure at day `t` is based only on information available up to day `t-1`. Regime classification also uses lagged realized volatility.

### Q13. What is Calmar ratio and why is it default?

**Answer:**  
Calmar is CAGR divided by absolute maximum drawdown. It rewards strategies that generate growth while controlling worst peak-to-trough loss. It is the default dashboard objective because the objective selectbox defaults to the fourth option, which is Calmar.

### Q14. Are multiple attributes weighted to choose the best strategy?

**Answer:**  
No. The current experiment framework uses single-objective ranking. The user selects one objective such as CAGR, Sharpe, Sortino, Calmar, Max Drawdown, or Final Value. Turnover and transaction costs are recorded as diagnostics but are not explicitly weighted in the ranking unless they affect the selected objective.

### Q15. Is the system production-ready?

**Answer:**  
No. It is a research prototype. It has strong portfolio analytics and backtesting functionality, but production readiness would require data governance, persistent data snapshots, monitoring, deployment, liquidity-aware costs, market impact modeling, and stronger validation like CPCV.

### Q16. Is regime detection implemented?

**Answer:**  
Full Markov-switching regime detection is future work. The current volatility-targeting overlay uses realized-volatility percentile regimes, which are rule-based and lagged, but not a full macro/Markov regime model.

### Q17. Is NLP sentiment implemented?

**Answer:**  
No, not as a full working pipeline. NLP for RBI policy text, earnings calls, and uncertainty scoring is future work or scaffolding. We should not overclaim it.

### Q18. What is the most important limitation?

**Answer:**  
The biggest limitation is that the project is still a research prototype. It relies on Yahoo Finance data, uses simplified transaction costs, does not yet include CPCV, does not model market impact/taxes, and does not yet implement full regime or NLP-driven allocation.

### Q19. What is the strongest part of the project?

**Answer:**  
The strongest part is the end-to-end research workflow: data cleaning, covariance estimation, hierarchical allocation, rolling backtesting, transaction costs, risk contribution, benchmark comparison, volatility targeting, and sensitivity experiments are connected into one platform.

### Q20. What would you improve next?

**Answer:**  
First, add CPCV/walk-forward robustness validation to reduce overfitting risk. Then use that validation layer to test regime detection and NLP sentiment signals. This order keeps the research scientifically defensible.

\---

## 31\. Common Mistakes to Avoid While Presenting

### Mistake 1: Saying It Is Fully AI-Based

Avoid:

```text
This is a full AI-powered trading system.
```

Say:

```text
This is a quantitative portfolio analytics and backtesting platform with planned AI/NLP extensions.
```

### Mistake 2: Saying Regime Detection Is Complete

Avoid:

```text
We implemented Markov-switching regimes.
```

Say:

```text
The current volatility-targeting overlay uses rule-based realized-volatility regimes. Full Markov-switching regimes are future work.
```

### Mistake 3: Saying Turnover Is Weighted in Ranking

Avoid:

```text
The best strategy is selected using weighted turnover, CAGR, Sharpe, and drawdown.
```

Say:

```text
The current ranking is single-objective. Turnover and transaction costs are diagnostics, not explicitly weighted decision attributes.
```

### Mistake 4: Saying All Metrics Are Fully Net of Costs

Avoid:

```text
Every metric is fully daily net-of-cost.
```

Say:

```text
Gross and net values are tracked, and costs reduce net final value. Some return-stream metrics should be interpreted carefully because daily net-return reconstruction is not fully applied everywhere.
```

### Mistake 5: Claiming a Final Winning Strategy Without Verified Export

Avoid:

```text
This strategy is definitely the best.
```

Say:

```text
The framework ranks strategies by the selected objective. We should state the winner only after verifying the specific experiment run and export.
```

\---

## 32\. Formula Sheet

### Returns

```text
simple\\\\\\\_return\\\\\\\_t = P\\\\\\\_t / P\\\\\\\_{t-1} - 1
log\\\\\\\_return\\\\\\\_t = log(P\\\\\\\_t / P\\\\\\\_{t-1})
```

### Annualized Volatility

```text
annualized\\\\\\\_volatility = std(daily\\\\\\\_returns) \\\\\\\* sqrt(252)
```

### Covariance

```text
Sigma\\\\\\\_ij = cov(r\\\\\\\_i, r\\\\\\\_j)
```

### Correlation

```text
rho\\\\\\\_ij = Sigma\\\\\\\_ij / (sigma\\\\\\\_i \\\\\\\* sigma\\\\\\\_j)
```

### Correlation Distance

```text
d\\\\\\\_ij = sqrt((1 - rho\\\\\\\_ij) / 2)
```

### Portfolio Return

```text
r\\\\\\\_p,t = w\\\\\\\_{t-1}' \\\\\\\* r\\\\\\\_t
```

### Portfolio Variance

```text
portfolio\\\\\\\_variance = w' \\\\\\\* Sigma \\\\\\\* w
```

### Portfolio Volatility

```text
portfolio\\\\\\\_volatility = sqrt(w' \\\\\\\* Sigma \\\\\\\* w)
```

### Equal Weight

```text
w\\\\\\\_i = 1 / N
```

### Inverse Volatility

```text
w\\\\\\\_i = (1 / sigma\\\\\\\_i) / sum\\\\\\\_j(1 / sigma\\\\\\\_j)
```

### Max-Sharpe Objective

```text
maximize (w' \\\\\\\* mu - rf) / sqrt(w' \\\\\\\* Sigma \\\\\\\* w)
```

### HRP Cluster Variance

```text
cluster\\\\\\\_variance\\\\\\\_C = w\\\\\\\_ivp,C' \\\\\\\* Sigma\\\\\\\_C \\\\\\\* w\\\\\\\_ivp,C
```

### HRP Allocation

```text
allocation\\\\\\\_to\\\\\\\_left = variance\\\\\\\_right / (variance\\\\\\\_left + variance\\\\\\\_right)
allocation\\\\\\\_to\\\\\\\_right = variance\\\\\\\_left / (variance\\\\\\\_left + variance\\\\\\\_right)
```

### HERC Cluster Risk

```text
cluster\\\\\\\_risk\\\\\\\_C = sqrt(w\\\\\\\_iv,C' \\\\\\\* Sigma\\\\\\\_C \\\\\\\* w\\\\\\\_iv,C)
```

### HERC Allocation

```text
weight\\\\\\\_left = parent\\\\\\\_weight \\\\\\\* risk\\\\\\\_right / (risk\\\\\\\_left + risk\\\\\\\_right)
weight\\\\\\\_right = parent\\\\\\\_weight \\\\\\\* risk\\\\\\\_left / (risk\\\\\\\_left + risk\\\\\\\_right)
```

### Cumulative Return

```text
cumulative\\\\\\\_return = product(1 + r\\\\\\\_t) - 1
```

### CAGR

```text
CAGR = product(1 + r\\\\\\\_t)^(252 / n) - 1
```

### Sharpe Ratio

```text
Sharpe = mean(r\\\\\\\_t - rf / 252) / std(r\\\\\\\_t - rf / 252) \\\\\\\* sqrt(252)
```

### Sortino Ratio

```text
Sortino = mean(r\\\\\\\_t - target / 252) / downside\\\\\\\_deviation \\\\\\\* sqrt(252)
```

### Drawdown

```text
drawdown\\\\\\\_t = portfolio\\\\\\\_value\\\\\\\_t / running\\\\\\\_max(portfolio\\\\\\\_value)\\\\\\\_t - 1
```

### Maximum Drawdown

```text
max\\\\\\\_drawdown = min(drawdown\\\\\\\_t)
```

### Calmar Ratio

```text
Calmar = CAGR / abs(Max Drawdown)
```

### VaR

```text
VaR\\\\\\\_95 = 5th percentile of returns
```

### CVaR

```text
CVaR\\\\\\\_95 = mean(returns <= VaR\\\\\\\_95)
```

### Turnover

```text
turnover = 0.5 \\\\\\\* sum(abs(target\\\\\\\_weights - current\\\\\\\_weights))
```

### Transaction Cost

```text
cost\\\\\\\_rate = (base\\\\\\\_bps + slippage\\\\\\\_bps) / 10000
transaction\\\\\\\_cost = turnover \\\\\\\* portfolio\\\\\\\_value \\\\\\\* cost\\\\\\\_rate
```

### Weight Drift

```text
new\\\\\\\_weight\\\\\\\_i = old\\\\\\\_weight\\\\\\\_i \\\\\\\* (1 + asset\\\\\\\_return\\\\\\\_i) / (1 + portfolio\\\\\\\_return)
```

### Threshold Rebalance Rule

```text
max(abs(current\\\\\\\_weights - target\\\\\\\_weights)) >= threshold
```

### Volatility Targeting Return

```text
r\\\\\\\_targeted,t = exposure\\\\\\\_t \\\\\\\* risky\\\\\\\_strategy\\\\\\\_return\\\\\\\_t
               + (1 - exposure\\\\\\\_t) \\\\\\\* defensive\\\\\\\_asset\\\\\\\_return\\\\\\\_t
```

### Volatility Targeting Exposure

```text
exposure\\\\\\\_t = clip(target\\\\\\\_vol\\\\\\\_t / realized\\\\\\\_vol\\\\\\\_{t-1}, exposure\\\\\\\_floor, exposure\\\\\\\_cap)
```

### Realized Volatility

```text
realized\\\\\\\_vol\\\\\\\_t = std(risky\\\\\\\_returns\\\\\\\_{t-window:t}) \\\\\\\* sqrt(252)
```

### EWMA Alpha

```text
alpha = 2 / (span + 1)
alpha = 2 / 253 ≈ 0.0079 when span = 252
```

### Ledoit-Wolf

```text
Sigma\\\\\\\_LW = (1 - delta) \\\\\\\* S + delta \\\\\\\* F
```

\---

## 33\. Final Team Understanding Checklist

Before presentation, every teammate should be able to answer:

1. What is the project trying to solve?
2. What is the asset universe?
3. Why do we use adjusted close?
4. Difference between simple and log returns.
5. What volatility measures.
6. Difference between covariance and correlation.
7. Why correlation is converted into distance.
8. What dendrograms show.
9. How Equal Weight works.
10. How Inverse Volatility works.
11. Why Mean-Variance can be unstable.
12. How HRP works.
13. How HERC differs from HRP.
14. What rolling walk-forward backtesting means.
15. What rebalancing is.
16. What turnover means.
17. How transaction costs are modeled.
18. Difference between gross and net portfolio value.
19. What CAGR, Sharpe, Sortino, Calmar, and max drawdown mean.
20. What risk contribution explains.
21. Why Equal Weight is the benchmark.
22. What volatility targeting does.
23. How look-ahead bias is avoided.
24. How sensitivity ranking works.
25. What is implemented vs future work.

\---

## 34\. Final Project Positioning

The best positioning is:

> Adaptive Portfolio Risk Analytics is a risk-aware portfolio research platform that compares hierarchical and baseline allocation strategies using robust covariance estimation, rolling backtesting, transaction-cost diagnostics, risk attribution, benchmark comparison, volatility targeting, and experiment sensitivity analysis.

What makes it strong:

```text
It is end-to-end.
It is modular.
It compares multiple strategies.
It includes realistic frictions.
It explains risk, not just return.
It has a dashboard.
It has a clear future research path.
```

What it is not yet:

```text
Not a production trading engine.
Not a full NLP sentiment model.
Not a full Markov-switching regime engine.
Not CPCV-validated yet.
Not liquidity/market-impact/tax aware yet.
```

Final viva line:

> The current project gives us a strong and defensible research infrastructure. The next step is not to add more complexity blindly, but to add CPCV and robustness validation first, then test regime and sentiment signals on top of that validation framework.

