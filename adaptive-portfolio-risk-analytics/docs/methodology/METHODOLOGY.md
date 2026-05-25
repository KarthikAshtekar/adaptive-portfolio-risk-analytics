# Methodology Overview

## Portfolio Optimization Approaches

### 1. Equal Weight (1/N)
- Allocates equal weight to all assets
- Baseline comparison
- No optimization required
- Surprisingly robust in many cases

### 2. Mean-Variance Optimization (Markowitz)
- Maximizes risk-adjusted returns (Sharpe ratio)
- Requires mean and covariance estimates
- Sensitive to estimation errors
- Prone to corner solutions and turnover

### 3. Hierarchical Risk Parity (HRP)
- Constructs portfolios through hierarchical clustering
- Addresses mean-variance limitations
- Reduces sensitivity to estimation errors
- More stable and lower turnover

### 4. Hierarchical Equal Risk Contribution (HERC)
- Extends HRP with equal risk contribution
- Balances risk within and across clusters
- Intuitive risk budgeting approach

## Covariance Estimation Methods

### Ledoit-Wolf Shrinkage
- Shrinks sample covariance toward target matrix
- Reduces condition number
- Optimal shrinkage intensity
- Prevents estimation error amplification

### Gerber Covariance (Rank-Sign)
- Uses rank and sign correlation
- Robust to outliers
- Less sensitive to non-normality
- Captures tail dependence

### Rolling Window
- Updates covariance dynamically
- Adapts to regime changes
- Requires minimal history
- Incorporates recent information

## Regime Detection

### Markov-Switching Models
- Identifies bull/bear market regimes
- Tracks volatility regimes
- Provides transition probabilities
- Enables dynamic allocation

### Volatility Targeting
- Adjusts leverage based on volatility
- Reduces portfolio volatility
- Increases risk-adjusted returns
- Responds to market conditions

### Defensive Risk Scaling
- Reduces positions in high-volatility regimes
- Increases cash allocation
- Limits downside risk
- Smooth transitions

## Sentiment and Macro Intelligence

### RBI Monetary Policy Sentiment
- Extracts from policy announcements
- Measures tightening/easing bias
- Incorporates in regime assessment
- Complements market-based signals

### Earnings Call Sentiment
- Management tone and guidance
- Forward-looking information
- Predicts earnings surprises
- Asset-specific signals

### Uncertainty Scoring
- Macro uncertainty quantification
- Risk premium indicator
- Portfolio risk adjustment
- Signal for defensive positioning

## Backtesting Framework

### Rolling Window Approach
- Walk-forward analysis
- Expanding/rolling training windows
- Out-of-sample testing
- Realistic simulation

### Combinatorial Purged Cross-Validation (CPCV)
- Addresses time-series bias
- Implements embargo periods
- Prevents look-ahead bias
- Multiple test partitions

### Transaction Costs
- Bid-ask spread impact
- Market impact from large trades
- Turnover-based costs
- Affects returns net of costs

## Risk Metrics

### Value-at-Risk (VaR)
- Probability of loss exceeding threshold
- Common risk measure
- Multiple calculation methods
- Depends on distribution assumptions

### Conditional Value-at-Risk (CVaR)
- Expected loss beyond VaR
- Coherent risk measure
- Captures tail risk
- More stable than VaR

### Maximum Drawdown
- Peak-to-trough decline
- Psychological importance
- Regime indicator
- Stress test baseline

### Sharpe Ratio
- Risk-adjusted return
- Return per unit of volatility
- Standard performance metric
- Assumes normal distribution

### Sortino Ratio
- Penalizes only downside volatility
- More relevant for downside risk
- Higher Sharpe alternatives
- Focuses on negative volatility

### Calmar Ratio
- Return to maximum drawdown
- Emphasizes recovery
- Long-term metric
- Less sensitive to volatility spikes

## Performance Attribution

### Return Attribution
- Allocation vs. selection effects
- Geographic vs. sector allocation
- Asset class contribution

### Risk Attribution
- Risk contribution by position
- Marginal risk metrics
- Portfolio risk decomposition

## Stress Testing

### Historical Scenarios
- 2008 Financial Crisis
- 2020 COVID Crash
- 1987 Black Monday
- 1998 LTCM Crisis

### Reverse Stress Testing
- Identify market moves causing losses
- Implied scenarios from constraints
- Risk appetite calibration

### Correlation Stress
- Estimate impact of correlation increase
- Portfolio resilience to shock
- Diversification benefit erosion
