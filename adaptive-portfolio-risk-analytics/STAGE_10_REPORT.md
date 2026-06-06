# Stage 10 Implementation Report: Risk Contribution Analytics

**Status**: Complete  
**Date**: 2026-06-06  
**Focus**: Add a research-grade risk attribution layer for portfolio weights, marginal risk, total risk, and percentage risk contribution

---

## 1. Files Created

- `src/analytics/risk_contribution.py`
- `tests/test_risk_contribution.py`
- `notebooks/10_risk_contribution/stage_10_risk_contribution.ipynb`
- `STAGE_10_REPORT.md`

## 2. Files Modified

- `src/analytics/__init__.py`
- `src/dashboard/plots.py`
- `src/dashboard/app.py`
- `tests/test_phase2a_integration.py`

---

## 3. Mathematical Intuition

Portfolio weights alone do not explain where risk comes from.

An asset can:

- receive a high capital weight but contribute modest risk if its volatility and covariance are low
- receive a low capital weight but still dominate risk if its volatility and correlation with the rest of the portfolio are high

This stage adds an attribution layer that decomposes total portfolio volatility into asset-level contributions.

---

## 4. Formula Explanation

Let:

- `w` = portfolio weights
- `Sigma` = covariance matrix

### Portfolio Volatility

```text
portfolio_vol = sqrt(w' Sigma w)
```

### Marginal Risk Contribution

```text
MRC = Sigma w / portfolio_vol
```

This measures how much portfolio volatility changes as each asset weight changes marginally.

### Total Risk Contribution

```text
TRC = w * MRC
```

This gives the absolute volatility contribution from each asset.

### Percentage Risk Contribution

```text
PRC = TRC / portfolio_vol
```

This normalizes total risk contribution so the components sum to approximately 1.

---

## 5. Risk Contribution Module

Implemented in:

- `src/analytics/risk_contribution.py`

Functions added:

- `portfolio_volatility()`
- `marginal_risk_contribution()`
- `total_risk_contribution()`
- `percentage_risk_contribution()`
- `risk_contribution_table()`
- `compare_risk_contributions()`

Input support:

- weights as `pd.Series`
- weights as `np.ndarray`
- covariance matrix as `pd.DataFrame`

Validation added for:

- square covariance matrices
- matching covariance labels
- no NaNs
- finite weights
- non-negative weights
- weights summing approximately to 1

---

## 6. Dashboard Additions

Added a new section to the Streamlit app:

- `Risk Contribution Analysis`

Displayed elements:

- risk contribution table
- percentage risk contribution bar chart
- grouped weight vs risk contribution chart

If the selected strategy is `HRP` or `HERC`, the dashboard also displays:

- HRP vs HERC percentage risk contribution comparison chart
- HRP vs HERC comparison table

These additions reuse the existing covariance matrix already computed in the dashboard flow.

---

## 7. Notebook Findings

Notebook:

- `notebooks/10_risk_contribution/stage_10_risk_contribution.ipynb`

Sections included:

1. Load data
2. Compute covariance using `CovarianceFactory`
3. Generate HRP weights
4. Generate HERC weights
5. Compute risk contribution tables
6. Compare weights vs risk contribution
7. Compare HRP vs HERC risk contribution
8. Interpret drawdown behavior

The notebook is designed to answer:

- Which assets receive high weight but low risk contribution?
- Which assets receive low weight but high risk contribution?
- Does HERC spread risk more evenly than HRP?
- Can percentage risk contribution help explain lower drawdowns?

---

## 8. Test Results

Executed:

```bash
.venv\Scripts\python.exe -m pytest tests\test_risk_contribution.py tests\test_phase2a_integration.py -q
.venv\Scripts\python.exe -m pytest tests\test_covariance_factory.py tests\test_herc_allocator.py tests\test_hrp.py tests\test_backtesting.py -q
```

Results:

- `37 passed` for risk contribution and integration coverage
- `41 passed` for covariance, HERC, HRP, and backtesting regression coverage

Covered checks include:

- positive portfolio volatility
- MRC/TRC/PRC shape and summation properties
- expected table columns
- `np.ndarray` weight acceptance
- negative weight validation
- mismatched label validation
- HRP vs HERC comparison table shape
- integration across Equal Weight, Inverse Volatility, HRP, and HERC

---

## 9. Interpretation of HRP vs HERC Risk Contribution

This stage makes the HRP vs HERC tradeoff easier to explain:

- HRP and HERC may look similar in capital weights but differ materially in percentage risk contribution
- if HERC spreads percentage risk contribution more evenly, that gives a direct explanation for lower drawdowns
- if HRP concentrates risk in a small number of volatile or correlated assets, it may retain higher CAGR in some periods but experience deeper drawdowns

The platform can now distinguish between:

- capital allocation
- actual risk attribution

That distinction is necessary for research-grade portfolio analysis.

---

## 10. Remaining Limitations

Still out of scope:

- benchmark framework
- threshold rebalancing
- transaction cost redesign
- volatility targeting
- NLP
- regime detection

Current limitations:

- dashboard attribution uses the covariance matrix computed in the app, which is currently the sample covariance path
- no rolling time-series risk contribution history has been added
- no separate transaction-cost-adjusted attribution is included

---

## 11. Conclusion

Stage 10 adds a complete risk attribution layer to the platform. The codebase can now explain:

1. what portfolio weights were assigned
2. which assets drive portfolio volatility
3. how HRP and HERC differ not only in capital allocation but in actual risk distribution

This improves the platform’s ability to interpret drawdown behavior and makes HRP vs HERC comparisons materially more informative.
