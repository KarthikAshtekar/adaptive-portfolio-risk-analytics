# Stage 8 Implementation Report: Covariance Research Engine

**Status**: Complete  
**Date**: 2026-06-05  
**Focus**: Build a modular covariance research framework for comparing estimation methods without changing downstream portfolio construction modules

---

## Executive Summary

Phase 2A.1 upgrades the platform from a single sample covariance workflow into a reusable covariance research engine. The new implementation adds a factory interface, multiple covariance estimators, metadata capture, reusable validation helpers, a research notebook, and focused tests.

This stage intentionally does **not** modify HRP, HERC, backtesting logic, analytics, transaction costs, benchmarking, or the Streamlit dashboard. Existing Phase 1 behavior remains available through the existing sample covariance API and the new `CovarianceFactory`.

---

## Files Created

- `src/covariance/ledoit_wolf.py`
- `src/covariance/ewma_covariance.py`
- `src/covariance/covariance_factory.py`
- `tests/test_ledoit_wolf.py`
- `tests/test_ewma_covariance.py`
- `tests/test_covariance_factory.py`
- `notebooks/08_covariance_research/stage_08_covariance_research.ipynb`
- `STAGE_8_REPORT.md`

## Files Modified

- `src/covariance/__init__.py`

---

## Covariance Methods Implemented

### 1. Sample Covariance

**Interface**: `CovarianceFactory.compute(returns_df, method="sample")`

**Intuition**: Estimate pairwise co-movement directly from historical returns with equal weight on all observations.

**Advantages**:
- Simple and transparent
- Baseline for all later comparisons
- Already compatible with the existing platform

**Disadvantages**:
- Sensitive to estimation noise
- Can become unstable when assets are numerous relative to observations
- Treats old and recent observations equally

### 2. Ledoit-Wolf Shrinkage

**Interface**: `compute_ledoit_wolf_covariance()` and `method="ledoit_wolf"`

**Intuition**: Blend the noisy sample covariance matrix with a structured shrinkage target to reduce out-of-sample instability. The shrinkage intensity is learned from the data using `sklearn.covariance.LedoitWolf`.

**Advantages**:
- More stable than raw sample covariance
- Typically better conditioned for optimization
- Automatically stores the fitted shrinkage coefficient in metadata

**Disadvantages**:
- Less interpretable than pure historical covariance
- Can smooth away some genuinely recent structure

### 3. EWMA Covariance

**Interface**: `compute_ewma_covariance()` and `method="ewma"`

**Intuition**: Weight recent returns more heavily than older returns using exponential decay. This makes the covariance estimate more responsive to regime changes and volatility clustering.

**Advantages**:
- Adapts faster to new market conditions
- Useful when covariance structure is time-varying
- Configurable through the `span` parameter

**Disadvantages**:
- More sensitive to short-term noise
- Can overreact in turbulent windows

### 4. EWMA + Ledoit-Wolf

**Interface**: `compute_ewma_ledoit_wolf_covariance()` and `method="ewma_ledoit_wolf"`

**Intuition**: First create EWMA-weighted returns so recent observations matter more, then apply Ledoit-Wolf shrinkage for regularization. This combines adaptiveness with noise control.

**Advantages**:
- Balances recency sensitivity and numerical stability
- Strong candidate for later rolling optimization workflows

**Disadvantages**:
- More complex than either component alone
- Requires interpretation of both decay and shrinkage effects

---

## Research API

### Covariance Factory

The new central interface is:

```python
from src.covariance import CovarianceFactory

cov_matrix = CovarianceFactory.compute(
    returns_df,
    method="ewma_ledoit_wolf",
    span=126,
)
```

Supported methods:

- `sample`
- `ledoit_wolf`
- `ewma`
- `ewma_ledoit_wolf`

All factory outputs:

- Return a `pd.DataFrame`
- Preserve original asset labels
- Preserve matrix shape
- Pass reusable validation checks before returning

### Metadata

Estimator metadata is attached via `DataFrame.attrs` and can be retrieved with:

```python
from src.covariance import extract_covariance_metadata

metadata = extract_covariance_metadata(cov_matrix)
```

Examples:

- `{"method": "sample"}`
- `{"method": "ledoit_wolf", "shrinkage": ...}`
- `{"method": "ewma", "span": 126}`
- `{"method": "ewma_ledoit_wolf", "span": 126, "shrinkage": ...}`

---

## Validation Utilities

Reusable validation helpers were added to verify that each estimator produces a valid covariance matrix:

- Square matrix
- Symmetric matrix
- Positive diagonal
- No NaNs

Public helpers:

- `validate_estimated_covariance_matrix()`
- `assert_valid_covariance_matrix()`

These checks are used by the factory before results are returned.

---

## Notebook Deliverable

**File**: `notebooks/08_covariance_research/stage_08_covariance_research.ipynb`

The notebook includes:

1. Load deterministic returns data
2. Sample covariance
3. Ledoit-Wolf covariance
4. EWMA covariance
5. EWMA + Ledoit-Wolf covariance
6. Heatmap comparison
7. Difference heatmaps
8. Eigenvalue comparison
9. Correlation structure comparison
10. Covariance stability discussion

Visuals included:

- Covariance heatmaps
- Difference heatmaps
- Eigenvalue profiles
- Correlation heatmaps

---

## Future Integration Points

This stage was designed as infrastructure for later portfolio construction research.

### HRP Integration

Later HRP experiments can swap covariance methods without changing clustering code:

- Convert covariance to correlation
- Convert correlation to distance
- Feed distance matrix into linkage construction

### HERC Integration

HERC can use the same factory to compare cluster stability and allocation sensitivity under different covariance assumptions.

### Rolling Optimization / Backtesting

Future phases can expose covariance method selection as a strategy parameter in:

- Rolling backtests
- HRP/HERC allocators
- Benchmark comparisons
- Volatility targeting workflows

---

## Test Results

Targeted regression tests were executed with the project virtual environment:

```bash
.venv\Scripts\python.exe -m pytest tests\test_ledoit_wolf.py tests\test_ewma_covariance.py tests\test_covariance_factory.py tests\test_covariance.py -q
```

Result:

- `21 passed`

Covered checks:

- Shape consistency
- Symmetry
- Positive diagonals
- No NaNs
- Label preservation
- Metadata capture
- Factory routing
- Backward compatibility of existing covariance utilities

---

## Backward Compatibility

The following areas were left unchanged:

- HRP
- HERC
- Backtesting
- Analytics
- Dashboard
- Optimization
- Clustering

Existing sample covariance logic still works through:

- `compute_covariance_matrix()`
- `CovarianceFactory.compute(..., method="sample")`
- `SampleCovarianceEstimator`

---

## Recommendations for Phase 2A.2

1. Add rolling estimator studies that compare covariance method choice inside a walk-forward backtest.
2. Introduce covariance-method configuration in optimization inputs without changing default behavior.
3. Measure portfolio sensitivity to covariance choice using turnover, diversification, and weight concentration diagnostics.
4. If HERC is introduced later, use the factory as the only covariance entry point to keep method comparison consistent.

---

## Conclusion

Stage 8 establishes a clean covariance research layer with four estimator choices, validation, metadata, tests, and a notebook for analysis. The platform can now study how covariance estimation affects downstream portfolio construction in later phases without forcing any immediate changes to existing Phase 1 workflows.
