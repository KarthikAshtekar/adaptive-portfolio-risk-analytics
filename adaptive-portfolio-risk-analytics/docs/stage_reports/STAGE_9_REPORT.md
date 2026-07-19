# Stage 9 Implementation Report: HERC Portfolio Construction Engine

**Status**: Complete  
**Date**: 2026-06-06  
**Focus**: Add a research-grade Hierarchical Equal Risk Contribution allocator on top of the existing HRP and covariance research framework

---

## Executive Summary

Phase 2A.2 extends the portfolio construction stack with a functional `HERCAllocator` that:

- preserves Phase 1 allocators and HRP behavior
- uses `CovarianceFactory` for covariance-method selection
- reuses the existing hierarchical clustering pipeline
- supports end-to-end comparison between:
  - Equal Weight
  - Inverse Volatility
  - HRP
  - HERC

The implementation was intentionally scoped to HERC integration only. No benchmark framework, transaction cost redesign, threshold rebalancing, volatility targeting, NLP, or regime detection work was added.

---

## Files Created

- `src/clustering/herc_allocator.py`
- `tests/test_herc_allocator.py`
- `notebooks/09_herc_portfolio_construction/stage_09_herc.ipynb`
- `STAGE_9_REPORT.md`

## Files Modified

- `src/clustering/__init__.py`
- `src/dashboard/app.py`
- `src/dashboard/plots.py`
- `src/optimization/__init__.py`

---

## HERC Methodology

### Allocator Interface

The new allocator mirrors existing usage patterns:

```python
from src.clustering import HERCAllocator

allocator = HERCAllocator(covariance_method="ledoit_wolf")
weights = allocator.optimize(returns_df)
```

Properties enforced:

- long only
- fully invested
- weights sum to 1
- no negative allocations
- no NaNs

### Covariance Integration

HERC routes covariance estimation through:

```python
CovarianceFactory.compute(returns_df, method=...)
```

Supported methods:

- `sample`
- `ledoit_wolf`
- `ewma`
- `ewma_ledoit_wolf`

This keeps Stage 8 as the single covariance entry point and avoids introducing allocator-specific covariance code.

### Clustering Pipeline Reuse

HERC uses the same clustering sequence already established in the project:

1. covariance matrix
2. implied correlation matrix
3. distance matrix
4. linkage matrix
5. hierarchical tree traversal

The allocator reuses:

- `compute_distance_matrix()`
- `compute_linkage_matrix()`

No duplicate linkage or distance logic was added.

### Allocation Logic

The implementation follows recursive branch-level risk budgeting:

1. compute cluster covariance for each sibling branch
2. compute cluster risk for each branch
3. allocate capital inversely to branch risk so sibling risk contributions are equal
4. recurse until leaf assets are reached

Cluster risk is measured as cluster volatility using local inverse-volatility weights inside each cluster.

### Validation Helpers

The module includes:

- `validate_weights()`
- `compute_cluster_risk()`
- `compare_hrp_herc_weights()`

These helpers support both tests and notebook analysis.

---

## Differences Between HRP and HERC

The existing HRP implementation was left unchanged.

### HRP

- uses quasi-diagonal leaf ordering
- applies midpoint recursive bisection across the ordered asset list
- scales left/right allocations from cluster variance estimates

### HERC

- traverses the explicit linkage tree
- allocates branch capital to equalize sibling branch risk
- measures cluster risk as cluster volatility under local inverse-volatility weights

### Why Allocations Differ

Even when HRP and HERC start from the same covariance estimate and linkage matrix, they can produce different weights because:

1. HRP splits by ordered midpoint, while HERC follows the actual tree structure.
2. HRP uses its existing cluster variance recursion, while HERC uses a local equal-risk cluster proxy and branch-volatility budgeting.

This makes HERC a distinct research comparator rather than a renamed HRP wrapper.

---

## Dashboard Integration

The Streamlit dashboard was changed minimally:

- added `HERC` to the strategy selector
- routed `HERC` to `HERCAllocator()` through the existing `get_allocator()` helper

No dashboard redesign or new workflow was added.

---

## Notebook Deliverable

**File**: `notebooks/09_herc_portfolio_construction/stage_09_herc.ipynb`

Sections included:

1. Load Data
2. Compute Covariance
3. HRP Allocation
4. HERC Allocation
5. Weight Comparison
6. Diversification Analysis
7. Discussion

The notebook compares:

- Sample
- Ledoit-Wolf
- EWMA
- EWMA + Ledoit-Wolf

It also includes:

- HRP/HERC weight tables
- grouped bar chart comparison
- diversification summary using:
  - largest weight
  - smallest weight
  - weight dispersion

---

## Results Summary

Implementation outcome:

- HERC works end-to-end through the allocator interface
- HERC can be backtested through the existing rolling backtester
- HERC is available in the dashboard strategy selector
- HRP and HERC can be compared under the same covariance estimate without changing existing HRP logic

The codebase now supports a cleaner research comparison between:

- Equal Weight
- Inverse Volatility
- HRP
- HERC

---

## Testing Summary

Executed with the project virtual environment:

```bash
.venv\Scripts\python.exe -m pytest tests\test_herc_allocator.py tests\test_hrp.py tests\test_optimization.py tests\test_backtesting.py -q
.venv\Scripts\python.exe -m pytest tests\test_covariance_factory.py -q
```

Results:

- `34 passed` for allocator, HRP, optimization, and backtesting coverage
- `7 passed` for covariance factory coverage

Covered checks:

- HERC weights sum to 1
- HERC weights are non-negative
- one output weight per asset
- no NaNs / finite output
- multiple covariance estimators work
- cluster-risk helper returns positive values
- HRP vs HERC comparison table shape and columns
- rolling backtester integration
- dashboard strategy selector exposes HERC

Note:

- one existing warning remains in `generate_rebalance_dates()` because pandas deprecates frequency code `M` in favor of `ME`

---

## Backward Compatibility

Phase 1 behavior remains intact:

- existing HRP implementation unchanged
- existing Equal Weight allocator unchanged
- existing Inverse Volatility allocator unchanged
- existing backtester workflow unchanged

HERC was added as a new path instead of refactoring old portfolio logic.

---

## When HRP May Be Preferred

- when strict continuity with existing Phase 1 HRP research is required
- when midpoint recursive bisection is the desired hierarchical allocation rule
- when comparability with prior HRP backtests matters more than trying an alternative cluster-risk budget

## When HERC May Be Preferred

- when branch-level equal risk contribution is the research objective
- when the explicit tree structure should drive recursive splits
- when comparing covariance estimators under a more risk-budget-oriented hierarchical allocator

---

## Future Improvements

1. Add dedicated strategy-comparison dashboards for HRP vs HERC weights and concentration metrics.
2. Extend rolling studies to compare turnover and stability by covariance estimator.
3. Add optional reporting around cluster structure and branch risk contributions at each recursion level.
4. If desired later, add a more exact intra-cluster ERC solver while keeping the current interface stable.

---

## Conclusion

Stage 9 completes the HERC portfolio construction engine as a clean extension of the current research platform. The allocator is integrated with the covariance factory, available in the dashboard, covered by tests, and packaged with a notebook for HRP versus HERC analysis without breaking existing Phase 1 functionality.
