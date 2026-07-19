# Phase 2A Audit Report

**Status**: Audit complete  
**Date**: 2026-06-06  
**Scope**: Stage 8 Covariance Research Engine and Stage 9 HERC integration quality, backward compatibility, and architecture alignment

---

## 1. Tests Created

Created:

- `tests/test_phase2a_integration.py`

This audit suite covers:

- CovarianceFactory routing and matrix quality
- covariance metadata integrity
- HRP covariance-method configurability detection
- HERC covariance-method support
- HERC integration with `RollingBacktester`
- Phase 1 allocator regressions
- dashboard strategy factory exposure
- public import API consistency

---

## 2. Commands Executed

Executed with the project virtual environment:

```bash
.venv\Scripts\python.exe -m pytest tests\test_phase2a_integration.py -q
.venv\Scripts\python.exe -m pytest tests\test_covariance_factory.py tests\test_herc_allocator.py tests\test_backtesting.py -q
```

---

## 3. Test Results

### Audit Suite

Command:

```bash
.venv\Scripts\python.exe -m pytest tests\test_phase2a_integration.py -q
```

Result:

- `14 passed`
- `1 skipped`
- `4 failed`

Failure summary:

- `test_herc_covariance_method_support_returns_labeled_weights[sample]`
- `test_herc_covariance_method_support_returns_labeled_weights[ledoit_wolf]`
- `test_herc_covariance_method_support_returns_labeled_weights[ewma]`
- `test_herc_covariance_method_support_returns_labeled_weights[ewma_ledoit_wolf]`

Why they failed:

- `HERCAllocator.optimize()` returns `np.ndarray`, not `pd.Series`
- the weights are numerically valid, but the integration API loses asset labels

Skipped test:

- `HRPAllocator` covariance-method support check was skipped because `HRPAllocator.__init__` does not expose `covariance_method`

### Existing Targeted Regression Suite

Command:

```bash
.venv\Scripts\python.exe -m pytest tests\test_covariance_factory.py tests\test_herc_allocator.py tests\test_backtesting.py -q
```

Result:

- `26 passed`
- `1 warning`

Warning:

- pandas deprecation warning in `generate_rebalance_dates()` for frequency code `M`

---

## 4. Whether CovarianceFactory Works Correctly

**Conclusion**: Yes

Verified for:

- `sample`
- `ledoit_wolf`
- `ewma`
- `ewma_ledoit_wolf`

For each method the audit confirmed:

- output is `pd.DataFrame`
- shape is `n_assets x n_assets`
- row and column labels match input assets
- matrix is symmetric
- diagonal values are positive
- no NaNs
- metadata is attached in `DataFrame.attrs`

---

## 5. Whether Metadata Is Correctly Attached

**Conclusion**: Yes

Verified metadata:

- `sample`
  - `method == "sample"`
- `ledoit_wolf`
  - `method == "ledoit_wolf"`
  - `shrinkage` exists
  - `0 <= shrinkage <= 1`
- `ewma`
  - `method == "ewma"`
  - `span` exists
- `ewma_ledoit_wolf`
  - `method == "ewma_ledoit_wolf"`
  - `span` exists
  - `shrinkage` exists
  - `0 <= shrinkage <= 1`

---

## 6. Whether HERC Supports All Covariance Estimators

**Conclusion**: Numerically yes, API-wise partially

Observed behavior:

- `HERCAllocator(covariance_method="sample")` computes valid weights
- `HERCAllocator(covariance_method="ledoit_wolf")` computes valid weights
- `HERCAllocator(covariance_method="ewma")` computes valid weights
- `HERCAllocator(covariance_method="ewma_ledoit_wolf")` computes valid weights

Confirmed properties:

- weights are finite
- weights are non-negative
- weights sum to 1
- backtester integration works

Audit finding:

- `HERCAllocator.optimize()` does **not** return `pd.Series`
- it currently returns `np.ndarray`, which strips asset labels and weakens research ergonomics

This is the reason the new integration tests fail.

---

## 7. Whether HRP Supports All Covariance Estimators

**Conclusion**: No, not through the allocator interface

Inspection result:

- `HRPAllocator.__init__` does not expose `covariance_method`
- when no covariance matrix is supplied, `HRPAllocator.fit()` always falls back to `clean_returns.cov()`

Implication:

- HRP cannot currently be configured in the same allocator-style way as HERC for:
  - `sample`
  - `ledoit_wolf`
  - `ewma`
  - `ewma_ledoit_wolf`

Per audit instructions, this was documented rather than converted into a failing test.

---

## 8. Whether HRP and HERC Are Truly Comparable Under the Same Covariance Method

**Conclusion**: Only partially

What is true:

- helper-level comparison is possible through `compare_hrp_herc_weights()`
- that helper computes one covariance estimate and feeds the same covariance/linkage inputs to both weight engines

What is not true:

- allocator-to-allocator parity is missing because `HRPAllocator` cannot be instantiated with `covariance_method`

Therefore:

- **research comparison exists at helper level**
- **allocator-level comparability is incomplete**

Recommended future patch:

- extend `HRPAllocator` to accept `covariance_method` and route through `CovarianceFactory`

---

## 9. Whether HERC Works Inside RollingBacktester

**Conclusion**: Yes

Verified with:

```python
RollingBacktester(
    allocator=HERCAllocator(covariance_method="ledoit_wolf"),
    train_window=60,
    rebalance_frequency="M",
).run(returns_df)
```

Confirmed:

- `portfolio_returns` exists and is non-empty
- `portfolio_values` exists and remains positive
- `drawdown` exists
- `weights_history` exists and is non-empty
- `performance_metrics` exists
- all rows of `weights_history` sum to 1
- `performance_metrics` includes:
  - `cagr`
  - `sharpe`
  - `volatility`
  - `max_drawdown`

---

## 10. Whether Dashboard Exposes HERC Correctly

**Conclusion**: Yes

Verified:

- `get_allocator("HERC")` returns a `HERCAllocator`
- dashboard strategy selector includes `HERC`

No dashboard UI changes were required for the audit.

---

## 11. Import API Consistency

**Conclusion**: Acceptable

Verified public imports:

```python
from src.covariance import CovarianceFactory
from src.clustering import HERCAllocator
from src.optimization import HERCAllocator
```

Observation:

- `src.optimization.HERCAllocator` and `src.clustering.HERCAllocator` resolve to the same class
- the preferred public import remains:

```python
from src.optimization import HERCAllocator
```

because HERC is a portfolio allocator, even though its implementation lives in the clustering package

---

## 12. Architectural Concerns

### Concern 1: HERC output loses labels

`HERCAllocator` stores labeled weights internally as `pd.Series`, but `get_weights()` returns `.values`.

Impact:

- breaks the audit expectation for labeled research outputs
- makes downstream diagnostics less explicit
- diverges from the richer internal representation already available in the allocator

### Concern 2: HRP and HERC are not symmetric research interfaces

HERC supports covariance-method selection through `CovarianceFactory`, but HRP does not.

Impact:

- Phase 2A covariance research is fully available for HERC
- Phase 2A covariance research is not available through the HRP allocator interface
- allocator-level comparison remains asymmetric

### Concern 3: Existing tests did not check labeled allocator outputs

The current Stage 9 tests validate HERC numerics and integration, but they do not assert preservation of asset labels at `optimize()` output.

Impact:

- a real integration mismatch passed the earlier suite

### Concern 4: Rebalance frequency warning remains

`RollingBacktester.generate_rebalance_dates()` still uses pandas frequency code `M`, which now emits a deprecation warning.

Impact:

- not a Phase 2A correctness failure
- should be cleaned up in a separate compatibility patch

---

## 13. Recommended Fixes

### Recommended fix 1

Return labeled weights from `HERCAllocator.optimize()` / `get_weights()` or expose an explicit labeled accessor.

Rationale:

- fixes the audit failure
- preserves asset identity
- improves notebook and research workflows

### Recommended fix 2

Extend `HRPAllocator` to support:

- `covariance_method`
- optional covariance kwargs
- `CovarianceFactory` routing

Rationale:

- enables true allocator-level HRP vs HERC covariance-method comparison

### Recommended fix 3

Strengthen allocator API expectations in tests.

Rationale:

- prevents future regressions where labeled portfolio outputs degrade into unlabeled arrays

### Recommended fix 4

Replace pandas frequency code `M` with `ME` in the rebalance-date helper in a separate patch.

Rationale:

- removes the standing deprecation warning without changing Phase 2A scope

---

## 14. Overall Audit Conclusion

Stage 8 is functioning correctly.

Stage 9 is mostly integrated correctly:

- covariance-method support works
- backtester integration works
- dashboard exposure works
- import API is usable

But two quality gaps remain:

1. `HERCAllocator.optimize()` returns unlabeled arrays instead of labeled `pd.Series`
2. `HRPAllocator` still lacks covariance-method configurability, so HRP and HERC are not yet fully comparable through the allocator interface

The new audit suite correctly surfaces the first issue as a failing integration test and documents the second as an architectural comparability gap.
