# Phase 2A Patch Report

**Status**: Complete  
**Date**: 2026-06-06  
**Scope**: Fix Phase 2A integration issues identified in `PHASE_2A_AUDIT_REPORT.md` without adding Phase 2B features

---

## 1. Files Modified

- `src/optimization/base.py`
- `src/clustering/herc_allocator.py`
- `src/optimization/hrp_allocator.py`
- `src/backtesting/rolling_backtester.py`
- `src/backtesting/__init__.py`
- `tests/test_herc_allocator.py`
- `tests/test_hrp.py`
- `tests/test_backtesting.py`
- `tests/test_phase2a_integration.py`

## 2. Files Created

- `PHASE_2A_PATCH_REPORT.md`

---

## 3. HERC Label Fix Summary

### Problem

`HERCAllocator.optimize()` returned `np.ndarray`, so asset labels were lost at the public allocator boundary.

### Fix

`HERCAllocator.get_weights()` now returns a labeled `pd.Series`:

- index = `returns_df.columns`
- values = portfolio weights
- name = `"weight"`

The allocator still preserves:

- fully invested weights
- non-negative weights
- no NaNs
- finite values only

### Compatibility

`RollingBacktester` remains compatible because it already converts allocator output with:

```python
np.asarray(target_weights, dtype=float)
```

That works for both `pd.Series` and `np.ndarray`.

---

## 4. HRP Covariance-Method Support Summary

### Problem

`HRPAllocator` previously defaulted to sample covariance via `clean_returns.cov()` and did not expose `covariance_method`.

### Fix

`HRPAllocator` now supports:

```python
HRPAllocator(covariance_method="sample")
HRPAllocator(covariance_method="ledoit_wolf")
HRPAllocator(covariance_method="ewma", covariance_kwargs={"span": 126})
HRPAllocator(covariance_method="ewma_ledoit_wolf", covariance_kwargs={"span": 126})
```

Internally, it now routes covariance estimation through:

```python
CovarianceFactory.compute(...)
```

### Result

HRP and HERC now share the same allocator-level covariance configuration surface for:

- `sample`
- `ledoit_wolf`
- `ewma`
- `ewma_ledoit_wolf`

### Additional Consistency

`HRPAllocator.get_weights()` now also returns a labeled `pd.Series` named `"weight"` for interface symmetry with HERC.

---

## 5. Backward Compatibility Checks

Preserved behavior:

- `HRPAllocator()` still works with default sample covariance
- `HERCAllocator` numerical behavior is preserved
- `RollingBacktester` still works without changes to caller code
- dashboard strategy selection remains unchanged
- Phase 1 strategies still run through `RollingBacktester`

Compatibility details:

- the allocator base interface was widened to allow `pd.Series | np.ndarray`
- downstream code that depends on array-like behavior continues to work

---

## 6. Rebalance Frequency Warning Fix

### Problem

pandas warns that frequency code `"M"` is deprecated in `date_range()` and should be replaced by `"ME"`.

### Fix

Added normalization in the backtester:

- `"M"` maps to `"ME"` for `date_range()`
- `"ME"` maps back to `"M"` for `to_period()` because pandas `Period` does not accept `"ME"`

This preserves backward compatibility for existing code that passes:

- `"M"`
- `"ME"`

### Result

Monthly rebalance-date generation no longer emits the prior pandas deprecation warning in the tested paths.

---

## 7. Test Commands Executed

Executed with the project virtual environment:

```bash
.venv\Scripts\python.exe -m pytest tests\test_phase2a_integration.py -q
.venv\Scripts\python.exe -m pytest tests\test_covariance_factory.py tests\test_herc_allocator.py tests\test_hrp.py tests\test_backtesting.py -q
```

---

## 8. Test Results

### Integration Suite

```bash
.venv\Scripts\python.exe -m pytest tests\test_phase2a_integration.py -q
```

Result:

- `22 passed`

Resolved from audit:

- previous 4 HERC label failures are gone
- HRP covariance-method support is no longer skipped

### Regression Suite

```bash
.venv\Scripts\python.exe -m pytest tests\test_covariance_factory.py tests\test_herc_allocator.py tests\test_hrp.py tests\test_backtesting.py -q
```

Result:

- `41 passed`

Observed:

- no pandas `"M"` deprecation warning in this requested test run

---

## 9. Remaining Issues

No Phase 2A integration failures remain in the requested test scope.

Still out of scope for this patch:

- benchmark framework
- transaction cost redesign
- threshold rebalancing
- volatility targeting
- NLP
- regime detection
- new risk contribution analytics

Also unchanged:

- `src/clustering/herc.py` remains the older placeholder implementation and is not used by the allocator path patched here

---

## 10. Conclusion

The Phase 2A integration patch is complete.

It fixes:

1. HERC labeled output
2. HRP covariance-method configurability
3. HRP/HERC allocator-level comparability
4. pandas monthly rebalance frequency deprecation handling

The requested integration and regression test commands now pass without failures, while preserving existing Phase 1 behavior.
