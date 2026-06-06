# Phase 2B Rebalance Audit Report

**Status**: Audit complete  
**Date**: 2026-06-07  
**Scope**: Threshold rebalance behavior in the realistic rolling backtester

---

## 1. Audit Conclusion

The old threshold implementation was not aligned with the intended semantics.

The issue was not that threshold mode could ever rebalance more than calendar mode in principle. That can happen in volatile markets.

The issue was that the old implementation recomputed `target_weights` on every backtest step and then compared current portfolio weights against that moving target. That means threshold mode was partly reacting to daily model target changes, not just to natural portfolio drift.

This has now been fixed.

---

## 2. What Was Wrong

### Old behavior

Inside `RollingBacktester.run()`, the allocator was called on every loop iteration:

- training window rolled daily
- target weights were recomputed daily
- threshold drift was measured against the newly recomputed target

That effectively mixed together two separate concepts:

1. portfolio drift from market moves
2. target drift from daily model recalculation

For threshold rebalancing research, those should not be the same thing.

### Why that matters

Threshold rebalancing is supposed to answer:

> "Has the live portfolio drifted far enough away from the intended target that it should be traded back?"

It is not supposed to answer:

> "Did the optimizer produce a meaningfully different target today?"

The old implementation therefore overstated threshold-triggered trading activity whenever target weights were unstable day to day.

---

## 3. Fix Applied

`RollingBacktester` now separates:

- **target updates**
- **rebalance decisions**

New logic:

- `target_weights` are stored as backtester state
- `target_weights` are recomputed only on `target_update_frequency`
- default `target_update_frequency = "M"`
- portfolio weights still drift naturally every day
- threshold mode compares drifted live weights to the latest stored target
- threshold rebalance triggers only when:

```python
max(abs(current_weights - target_weights)) >= threshold
```

No allocator, covariance, or analytics logic was changed.

---

## 4. Diagnostics Added

`rebalance_summary` now includes:

- `calendar_rebalances`
- `threshold_rebalances`
- `calendar_or_threshold_rebalances`
- `rebalance_reason_counts`
- `average_turnover_by_reason`
- `max_weight_drift`
- `average_max_weight_drift`
- `max_weight_drift_by_reason`

This makes it explicit whether trading came from the calendar schedule, threshold breaches, or both.

---

## 5. Before/After Sample Audit

Sample configuration used for the audit:

- Allocator: `InverseVolatilityAllocator`
- Train window: `60`
- Rebalance frequency: `M`
- Threshold: `5%`
- Returns: deterministic Gaussian sample with seed `11`

### Rebalance counts

| Mode | Old Count | New Count |
|---|---:|---:|
| `calendar` | 17 | 17 |
| `threshold` | 6 | 4 |
| `calendar_or_threshold` | 18 | 17 |

### Reason counts

Old:

- `calendar`: `{"calendar": 16, "initial": 1}`
- `threshold`: `{"threshold": 5, "initial": 1}`
- `calendar_or_threshold`: `{"calendar": 16, "threshold": 1, "initial": 1}`

New:

- `calendar`: `{"calendar": 16, "initial": 1}`
- `threshold`: `{"threshold": 3, "initial": 1}`
- `calendar_or_threshold`: `{"calendar": 16, "initial": 1}`

### Turnover comparison

Average turnover per rebalance in the same audit sample:

| Mode | Old Avg Turnover | New Avg Turnover |
|---|---:|---:|
| `calendar` | 0.025193 | 0.025193 |
| `threshold` | 0.049136 | 0.046994 |
| `calendar_or_threshold` | 0.025698 | 0.025193 |

### Max weight drift comparison

| Mode | Old Max Drift | New Max Drift |
|---|---:|---:|
| `calendar` | 0.046909 | 0.046909 |
| `threshold` | 0.056423 | 0.059877 |
| `calendar_or_threshold` | 0.061826 | 0.046909 |

Interpretation:

- `calendar` behavior was unchanged
- `threshold` rebalances fell after removing daily target-motion noise
- `calendar_or_threshold` lost the extra threshold-driven rebalance in this sample because that event was caused by daily target recomputation, not by genuine live-weight drift against the stored monthly target

---

## 6. Test Evidence

Added or updated checks now verify:

### Test A

With constant equal asset returns and no drift, threshold mode does not rebalance repeatedly.

### Test B

For the same return series:

- `3%` threshold rebalance count >= `5%` threshold rebalance count >= `10%` threshold rebalance count

Observed deterministic sample:

- `3%`: `5`
- `5%`: `2`
- `10%`: `1`

### Test C

Calendar mode still rebalances on schedule.

### Test D

`calendar_or_threshold` has at least as many rebalances as calendar mode.

### Test E

`rebalance_log` records:

- `rebalance_date`
- `rebalance_reason`
- `turnover`
- `transaction_cost`
- `max_weight_drift`

### Regression coverage

Executed:

```bash
.\.venv\Scripts\python.exe -m pytest tests\test_turnover.py tests\test_rebalance_rules.py tests\test_backtest_diagnostics.py tests\test_backtesting.py tests\test_strategy_comparison.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_phase2a_integration.py tests\test_hrp.py tests\test_herc_allocator.py tests\test_risk_contribution.py tests\test_benchmark_factory.py -q
```

Results:

- `41 passed`
- `70 passed`

---

## 7. Final Recommendation

Keep the new separation between:

- target update cadence
- rebalance trigger cadence

This is the correct research-grade design for threshold rebalancing.

Default recommendation:

- `target_update_frequency = "M"`
- `rebalance_mode = "threshold"` or `calendar_or_threshold` depending the experiment

Reason:

- it preserves realistic daily portfolio drift
- it prevents threshold logic from firing due to daily optimizer noise
- it leaves calendar behavior unchanged
- it makes threshold backtests interpretable

The old behavior should not be restored.
