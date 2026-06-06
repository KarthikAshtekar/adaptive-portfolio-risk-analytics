# Stage 11 Implementation Report: Benchmark Framework and Strategy Comparison Engine

**Status**: Complete  
**Date**: 2026-06-07  
**Focus**: Add reusable benchmark utilities and a multi-strategy comparison engine so the platform can answer strategy performance relative to explicit baselines

---

## 1. Files Created

- `src/benchmarks/__init__.py`
- `src/benchmarks/benchmark_factory.py`
- `src/benchmarks/strategy_comparison.py`
- `tests/test_benchmark_factory.py`
- `tests/test_strategy_comparison.py`
- `notebooks/11_benchmark_comparison/stage_11_benchmark_comparison.ipynb`
- `STAGE_11_REPORT.md`

## 2. Files Modified

- `src/dashboard/app.py`
- `src/dashboard/plots.py`

---

## 3. Benchmark Framework Design

Stage 11 introduces a dedicated `benchmarks` package rather than embedding benchmark logic inside the dashboard or backtester.

Core pieces:

- `BenchmarkFactory`
- `run_strategy_comparison()`
- `build_performance_comparison_table()`
- `compute_relative_performance()`

Supported strategies / benchmarks:

- `equal_weight`
- `inverse_volatility`
- `hrp`
- `herc`

Supported aliases:

- `Equal Weight`
- `Inverse Volatility`
- `HRP`
- `HERC`

Design principles:

- preserve existing allocator behavior
- pass `covariance_method` only to allocators that support it
- normalize aliases once through the factory
- keep comparison logic reusable outside the dashboard

---

## 4. Strategy Comparison Methodology

### Strategy Execution

`run_strategy_comparison()`:

1. normalizes each requested strategy name
2. constructs the correct allocator through `BenchmarkFactory`
3. runs the existing `RollingBacktester`
4. stores:
   - `portfolio_returns`
   - `portfolio_values`
   - `drawdown`
   - `weights_history`
   - `performance_metrics`
   - `latest_weights`

### Performance Table

`build_performance_comparison_table()` returns a standardized table with:

- `cumulative_return`
- `cagr`
- `sharpe`
- `sortino`
- `volatility`
- `max_drawdown`
- `calmar`
- `var_95`
- `cvar_95`
- `final_value`

### Relative Performance

`compute_relative_performance()` compares every strategy against a chosen benchmark and returns:

- `strategy`
- `benchmark`
- `excess_cagr`
- `excess_sharpe`
- `drawdown_difference`
- `volatility_difference`
- `final_value_difference`

This makes the comparison question explicit:

`better than what?`

---

## 5. Dashboard Additions

The existing single-strategy workflow was preserved.

Added a new dashboard section:

- `Benchmark Comparison`

Sidebar inputs added:

- strategy multiselect
- benchmark selector

Main outputs added:

1. performance comparison table
2. portfolio growth comparison chart
3. drawdown comparison chart
4. metric comparison chart
5. final value comparison chart
6. relative performance chart
7. relative performance table versus selected benchmark

All visualizations use Plotly and reuse the new benchmark utilities.

---

## 6. Notebook Findings

Notebook:

- `notebooks/11_benchmark_comparison/stage_11_benchmark_comparison.ipynb`

Sections included:

1. Load data
2. Generate returns
3. Run all strategies
4. Compare performance metrics
5. Compare drawdowns
6. Compare final values
7. Compute relative performance versus Equal Weight
8. Interpret results

Research questions addressed:

- Does HRP outperform Equal Weight?
- Does HERC reduce drawdown relative to HRP?
- Does Inverse Volatility compete with HRP/HERC?
- Which strategy has the best Sharpe?
- Which strategy has the best Calmar?
- Which strategy is most defensive?

---

## 7. Test Results

Executed:

```bash
.venv\Scripts\python.exe -m pytest tests\test_benchmark_factory.py tests\test_strategy_comparison.py -q
.venv\Scripts\python.exe -m pytest tests\test_phase2a_integration.py tests\test_backtesting.py tests\test_hrp.py tests\test_herc_allocator.py tests\test_risk_contribution.py -q
```

Results:

- `17 passed` for benchmark factory and comparison tests
- `71 passed` for the combined regression suite covering Phase 2A strategies, backtesting, and risk contribution analytics

Covered checks include:

- allocator factory routing
- alias support
- invalid strategy rejection
- multi-strategy comparison result structure
- expected performance comparison columns
- relative performance table columns
- positive final values
- supported covariance-method comparison runs for HRP and HERC
- regression coverage for previously implemented phases

---

## 8. Interpretation

### HRP vs Equal Weight

The framework now evaluates HRP relative to Equal Weight directly through:

- absolute metrics
- growth curves
- drawdown comparisons
- excess CAGR / excess Sharpe

### HERC vs HRP

HERC can now be evaluated in the same benchmark framework as HRP, making it easier to see whether lower drawdowns justify lower CAGR in some samples.

### Inverse Volatility vs HRP/HERC

Inverse Volatility is now promoted from an isolated strategy to a formal benchmark candidate. This is useful because it often acts as a stronger defensive baseline than Equal Weight.

---

## 9. Remaining Limitations

Still out of scope:

- threshold rebalancing
- transaction cost redesign
- volatility targeting
- NLP
- regime detection
- sentiment analysis

Current limitations:

- dashboard benchmark comparison currently uses `sample` covariance in the comparison section for simplicity and consistency with the existing app flow
- there is no persistent benchmark configuration object yet
- no export layer for benchmark comparison tables has been added

---

## 10. Conclusion

Stage 11 adds the missing benchmark structure to the platform.

The system can now:

1. compare Equal Weight, Inverse Volatility, HRP, and HERC side by side
2. measure performance relative to an explicit benchmark
3. visualize multi-strategy growth curves and drawdowns
4. answer strategy questions in benchmark-relative terms rather than isolated metric snapshots

This closes a major research gap by turning the platform from a set of standalone strategy runs into a benchmark-aware comparison engine.
