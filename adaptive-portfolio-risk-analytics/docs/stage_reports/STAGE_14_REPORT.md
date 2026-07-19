# Stage 14 Report

## Files Created

- `src/experiments/__init__.py`
- `src/experiments/config.py`
- `src/experiments/runner.py`
- `src/experiments/sensitivity.py`
- `src/experiments/reporting.py`
- `tests/test_experiment_config.py`
- `tests/test_experiment_runner.py`
- `tests/test_sensitivity.py`
- `notebooks/14_experiment_sensitivity/stage_14_experiment_sensitivity.ipynb`
- `STAGE_14_REPORT.md`

## Files Modified

- `src/dashboard/app.py`
- `src/dashboard/plots.py`

## Experiment Framework Design

Stage 14 adds a research orchestration layer around the existing platform.

The design does not introduce:

- new allocators
- new covariance estimators
- new backtesting mechanics

Instead it coordinates the existing stack:

- `BenchmarkFactory`
- `RollingBacktester`
- transaction cost engine
- volatility targeting overlay
- performance analytics

The framework is built from four layers:

1. configuration dataclass
2. parameter-grid generator
3. single-run and grid runner
4. sensitivity/reporting helpers

Each experiment run produces one clean result row with both configuration metadata and portfolio metrics.

## Parameters Supported

The experiment grid supports:

- strategy
- covariance method
- rebalance mode
- threshold
- transaction cost bps
- slippage bps
- volatility targeting enabled or disabled
- target volatility
- defensive asset
- train window
- initial capital

The default Phase 2D grid remains intentionally small enough to run locally.

## Sensitivity Methods

Implemented helpers:

- `rank_experiments()`
- `summarize_by_parameter()`
- `compute_parameter_sensitivity()`

Supported objectives:

- `cagr`
- `sharpe`
- `sortino`
- `calmar`
- `max_drawdown`
- `final_value`

For `max_drawdown`, less negative is treated as better by sorting in descending numeric order.

## Reporting and Storage

Implemented reporting helpers:

- `build_experiment_summary_table()`
- `build_top_n_table()`
- `build_parameter_pivot()`
- `export_experiment_results()`

Export location:

- `outputs/experiments/`

Stored artifacts are summary/result tables only.
No raw market data is exported.

## Optional MLflow Integration

Implemented:

- `log_experiment_to_mlflow()`

Behavior:

- attempts to import `mlflow`
- logs parameters and numeric metrics when available
- skips gracefully if `mlflow` is not installed

MLflow remains optional and does not affect experiment execution.

## Dashboard Additions

Added a lightweight `Experiment Sensitivity` section to the dashboard.

Inputs:

- strategies
- covariance methods
- rebalance modes
- thresholds
- objective metric
- max runs

Outputs:

- experiment results table
- top 10 configurations
- metric by covariance method
- metric by rebalance mode
- sensitivity heatmap
- parameter sensitivity summary

The existing portfolio workflow remains intact.

## Notebook Findings

Notebook:

- `notebooks/14_experiment_sensitivity/stage_14_experiment_sensitivity.ipynb`

Sections:

1. Load data
2. Build experiment config
3. Generate parameter grid
4. Run grid
5. Rank experiments
6. Summarize by strategy
7. Summarize by covariance method
8. Summarize by rebalance mode
9. Sensitivity heatmaps
10. Top 10 configurations
11. Interpretation

The notebook is designed to answer:

- which strategy performs best by Calmar
- which strategy performs best by Sharpe
- which covariance method is most robust
- whether volatility targeting improves drawdown
- whether threshold rebalancing reduces turnover without hurting returns too much
- which parameters matter most

## Test Results

Executed:

```bash
.\.venv\Scripts\python.exe -m py_compile src\experiments\__init__.py src\experiments\config.py src\experiments\runner.py src\experiments\sensitivity.py src\experiments\reporting.py src\dashboard\app.py src\dashboard\plots.py
.\.venv\Scripts\python.exe -m pytest tests\test_experiment_config.py tests\test_experiment_runner.py tests\test_sensitivity.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_experiment_config.py tests\test_experiment_runner.py tests\test_sensitivity.py tests\test_defensive_assets.py tests\test_volatility_targeting.py tests\test_data_pipeline.py tests\test_turnover.py tests\test_rebalance_rules.py tests\test_backtest_diagnostics.py tests\test_backtesting.py tests\test_strategy_comparison.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_phase2a_integration.py tests\test_hrp.py tests\test_herc_allocator.py tests\test_risk_contribution.py tests\test_benchmark_factory.py -q
```

Results:

- experiment-specific tests passed
- Stage 13 defensive sleeve and vol-targeting tests remained green
- Stage 12 realistic backtesting and strategy comparison tests remained green
- Phase 2A HRP/HERC and benchmark regressions remained green

## Limitations

- dashboard sensitivity study reuses a small grid and shared global controls to stay responsive
- volatility targeting in experiment mode uses the existing overlay layer rather than a fully integrated multi-sleeve transaction-cost model
- parameter sensitivity is descriptive, not an optimizer
- MLflow logging is local and optional only
- no walk-forward robustness scoring is included yet

## Future Extensions

- Optuna optimization
- walk-forward parameter validation
- MLflow experiment server
- model registry
- multi-objective optimization
