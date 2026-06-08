# Stage 13 Report

## Files Created

- `src/backtesting/volatility_targeting.py`
- `src/data_pipeline/defensive_assets.py`
- `tests/test_volatility_targeting.py`
- `tests/test_defensive_assets.py`
- `notebooks/13_adaptive_volatility_targeting/stage_13_adaptive_volatility_targeting.ipynb`
- `STAGE_13_REPORT.md`

## Files Modified

- `src/backtesting/__init__.py`
- `src/data_pipeline/__init__.py`
- `src/dashboard/app.py`
- `src/dashboard/plots.py`

## Volatility Targeting Methodology

Stage 13 adds a rule-based overlay on top of an existing risky strategy return stream.

The targeted return series is:

```text
r_targeted_t =
exposure_t * risky_strategy_return_t
+
(1 - exposure_t) * defensive_asset_return_t
```

Exposure is computed from lagged realized volatility:

```text
exposure_t = clip(target_vol_t / realized_vol_{t-1}, exposure_floor, exposure_cap)
```

Then a no-trade band is applied:

- if the new exposure differs from the prior exposure by less than `no_trade_band`
- keep the prior exposure

This reduces exposure churn without changing the base portfolio construction logic.

## Defensive Sleeve Design

The defensive sleeve is sourced through a standalone module:

- preferred ticker first
- fallback ticker second
- synthetic risk-free series if both fail

Default candidates:

- `LIQUIDBEES.NS`
- `LIQUIDETF.NS`

Fallback synthetic return:

- annual rate `0.04`
- daily rate `0.04 / 252`

## Why the Defensive Ticker Is Treated Separately

The defensive asset is intentionally excluded from the risky universe:

- it is not part of covariance estimation
- it is not passed into clustering
- it is not part of HRP or HERC allocation
- it is not subject to risky-universe asset-dropping rules

This keeps the overlay conceptually clean:

- the base allocator decides the risky portfolio
- the overlay decides how much of that risky portfolio to hold

## Regime-Specific Target Volatility Logic

Realized volatility is computed from risky portfolio returns using a rolling annualized window.

Then the current regime is classified from the percentile rank of lagged realized volatility against historical lagged realized volatility:

- `calm`: percentile `<= 40%`, target vol `12%`
- `normal`: `40% < percentile <= 80%`, target vol `10%`
- `stress`: `80% < percentile <= 95%`, target vol `6%`
- `crisis`: percentile `> 95%`, target vol `3%`

The dashboard also supports a fixed target mode by flattening all regime targets to the chosen base target volatility.

## No-Look-Ahead Safeguards

The overlay uses only information available up to `t-1`:

- realized volatility is shifted by one day before exposure is computed
- regime classification is based on lagged realized volatility
- current-day exposure is applied to current-day return only after that lagged calculation

This was explicitly tested by altering future returns and confirming past exposure values remain unchanged.

## Dashboard Additions

Added sidebar controls:

- enable volatility targeting
- defensive sleeve selection
- synthetic annual rate
- vol target mode (`Adaptive`, `Fixed`)
- base target vol
- realized vol window
- regime lookback window
- exposure floor
- exposure cap
- no-trade band

Added section:

- `Adaptive Volatility Targeting`

Outputs:

- base vs targeted growth
- exposure path
- realized vs target volatility
- regime timeline
- defensive allocation
- summary table
- defensive sleeve metadata
- overlay diagnostics table

## Notebook Findings

The Stage 13 notebook is structured to answer:

1. whether adaptive volatility targeting reduces drawdown
2. whether Calmar improves
3. how much CAGR is sacrificed
4. how much time is spent in each regime
5. how much capital moves into the defensive sleeve
6. whether HERC plus targeting improves on plain HERC drawdown

The notebook uses the existing rolling backtester for the base risky strategy, then applies the overlay as a second layer.

## Test Results

Executed:

```bash
.\.venv\Scripts\python.exe -m pytest tests\test_defensive_assets.py tests\test_volatility_targeting.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_turnover.py tests\test_rebalance_rules.py tests\test_backtest_diagnostics.py tests\test_backtesting.py tests\test_strategy_comparison.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_phase2a_integration.py tests\test_hrp.py tests\test_herc_allocator.py tests\test_risk_contribution.py tests\test_benchmark_factory.py tests\test_data_pipeline.py -q
```

Results:

- focused Stage 13 overlay tests passed
- existing Phase 2B backtesting tests remained green
- existing Phase 2A and data pipeline regressions remained green

## Limitations

- defensive sleeve trading costs are not modeled separately from the base backtest
- fixed and adaptive targeting are controlled at the dashboard layer, not a benchmark-comparison factory layer
- regime logic is percentile-based and intentionally simple
- no volatility forecast model is used
- no leverage is allowed

## Future Extension Points

- ML regime classifier
- macro regime detection
- dynamic defensive asset selection
- leverage support
- volatility forecast models
