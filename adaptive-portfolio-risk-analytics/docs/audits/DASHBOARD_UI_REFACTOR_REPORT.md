# Dashboard UI Refactor Report

## Files Modified

- `src/dashboard/app.py`

## Layout Changes

- Replaced the long flat sidebar with grouped `st.sidebar.expander()` sections:
  - `Basic Portfolio Setup`
  - `Strategy & Backtest Settings`
  - `Advanced Risk Controls`
  - `Experiment Sensitivity`
- Kept only `Basic Portfolio Setup` expanded by default.
- Moved the main page output into tabs:
  - `Portfolio Overview`
  - `Backtest Results`
  - `Risk & Allocation`
  - `Trading Activity`
  - `Volatility Targeting`
  - `Experiment Sensitivity`

## Ticker Selector Improvements

- Added a predefined Indian research universe in the dashboard layer.
- Added universe presets:
  - `Core Diversified`
  - `Banks + IT + Gold`
  - `Full Research Universe`
  - `Custom`
- Added a `Select All Assets In Preset` toggle.
- Replaced the primary raw ticker entry flow with a searchable multiselect showing:
  - `TICKER — Company Name`
- Kept a collapsed `Manual Ticker Override` fallback.
- Manual override, when provided, takes precedence over multiselect selection.

## Validation Added

- at least 2 risky assets must be selected
- start date must be before end date
- exposure floor must be less than or equal to exposure cap
- defensive sleeve must remain separate from the risky universe
- sensitivity max runs must be positive
- sensitivity thresholds must parse correctly
- sensitivity thresholds must stay between 0 and 1

Validation errors are surfaced with `st.warning` or `st.error` rather than crashing the app.

## Tests / Checks Run

Executed:

```bash
python -m py_compile src/dashboard/app.py
python -m pytest tests/test_experiment_config.py tests/test_experiment_runner.py tests/test_sensitivity.py -q
python -m pytest tests/test_defensive_assets.py tests/test_volatility_targeting.py -q
python -m pytest tests/test_backtesting.py tests/test_strategy_comparison.py -q
```

Observed results:

- `py_compile` passed
- experiment tests passed
- defensive asset and volatility targeting tests passed
- backtesting and strategy comparison tests passed

## Remaining UI Limitations

- the ticker universe is still hardcoded in `app.py`
- the dashboard remains a single-file Streamlit app and would benefit from further component extraction
- sensitivity study still shares some global sidebar controls to avoid duplicating orchestration inputs
- no dedicated UI tests were added in this refactor
