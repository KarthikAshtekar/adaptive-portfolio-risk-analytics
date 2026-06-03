# Stage 2 Report: Returns and Risk

## Objective

Transform Stage 1 price data into portfolio-relevant return and volatility information without changing the runtime Yahoo Finance acquisition flow.

## What Was Implemented

- Extended [src/data_pipeline/preprocess.py](/d:/PGDBA/MyNotes/SEM-3/Part-1/FRM/Project/adaptive-portfolio-risk-analytics/src/data_pipeline/preprocess.py) with Stage 2 utilities for:
  - simple return calculation
  - log return calculation
  - simple vs log return comparison
  - daily volatility
  - annualized volatility
  - 30-day rolling annualized volatility
  - 90-day rolling annualized volatility
- Added `ReturnsRiskOutputs` so Stage 2 outputs are grouped consistently.
- Set `returns_df` to the log-return series by default for downstream use.
- Kept the Stage 1 data acquisition layer unchanged.
- Created and executed the learning notebook at [notebooks/02_returns_and_risk/stage_02_returns_and_risk.ipynb](/d:/PGDBA/MyNotes/SEM-3/Part-1/FRM/Project/adaptive-portfolio-risk-analytics/notebooks/02_returns_and_risk/stage_02_returns_and_risk.ipynb).

## Files Modified

- [src/data_pipeline/preprocess.py](/d:/PGDBA/MyNotes/SEM-3/Part-1/FRM/Project/adaptive-portfolio-risk-analytics/src/data_pipeline/preprocess.py)
- [src/data_pipeline/__init__.py](/d:/PGDBA/MyNotes/SEM-3/Part-1/FRM/Project/adaptive-portfolio-risk-analytics/src/data_pipeline/__init__.py)
- [tests/test_returns_risk.py](/d:/PGDBA/MyNotes/SEM-3/Part-1/FRM/Project/adaptive-portfolio-risk-analytics/tests/test_returns_risk.py)
- [notebooks/02_returns_and_risk/stage_02_returns_and_risk.ipynb](/d:/PGDBA/MyNotes/SEM-3/Part-1/FRM/Project/adaptive-portfolio-risk-analytics/notebooks/02_returns_and_risk/stage_02_returns_and_risk.ipynb)
- [STAGE_2_REPORT.md](/d:/PGDBA/MyNotes/SEM-3/Part-1/FRM/Project/adaptive-portfolio-risk-analytics/STAGE_2_REPORT.md)

## Tests Added

- deterministic simple return calculation
- deterministic log return calculation
- first observation removal
- no look-ahead bias in return construction
- daily volatility calculation using sample standard deviation
- annualization logic validation
- rolling volatility validation across multiple windows
- default downstream output validation for log returns

## Validation Performed

### Automated Tests

- Command run: `.\.venv\Scripts\python.exe -m pytest tests/test_returns_risk.py tests/test_data_pipeline.py -q`
- Result: `14 passed`

### Notebook Execution

- Command run: `.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks/02_returns_and_risk/stage_02_returns_and_risk.ipynb --ExecutePreprocessor.kernel_name=apra-project-venv`
- Result: notebook executed successfully and saved with outputs

### Live Runtime Validation

- Universe: `HDFCBANK.NS`, `TCS.NS`, `GOLDBEES.NS`
- Requested range: `2022-01-01` to `2024-12-31`
- `simple_returns_df` shape: `(737, 3)`
- `log_returns_df` shape: `(737, 3)`
- `returns_df` shape: `(737, 3)`
- `rolling_volatility_df` shape: `(737, 6)`

Volatility summary from live data:

| Asset | Daily Volatility | Annualized Volatility |
| --- | ---: | ---: |
| `GOLDBEES.NS` | 0.007500 | 0.119059 |
| `HDFCBANK.NS` | 0.013805 | 0.219153 |
| `TCS.NS` | 0.013173 | 0.209110 |

## Visualizations Created

- simple returns time series
- log returns time series
- 30-day rolling annualized volatility plot
- 90-day rolling annualized volatility plot
- simple return distribution histograms
- log return distribution histograms

## Outputs Generated

- `simple_returns_df`
- `returns_df`
- `log_returns_df`
- `volatility_summary_df`
- `rolling_volatility_df`
- `return_comparison_df`

`returns_df` is intentionally defined as the log-return panel for downstream stages.

## Key Findings

- Daily simple and log returns are very similar for the sample universe, with correlations above `0.9997` for all three assets.
- `HDFCBANK.NS` had the highest annualized volatility in the sample period, followed closely by `TCS.NS`.
- `GOLDBEES.NS` was materially less volatile than the two equity assets.
- Rolling volatility shows that risk changes over time, which means later covariance and correlation estimates must be based on returns, not static price levels.

## Explanation

### Why prices are not used directly in portfolio optimization

Raw prices are not comparable across assets. A stock trading at `2000` and an ETF trading at `50` do not imply different risk solely because one has a larger price level. Portfolio optimization needs relative movement, not absolute denomination.

### Why returns are used

Returns convert price movements into a normalized scale. This makes assets comparable, allows aggregation across securities, and provides the right input for volatility, covariance, correlation, and portfolio-weight calculations.

### Difference between simple returns and log returns

- Simple return: percentage change from one period to the next.
- Log return: natural log of the price ratio between two periods.

Simple returns are easier to interpret directly. Log returns are preferred for downstream modeling because they add across time more naturally and behave better in many statistical workflows. For that reason, Stage 2 sets `returns_df` equal to `log_returns_df`.

### Why volatility is considered a risk measure

Volatility measures the dispersion of returns. Larger dispersion means portfolio outcomes are less predictable and can deviate more sharply from expectations. While volatility is not the only notion of risk, it is a core and widely used one in portfolio construction.

### Why Stage 2 is required before covariance and correlation analysis

Covariance and correlation are defined on return series, not raw prices. Without Stage 2, later risk-engine calculations would be distorted by non-stationary price levels and asset denomination effects. Stage 2 creates the normalized return inputs that Stage 3 will eventually depend on.

## Risks / Issues Discovered

- Notebook execution emits a non-blocking Windows `zmq` event-loop warning during `nbconvert`.
- Rolling volatility uses annualized rolling standard deviation, which is appropriate for comparison but should be documented consistently in later stages.
- The repository still contains later-stage modules built earlier in the project lifecycle; they were not changed in this stage.

## Stop Condition

Stage 2 is complete.

No Stage 3 or later work was started.
