# Stage 1 Report: Data Understanding

## Objective

Build and verify the runtime data acquisition layer using Yahoo Finance without introducing persistent dataset storage.

## What Was Implemented

- Added a provider-style runtime data interface in [src/data_pipeline/ingest.py](../../src/data_pipeline/ingest.py).
- Introduced `DataProvider` and `YahooFinanceProvider` for on-demand downloads.
- Preserved backward compatibility through `YFinanceIngester.fetch()` so existing callers still receive price-only output.
- Added `MarketDataBundle` to carry:
  - `prices_df`
  - `volume_df`
  - raw download payload
  - price field metadata
- Added `build_data_inspection_table()` to verify:
  - dates
  - adjusted close coverage
  - volume coverage
  - missing values
- Updated [.gitignore](../../.gitignore) so notebooks under `notebooks/` can be tracked while checkpoint files remain ignored.
- Defined the Stage 1 sample universe:
  - `HDFCBANK.NS`
  - `TCS.NS`
  - `GOLDBEES.NS`
- Created the exploratory notebook at [notebooks/01_data_exploration/stage_01_data_exploration.ipynb](../../notebooks/01_data_exploration/stage_01_data_exploration.ipynb).

## Files Modified

- [src/data_pipeline/ingest.py](../../src/data_pipeline/ingest.py)
- [src/data_pipeline/__init__.py](../../src/data_pipeline/__init__.py)
- [tests/test_data_pipeline.py](../../tests/test_data_pipeline.py)
- [notebooks/01_data_exploration/stage_01_data_exploration.ipynb](../../notebooks/01_data_exploration/stage_01_data_exploration.ipynb)
- [.gitignore](../../.gitignore)
- [STAGE_1_REPORT.md](STAGE_1_REPORT.md)

## Tests Added

- Provider returns both adjusted prices and volume for multi-asset downloads.
- Legacy `fetch()` behavior still returns price-only output.
- Provider falls back from `Adj Close` to `Close` when adjusted prices are unavailable.
- Inspection table correctly counts missing prices and missing volume.

## Validation Performed

### Automated Tests

- Command run: `pytest tests/test_data_pipeline.py -q`
- Result: `7 passed`

### Live Runtime Validation

- Provider: `YahooFinanceProvider`
- Universe: `HDFCBANK.NS`, `TCS.NS`, `GOLDBEES.NS`
- Requested range: `2022-01-01` to `2024-12-31`
- Price field used: `Adj Close`
- `prices_df` shape: `(738, 3)`
- `volume_df` shape: `(738, 3)`

Live inspection summary:

| Symbol | Start Date | End Date | Price Obs | Volume Obs | Missing Prices | Missing Volume |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `GOLDBEES.NS` | `2022-01-03` | `2024-12-30` | 738 | 738 | 0 | 0 |
| `HDFCBANK.NS` | `2022-01-03` | `2024-12-30` | 738 | 738 | 0 | 0 |
| `TCS.NS` | `2022-01-03` | `2024-12-30` | 738 | 738 | 0 | 0 |

## Outputs Generated

The Stage 1 notebook generates the following in memory from live downloads:

- `prices_df`
- `volume_df`
- `inspection_df`
- descriptive statistics for prices
- descriptive statistics for volume
- missing-value summary
- normalized price chart
- missing-value heatmaps

No persistent local datasets, CSV repositories, or ETL outputs were created.

## Explanation

Stage 1 focuses on data understanding before any return or optimization logic. The new provider layer keeps acquisition explicit and lightweight:

- user selects assets
- user selects date range
- data is fetched on demand
- processing happens in memory
- outputs are ready for later analytics stages

Using adjusted close prices ensures downstream return calculations reflect corporate actions more accurately than raw close alone. Volume is included now so we can validate completeness and basic tradability before moving into Stage 2.

## Remaining Work

- Stage 2: compute simple returns and log returns from `prices_df`
- compare return definitions
- calculate volatility and rolling volatility
- add Stage 2 visualizations and report

## Risks / Issues Discovered

- Live runtime validation depends on Yahoo Finance availability and network access.
- The local environment did not initially have `yfinance` installed even though it is declared in `requirements.txt`; it was installed during validation.
- The repository already contains later-stage modules. They were intentionally left untouched to preserve the staged implementation flow.
- No cache layer has been added yet. This matches the brief, but repeated notebook runs will trigger fresh downloads.
