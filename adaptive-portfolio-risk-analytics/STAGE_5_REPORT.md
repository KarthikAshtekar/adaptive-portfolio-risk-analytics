# Stage 5 Report: Hierarchical Risk Parity

## Files Added / Updated
- `src/optimization/hrp_allocator.py`
- `src/optimization/__init__.py`
- `tests/test_hrp.py`
- `notebooks/05_hrp_portfolio_construction/stage_05_hrp_portfolio_construction.ipynb`
- `STAGE_5_REPORT.md`

## Functions Added
- `get_quasi_diagonal_order(linkage_matrix)`
- `compute_cluster_variance(covariance_matrix, cluster_assets)`
- `recursive_bisection(covariance_matrix, ordered_assets)`
- `allocate_hrp_weights(covariance_matrix_df, linkage_matrix)`

## What Was Implemented
- A complete HRP allocation pipeline in `src/optimization/hrp_allocator.py`.
- Utilities that use the covariance matrix and the linkage matrix from Stage 4 without recomputing hierarchical clustering unnecessarily.
- An `HRPAllocator` adapter that can consume returns, an optional covariance matrix, and an optional linkage matrix.
- Stage 5 notebook demonstrating HRP construction, recursive allocation steps, and comparison against baseline methods.

## Final HRP Weights
Final weights from the Stage 5 HRP pipeline are:

- Asset A: 0.186064
- Asset B: 0.142724
- Asset C: 0.144100
- Asset D: 0.173013
- Asset E: 0.189329
- Asset F: 0.164770

The allocation is normalized to sum to 1 and assigns a non-negative weight to every asset.

## Cluster Allocations
- Recursive bisection divides the ordered asset hierarchy into left and right sub-clusters.
- Each branch receives capital inversely proportional to cluster variance.
- The notebook shows intermediate cluster allocations and the full allocation path.

## Comparison with Benchmarks
- Equal Weight: naive 1/N allocation.
- Inverse Volatility: risk parity by individual asset volatility.
- HRP: hierarchical risk parity that accounts for both covariance structure and cluster relationships.

## Observations
- HRP provides a structured allocation that is sensitive to the covariance-based clustering hierarchy.
- It is more robust than pure equal weighting because it respects asset similarity and cluster risk.
- Compared to inverse volatility, HRP can reduce concentration risk in highly correlated clusters.
- The notebook includes visual comparisons and interpretation of the differences.
