# Stage 3 Report

## Objective

Implement covariance, correlation, and distance analysis for portfolio diversification and future clustering.

## Files Added / Modified

- src/covariance/covariance.py
- src/covariance/distance.py
- src/covariance/__init__.py
- tests/test_covariance.py
- notebooks/03_correlation_covariance/stage_03_correlation_covariance.ipynb

## Outputs Generated

- covariance_matrix_df
- correlation_matrix_df
- distance_matrix_df
- correlation_rankings_df
- average_correlation

## Key Findings

Highest Correlation Pair:
HDFCBANK.NS ↔ TCS.NS = 0.318

Lowest Correlation Pair:
GOLDBEES.NS ↔ HDFCBANK.NS = -0.037

Average Correlation:
0.1138

## Portfolio Insights

- Gold exhibited near-zero correlation with equity assets.
- Diversification benefits arise from low correlations rather than simply holding many assets.
- Distance matrices provide the foundation for hierarchical clustering.

## Stage 4 Readiness

Distance matrix is available for hierarchical clustering and dendrogram generation.