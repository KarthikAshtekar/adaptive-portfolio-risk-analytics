# Stage 4 Report: Hierarchical Clustering & Dendrograms

## Objective

Use the Stage 3 distance matrix to discover groups of similar assets and prepare the structure required for Hierarchical Risk Parity.

## What Was Implemented

- Added hierarchical clustering utilities in `src/clustering/hierarchical.py`:
  - `compute_linkage_matrix(distance_matrix_df, method="ward")`
  - `assign_clusters(linkage_matrix, n_clusters)`
  - `get_cluster_members(assets, cluster_labels)`
- Added dendrogram rendering in `src/clustering/dendrograms.py`:
  - `plot_dendrogram(linkage_matrix, labels)`
- Updated `src/clustering/__init__.py` to expose the new Stage 4 public API.
- Added deterministic tests in `tests/test_clustering.py`.
- Created the Stage 4 notebook:
  - `notebooks/04_hierarchical_clustering/stage_04_hierarchical_clustering.ipynb`
- Documented Stage 4 progress in `STAGE_4_REPORT.md`.

## Files Added / Modified

- `src/clustering/hierarchical.py`
- `src/clustering/dendrograms.py`
- `src/clustering/__init__.py`
- `tests/test_clustering.py`
- `notebooks/04_hierarchical_clustering/stage_04_hierarchical_clustering.ipynb`
- `STAGE_4_REPORT.md`

## Tests Added

- `test_compute_linkage_matrix_generates_valid_matrix`
- `test_compute_linkage_matrix_all_supported_methods`
- `test_assign_clusters_returns_integer_labels`
- `test_assign_clusters_every_asset_assigned_once`
- `test_get_cluster_members_returns_expected_structure`
- `test_plot_dendrogram_returns_figure`

## Dendrogram Observations

- Ward linkage tends to produce compact, balanced clusters by minimizing variance within groups.
- Average linkage can yield more elongated clusters depending on pairwise distances.
- Comparing dendrograms is useful to understand whether the asset universe contains clear binary splits or more gradual grouping structure.

## Cluster Assignments

- Assets are assigned to clusters via `assign_clusters(linkage_matrix, n_clusters)`.
- `get_cluster_members` returns a dictionary keyed by cluster label with ordered asset lists.
- This output is intentionally simple and stage-appropriate: it does not compute weights or apply HRP allocation.

## Cluster Interpretation

- Clustering identifies groups of assets with similar pairwise distances, which extends the pairwise correlation information into a group-level structure.
- Two assets can belong to the same cluster because they share a similar distance profile relative to the full universe, not just because they are directly highly correlated.
- Clusters reveal hidden structure by aggregating information across the entire distance geometry.

## Readiness for HRP

- The stage produces the core cluster structure needed for HRP: a linkage tree and explicit cluster memberships.
- It stops short of Stage 5 work, leaving weight allocation and portfolio construction for the next phase.

## Notes

- The notebook includes a fallback path to recompute Stage 3 outputs only when pre-saved stage artifacts are unavailable.
- This stage avoids all HRP-specific allocation logic by design.
