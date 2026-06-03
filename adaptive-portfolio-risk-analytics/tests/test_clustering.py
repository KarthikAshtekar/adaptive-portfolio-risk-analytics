import numpy as np
import pandas as pd

from src.clustering import (
    compute_linkage_matrix,
    assign_clusters,
    get_cluster_members,
    plot_dendrogram,
    SUPPORTED_LINKAGE_METHODS,
)


def make_test_distance_matrix() -> pd.DataFrame:
    assets = ["A", "B", "C", "D"]
    distance = pd.DataFrame(
        [
            [0.0, 0.2, 0.8, 0.7],
            [0.2, 0.0, 0.9, 0.6],
            [0.8, 0.9, 0.0, 0.4],
            [0.7, 0.6, 0.4, 0.0],
        ],
        index=assets,
        columns=assets,
    )
    return distance


def test_compute_linkage_matrix_generates_valid_matrix():
    distance = make_test_distance_matrix()
    linkage_matrix = compute_linkage_matrix(distance, method="average")

    assert linkage_matrix.shape == (len(distance) - 1, 4)
    assert np.isfinite(linkage_matrix).all()


def test_compute_linkage_matrix_all_supported_methods():
    distance = make_test_distance_matrix()
    for method in SUPPORTED_LINKAGE_METHODS:
        matrix = compute_linkage_matrix(distance, method=method)
        assert matrix.shape == (len(distance) - 1, 4)
        assert np.isfinite(matrix).all()


def test_assign_clusters_returns_integer_labels():
    distance = make_test_distance_matrix()
    linkage_matrix = compute_linkage_matrix(distance, method="complete")
    labels = assign_clusters(linkage_matrix, n_clusters=2)

    assert len(labels) == len(distance)
    assert labels.dtype.kind in {"i", "u"}
    assert set(labels).issubset({1, 2})


def test_assign_clusters_every_asset_assigned_once():
    distance = make_test_distance_matrix()
    linkage_matrix = compute_linkage_matrix(distance, method="single")
    labels = assign_clusters(linkage_matrix, n_clusters=3)

    assert len(labels) == len(distance)
    assert sorted(labels.tolist()).count(1) + sorted(labels.tolist()).count(2) + sorted(labels.tolist()).count(3) == len(distance)


def test_get_cluster_members_returns_expected_structure():
    assets = ["A", "B", "C", "D"]
    cluster_labels = np.array([1, 1, 2, 2], dtype=int)
    members = get_cluster_members(assets, cluster_labels)

    assert isinstance(members, dict)
    assert set(members.keys()) == {1, 2}
    assert members[1] == ["A", "B"]
    assert members[2] == ["C", "D"]


def test_plot_dendrogram_returns_figure():
    distance = make_test_distance_matrix()
    linkage_matrix = compute_linkage_matrix(distance, method="average")
    fig = plot_dendrogram(linkage_matrix, labels=distance.index.tolist())

    assert fig is not None
    assert hasattr(fig, "savefig")
