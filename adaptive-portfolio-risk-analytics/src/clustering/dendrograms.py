"""Dendrogram generation utilities."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram


def plot_dendrogram(linkage_matrix, labels: Iterable[str] | None = None):
    """Generate a publication-quality dendrogram figure."""
    if labels is not None:
        labels = list(labels)

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(
        linkage_matrix,
        labels=labels,
        orientation="top",
        leaf_rotation=45,
        leaf_font_size=10,
        color_threshold=None,
        ax=ax,
    )

    ax.set_title("Hierarchical Clustering Dendrogram")
    ax.set_xlabel("Asset")
    ax.set_ylabel("Distance")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig
