"""Dendrogram generation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram


class DendrogramAnalyzer:
    """Generate dendrogram diagnostics and figures."""

    @staticmethod
    def to_plotly_figure(
        linkage_matrix: np.ndarray,
        labels: list[Any] | None = None,
        title: str = "Hierarchical Clustering Dendrogram",
    ) -> go.Figure:
        """Convert scipy dendrogram output to a Plotly figure."""
        dendro = dendrogram(linkage_matrix, labels=labels, no_plot=True)

        fig = go.Figure()
        for xs, ys in zip(dendro["icoord"], dendro["dcoord"]):
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color="#1f77b4", width=2),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        tick_vals = [5 + 10 * i for i in range(len(dendro.get("ivl", [])))]
        fig.update_layout(
            title=title,
            xaxis=dict(
                title="Assets",
                tickmode="array",
                tickvals=tick_vals,
                ticktext=dendro.get("ivl", []),
            ),
            yaxis=dict(title="Distance"),
            template="plotly_white",
            margin=dict(l=40, r=20, t=50, b=40),
        )
        return fig
