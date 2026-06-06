"""Unit tests for the HERC allocator and related comparison helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting import RollingBacktester
from src.clustering import (
    HERCAllocator,
    compare_hrp_herc_weights,
    validate_weights,
)


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    dates = pd.date_range(start="2020-01-01", periods=360, freq="B")
    data = rng.normal(
        loc=0.0004,
        scale=np.array([0.009, 0.012, 0.015, 0.018, 0.011]),
        size=(len(dates), 5),
    )
    return pd.DataFrame(data, index=dates, columns=["A", "B", "C", "D", "E"])


@pytest.mark.parametrize(
    "covariance_method",
    ["sample", "ledoit_wolf", "ewma", "ewma_ledoit_wolf"],
)
def test_herc_weights_are_long_only_and_fully_invested(
    sample_returns: pd.DataFrame,
    covariance_method: str,
) -> None:
    allocator = HERCAllocator(covariance_method=covariance_method)
    weights = allocator.optimize(sample_returns)

    assert isinstance(weights, pd.Series)
    assert weights.shape == (sample_returns.shape[1],)
    assert weights.index.tolist() == sample_returns.columns.tolist()
    assert weights.name == "weight"
    assert np.isclose(weights.sum(), 1.0, atol=1e-8)
    assert np.all(weights >= 0.0)
    assert np.isfinite(weights).all()


def test_herc_output_shape_matches_assets(sample_returns: pd.DataFrame) -> None:
    allocator = HERCAllocator(covariance_method="ledoit_wolf")
    weights = allocator.optimize(sample_returns)

    assert len(weights) == sample_returns.shape[1]
    assert weights.index.tolist() == sample_returns.columns.tolist()


def test_validate_weights_normalizes_valid_series() -> None:
    weights = pd.Series([0.25, 0.25, 0.50], index=["A", "B", "C"], dtype=float)
    validated = validate_weights(weights)

    assert np.isclose(float(validated.sum()), 1.0, atol=1e-8)
    assert (validated >= 0.0).all()


def test_validate_weights_rejects_nans() -> None:
    weights = pd.Series([0.5, np.nan, 0.5], index=["A", "B", "C"], dtype=float)

    with pytest.raises(ValueError):
        validate_weights(weights)


def test_compute_cluster_risk_positive(sample_returns: pd.DataFrame) -> None:
    allocator = HERCAllocator()
    covariance_matrix = sample_returns.cov()
    cluster_risk = allocator.compute_cluster_risk(covariance_matrix, ["A", "B", "C"])

    assert cluster_risk > 0.0


def test_compare_hrp_herc_weights_returns_expected_columns(
    sample_returns: pd.DataFrame,
) -> None:
    comparison_df = compare_hrp_herc_weights(
        sample_returns,
        covariance_method="ledoit_wolf",
    )

    assert list(comparison_df.columns) == [
        "Asset",
        "HRP Weight",
        "HERC Weight",
        "Difference",
    ]
    assert comparison_df["Asset"].tolist() == sample_returns.columns.tolist()
    assert np.isclose(comparison_df["HRP Weight"].sum(), 1.0, atol=1e-8)
    assert np.isclose(comparison_df["HERC Weight"].sum(), 1.0, atol=1e-8)


def test_herc_backtester_integration(sample_returns: pd.DataFrame) -> None:
    backtester = RollingBacktester(
        allocator=HERCAllocator(covariance_method="ewma"),
        train_window=126,
        rebalance_frequency="M",
    )
    results = backtester.run(sample_returns)

    assert not results["weights_history"].empty
    assert np.allclose(results["weights_history"].sum(axis=1).values, 1.0)
    assert (results["weights_history"] >= 0.0).all().all()


def test_dashboard_strategy_selector_exposes_herc() -> None:
    from src.dashboard.app import get_allocator

    allocator = get_allocator("HERC")

    assert isinstance(allocator, HERCAllocator)
