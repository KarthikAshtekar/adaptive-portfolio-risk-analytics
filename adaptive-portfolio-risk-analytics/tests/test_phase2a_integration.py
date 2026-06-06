"""Phase 2A integration tests for covariance research and HERC."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting import RollingBacktester
from src.clustering import HERCAllocator as ClusteringHERCAllocator
from src.covariance import CovarianceFactory
from src.optimization import (
    EqualWeightAllocator,
    HERCAllocator as OptimizationHERCAllocator,
    HRPAllocator,
    InverseVolatilityAllocator,
)


@pytest.fixture
def deterministic_returns() -> pd.DataFrame:
    """Deterministic return sample for cross-module integration checks."""
    rng = np.random.default_rng(20260606)
    dates = pd.date_range(start="2021-01-01", periods=180, freq="B")
    common_factor = rng.normal(0.0002, 0.0045, size=(len(dates), 1))
    idiosyncratic = rng.normal(
        loc=0.0003,
        scale=np.array([0.007, 0.010, 0.012, 0.009, 0.011]),
        size=(len(dates), 5),
    )
    data = common_factor + idiosyncratic
    return pd.DataFrame(data, index=dates, columns=["A", "B", "C", "D", "E"])


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("sample", {}),
        ("ledoit_wolf", {}),
        ("ewma", {"span": 60}),
        ("ewma_ledoit_wolf", {"span": 60}),
    ],
)
def test_covariance_factory_routing_and_matrix_quality(
    deterministic_returns: pd.DataFrame,
    method: str,
    kwargs: dict[str, int],
) -> None:
    covariance_matrix = CovarianceFactory.compute(
        deterministic_returns,
        method=method,
        **kwargs,
    )

    assert isinstance(covariance_matrix, pd.DataFrame)
    assert covariance_matrix.shape == (
        deterministic_returns.shape[1],
        deterministic_returns.shape[1],
    )
    assert covariance_matrix.index.tolist() == deterministic_returns.columns.tolist()
    assert covariance_matrix.columns.tolist() == deterministic_returns.columns.tolist()
    assert np.allclose(covariance_matrix.values, covariance_matrix.values.T, atol=1e-10)
    assert (np.diag(covariance_matrix.values) > 0.0).all()
    assert not covariance_matrix.isna().any().any()
    assert "covariance_metadata" in covariance_matrix.attrs


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("sample", {}),
        ("ledoit_wolf", {}),
        ("ewma", {"span": 60}),
        ("ewma_ledoit_wolf", {"span": 60}),
    ],
)
def test_covariance_factory_metadata(
    deterministic_returns: pd.DataFrame,
    method: str,
    kwargs: dict[str, int],
) -> None:
    covariance_matrix = CovarianceFactory.compute(
        deterministic_returns,
        method=method,
        **kwargs,
    )
    metadata = covariance_matrix.attrs["covariance_metadata"]

    assert metadata["method"] == method
    if method == "ledoit_wolf":
        assert "shrinkage" in metadata
        assert 0.0 <= metadata["shrinkage"] <= 1.0
    if method == "ewma":
        assert metadata["span"] == 60
    if method == "ewma_ledoit_wolf":
        assert metadata["span"] == 60
        assert "shrinkage" in metadata
        assert 0.0 <= metadata["shrinkage"] <= 1.0


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("sample", {}),
        ("ledoit_wolf", {}),
        ("ewma", {"span": 60}),
        ("ewma_ledoit_wolf", {"span": 60}),
    ],
)
def test_hrp_covariance_method_support_returns_labeled_weights(
    deterministic_returns: pd.DataFrame,
    method: str,
    kwargs: dict[str, int],
) -> None:
    allocator = HRPAllocator(
        covariance_method=method,
        covariance_kwargs=kwargs,
    )
    weights = allocator.optimize(deterministic_returns)

    assert isinstance(weights, pd.Series)
    assert weights.index.tolist() == deterministic_returns.columns.tolist()
    assert weights.name == "weight"
    assert np.isclose(float(weights.sum()), 1.0, atol=1e-8)
    assert (weights >= 0.0).all()
    assert not weights.isna().any()
    assert np.isfinite(weights.values).all()


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("sample", {}),
        ("ledoit_wolf", {}),
        ("ewma", {"span": 60}),
        ("ewma_ledoit_wolf", {"span": 60}),
    ],
)
def test_herc_covariance_method_support_returns_labeled_weights(
    deterministic_returns: pd.DataFrame,
    method: str,
    kwargs: dict[str, int],
) -> None:
    allocator = ClusteringHERCAllocator(
        covariance_method=method,
        covariance_kwargs=kwargs,
    )
    weights = allocator.optimize(deterministic_returns)

    assert isinstance(weights, pd.Series)
    assert weights.index.tolist() == deterministic_returns.columns.tolist()
    assert weights.name == "weight"
    assert np.isclose(float(weights.sum()), 1.0, atol=1e-8)
    assert (weights >= 0.0).all()
    assert not weights.isna().any()
    assert np.isfinite(weights.values).all()


def test_herc_backtester_integration(deterministic_returns: pd.DataFrame) -> None:
    results = RollingBacktester(
        allocator=ClusteringHERCAllocator(covariance_method="ledoit_wolf"),
        train_window=60,
        rebalance_frequency="M",
    ).run(deterministic_returns)

    assert "portfolio_returns" in results
    assert "portfolio_values" in results
    assert "drawdown" in results
    assert "weights_history" in results
    assert "performance_metrics" in results
    assert not results["portfolio_returns"].empty
    assert not results["weights_history"].empty
    assert (results["portfolio_values"] > 0.0).all()
    assert np.allclose(results["weights_history"].sum(axis=1).values, 1.0, atol=1e-8)

    for metric in ("cagr", "sharpe", "volatility", "max_drawdown"):
        assert metric in results["performance_metrics"]


@pytest.mark.parametrize(
    "allocator",
    [
        EqualWeightAllocator(),
        InverseVolatilityAllocator(),
        HRPAllocator(),
    ],
)
def test_phase1_strategies_still_run_in_backtester(
    deterministic_returns: pd.DataFrame,
    allocator,
) -> None:
    results = RollingBacktester(
        allocator=allocator,
        train_window=60,
        rebalance_frequency="M",
    ).run(deterministic_returns)

    assert not results["portfolio_returns"].empty
    assert (results["portfolio_values"] > 0.0).all()
    for metric in ("cagr", "sharpe", "volatility", "max_drawdown"):
        assert metric in results["performance_metrics"]


def test_dashboard_strategy_factory_exposes_herc() -> None:
    from src.dashboard.app import get_allocator

    allocator = get_allocator("HERC")

    assert isinstance(allocator, ClusteringHERCAllocator)


def test_import_api_consistency() -> None:
    assert CovarianceFactory is not None
    assert ClusteringHERCAllocator is not None
    assert OptimizationHERCAllocator is not None
    assert OptimizationHERCAllocator is ClusteringHERCAllocator
