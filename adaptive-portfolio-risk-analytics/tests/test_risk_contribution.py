"""Unit tests for risk contribution analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics import (
    compare_risk_contributions,
    marginal_risk_contribution,
    percentage_risk_contribution,
    portfolio_volatility,
    risk_contribution_table,
    total_risk_contribution,
)
from src.optimization import EqualWeightAllocator, HERCAllocator, HRPAllocator, InverseVolatilityAllocator


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    rng = np.random.default_rng(2030)
    dates = pd.date_range(start="2021-01-01", periods=252, freq="B")
    data = rng.normal(
        loc=0.0004,
        scale=np.array([0.010, 0.013, 0.016, 0.009]),
        size=(len(dates), 4),
    )
    return pd.DataFrame(data, index=dates, columns=["A", "B", "C", "D"])


@pytest.fixture
def sample_covariance(sample_returns: pd.DataFrame) -> pd.DataFrame:
    return sample_returns.cov()


@pytest.fixture
def sample_weights(sample_returns: pd.DataFrame) -> pd.Series:
    weights = EqualWeightAllocator().optimize(sample_returns)
    return pd.Series(weights, index=sample_returns.columns, name="weight", dtype=float)


def test_portfolio_volatility_returns_positive_value(
    sample_weights: pd.Series,
    sample_covariance: pd.DataFrame,
) -> None:
    vol = portfolio_volatility(sample_weights, sample_covariance)

    assert vol > 0.0


def test_marginal_risk_contribution_returns_one_value_per_asset(
    sample_weights: pd.Series,
    sample_covariance: pd.DataFrame,
) -> None:
    mrc = marginal_risk_contribution(sample_weights, sample_covariance)

    assert isinstance(mrc, pd.Series)
    assert len(mrc) == len(sample_weights)
    assert mrc.index.tolist() == sample_covariance.columns.tolist()


def test_total_risk_contribution_sums_to_portfolio_volatility(
    sample_weights: pd.Series,
    sample_covariance: pd.DataFrame,
) -> None:
    trc = total_risk_contribution(sample_weights, sample_covariance)
    vol = portfolio_volatility(sample_weights, sample_covariance)

    assert np.isclose(float(trc.sum()), vol, atol=1e-8)


def test_percentage_risk_contribution_sums_to_one(
    sample_weights: pd.Series,
    sample_covariance: pd.DataFrame,
) -> None:
    prc = percentage_risk_contribution(sample_weights, sample_covariance)

    assert np.isclose(float(prc.sum()), 1.0, atol=1e-8)


def test_risk_contribution_table_returns_expected_columns(
    sample_weights: pd.Series,
    sample_covariance: pd.DataFrame,
) -> None:
    table = risk_contribution_table(sample_weights, sample_covariance)

    assert list(table.columns) == [
        "Asset",
        "Weight",
        "Marginal Risk Contribution",
        "Total Risk Contribution",
        "Percentage Risk Contribution",
    ]


def test_ndarray_weights_are_accepted_and_labeled(
    sample_weights: pd.Series,
    sample_covariance: pd.DataFrame,
) -> None:
    table = risk_contribution_table(sample_weights.values, sample_covariance)

    assert set(table["Asset"]) == set(sample_covariance.columns)
    assert np.isclose(float(table["Weight"].sum()), 1.0, atol=1e-8)


def test_invalid_negative_weights_raise_value_error(sample_covariance: pd.DataFrame) -> None:
    weights = pd.Series([0.5, 0.5, -0.1, 0.1], index=sample_covariance.columns)

    with pytest.raises(ValueError, match="non-negative"):
        portfolio_volatility(weights, sample_covariance)


def test_mismatched_labels_raise_value_error(sample_covariance: pd.DataFrame) -> None:
    weights = pd.Series([0.25, 0.25, 0.25, 0.25], index=["X", "Y", "Z", "W"])

    with pytest.raises(ValueError, match="match covariance matrix labels"):
        risk_contribution_table(weights, sample_covariance)


def test_compare_risk_contributions_returns_expected_columns(
    sample_returns: pd.DataFrame,
    sample_covariance: pd.DataFrame,
) -> None:
    hrp_weights = HRPAllocator().optimize(sample_returns)
    herc_weights = HERCAllocator().optimize(sample_returns)

    comparison = compare_risk_contributions(hrp_weights, herc_weights, sample_covariance)

    assert list(comparison.columns) == [
        "Asset",
        "HRP Weight",
        "HERC Weight",
        "HRP % Risk Contribution",
        "HERC % Risk Contribution",
        "Risk Contribution Difference",
    ]


@pytest.mark.parametrize(
    "allocator",
    [
        EqualWeightAllocator(),
        InverseVolatilityAllocator(),
        HRPAllocator(),
        HERCAllocator(),
    ],
)
def test_risk_contribution_table_works_for_supported_allocators(
    allocator,
    sample_returns: pd.DataFrame,
    sample_covariance: pd.DataFrame,
) -> None:
    weights = allocator.optimize(sample_returns)
    table = risk_contribution_table(weights, sample_covariance)

    assert not table.empty
    assert np.isclose(float(table["Percentage Risk Contribution"].sum()), 1.0, atol=1e-8)
