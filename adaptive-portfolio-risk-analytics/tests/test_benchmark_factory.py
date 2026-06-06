"""Tests for the benchmark factory."""

from __future__ import annotations

import pytest

from src.benchmarks import BenchmarkFactory
from src.optimization import (
    EqualWeightAllocator,
    HERCAllocator,
    HRPAllocator,
    InverseVolatilityAllocator,
)


def test_factory_returns_equal_weight_allocator() -> None:
    allocator = BenchmarkFactory.get_allocator("equal_weight")

    assert isinstance(allocator, EqualWeightAllocator)


def test_factory_returns_inverse_volatility_allocator() -> None:
    allocator = BenchmarkFactory.get_allocator("inverse_volatility")

    assert isinstance(allocator, InverseVolatilityAllocator)


def test_factory_returns_hrp_allocator() -> None:
    allocator = BenchmarkFactory.get_allocator("hrp", covariance_method="ledoit_wolf")

    assert isinstance(allocator, HRPAllocator)
    assert allocator.covariance_method == "ledoit_wolf"


def test_factory_returns_herc_allocator() -> None:
    allocator = BenchmarkFactory.get_allocator("herc", covariance_method="ewma")

    assert isinstance(allocator, HERCAllocator)
    assert allocator.covariance_method == "ewma"


def test_invalid_strategy_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unsupported strategy"):
        BenchmarkFactory.get_allocator("markowitz")


@pytest.mark.parametrize(
    ("strategy_name", "expected_type"),
    [
        ("Equal Weight", EqualWeightAllocator),
        ("Inverse Volatility", InverseVolatilityAllocator),
        ("HRP", HRPAllocator),
        ("HERC", HERCAllocator),
    ],
)
def test_aliases_work_correctly(strategy_name: str, expected_type) -> None:
    allocator = BenchmarkFactory.get_allocator(strategy_name)

    assert isinstance(allocator, expected_type)
