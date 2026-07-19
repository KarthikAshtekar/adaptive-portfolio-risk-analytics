"""Portfolio allocation exports."""

from .base import BaseAllocator
from .equal_weight import EqualWeightAllocator
from .mean_variance import MeanVarianceAllocator
from .inverse_volatility import InverseVolatilityAllocator
from .hrp_allocator import (
    HRPAllocator,
    allocate_hrp_weights,
    get_quasi_diagonal_order,
    compute_cluster_variance,
    recursive_bisection,
)
from .dynamic_allocation import DynamicAllocationAllocator

# Backward-compatible aliases.
PortfolioOptimizer = BaseAllocator
EqualWeightOptimizer = EqualWeightAllocator
MeanVarianceOptimizer = MeanVarianceAllocator
InverseVolatilityOptimizer = InverseVolatilityAllocator
HRPOptimizer = HRPAllocator
DynamicAllocationOptimizer = DynamicAllocationAllocator


def __getattr__(name: str):
    """Load clustering-backed HERC exports without creating an import cycle."""

    if name in {"HERCAllocator", "HERCOptimizer"}:
        from src.clustering.herc_allocator import HERCAllocator

        return HERCAllocator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseAllocator",
    "EqualWeightAllocator",
    "MeanVarianceAllocator",
    "InverseVolatilityAllocator",
    "HRPAllocator",
    "HERCAllocator",
    "DynamicAllocationAllocator",
    "allocate_hrp_weights",
    "get_quasi_diagonal_order",
    "compute_cluster_variance",
    "recursive_bisection",
    "PortfolioOptimizer",
    "EqualWeightOptimizer",
    "MeanVarianceOptimizer",
    "InverseVolatilityOptimizer",
    "DynamicAllocationOptimizer",
    "HRPOptimizer",
    "HERCOptimizer",
]
