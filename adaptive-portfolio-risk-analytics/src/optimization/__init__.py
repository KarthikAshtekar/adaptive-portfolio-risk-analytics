"""Portfolio allocation exports."""

from .base import BaseAllocator
from .equal_weight import EqualWeightAllocator
from .mean_variance import MeanVarianceAllocator
from .inverse_volatility import InverseVolatilityAllocator
from .hrp_allocator import HRPAllocator
from .dynamic_allocation import DynamicAllocationAllocator

# Backward-compatible aliases.
PortfolioOptimizer = BaseAllocator
EqualWeightOptimizer = EqualWeightAllocator
MeanVarianceOptimizer = MeanVarianceAllocator
InverseVolatilityOptimizer = InverseVolatilityAllocator
DynamicAllocationOptimizer = DynamicAllocationAllocator

__all__ = [
    "BaseAllocator",
    "EqualWeightAllocator",
    "MeanVarianceAllocator",
    "InverseVolatilityAllocator",
    "HRPAllocator",
    "DynamicAllocationAllocator",
    "PortfolioOptimizer",
    "EqualWeightOptimizer",
    "MeanVarianceOptimizer",
    "InverseVolatilityOptimizer",
    "DynamicAllocationOptimizer",
]
