"""Factory for benchmark and strategy allocator construction."""

from __future__ import annotations

from src.optimization import (
    EqualWeightAllocator,
    HERCAllocator,
    HRPAllocator,
    InverseVolatilityAllocator,
)


STRATEGY_ALIASES = {
    "equal_weight": "Equal Weight",
    "equal weight": "Equal Weight",
    "inverse_volatility": "Inverse Volatility",
    "inverse volatility": "Inverse Volatility",
    "hrp": "HRP",
    "herc": "HERC",
}


class BenchmarkFactory:
    """Create allocator instances for supported benchmark strategies."""

    @staticmethod
    def normalize_strategy_name(strategy_name: str) -> str:
        """Map canonical names and aliases to display names."""
        if not isinstance(strategy_name, str):
            raise TypeError("strategy_name must be a string")

        normalized = strategy_name.strip().lower()
        if normalized not in STRATEGY_ALIASES:
            supported = ", ".join(sorted(STRATEGY_ALIASES))
            raise ValueError(
                f"unsupported strategy '{strategy_name}'. Supported names: {supported}"
            )
        return STRATEGY_ALIASES[normalized]

    @staticmethod
    def get_allocator(
        strategy_name: str,
        covariance_method: str = "sample",
        covariance_kwargs: dict | None = None,
    ):
        """Return the allocator for the requested strategy or benchmark."""
        display_name = BenchmarkFactory.normalize_strategy_name(strategy_name)
        covariance_kwargs = dict(covariance_kwargs or {})

        if display_name == "Equal Weight":
            return EqualWeightAllocator()
        if display_name == "Inverse Volatility":
            return InverseVolatilityAllocator()
        if display_name == "HRP":
            return HRPAllocator(
                covariance_method=covariance_method,
                covariance_kwargs=covariance_kwargs,
            )
        if display_name == "HERC":
            return HERCAllocator(
                covariance_method=covariance_method,
                covariance_kwargs=covariance_kwargs,
            )

        raise ValueError(f"unsupported strategy '{strategy_name}'")
