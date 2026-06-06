"""Transaction cost utilities."""

from __future__ import annotations

import numpy as np

from .turnover import calculate_turnover


class TransactionCostModel:
    """Estimate transaction costs from turnover and portfolio value."""

    def __init__(
        self,
        base_bps: float = 10.0,
        slippage_bps: float = 5.0,
        volatility_multiplier: float = 0.0,
    ):
        self.base_bps = float(base_bps)
        self.slippage_bps = float(slippage_bps)
        self.volatility_multiplier = float(volatility_multiplier)

    def estimate_cost(
        self,
        turnover: float,
        portfolio_value: float,
        portfolio_volatility: float | None = None,
    ) -> float:
        """Estimate transaction cost from turnover and cost-rate assumptions."""
        cost_rate = (self.base_bps + self.slippage_bps) / 10000.0
        if portfolio_volatility is not None:
            cost_rate += self.volatility_multiplier * float(portfolio_volatility)

        return float(turnover) * float(portfolio_value) * cost_rate

class TransactionCostCalculator:
    """Backward-compatible transaction cost adapter for rebalancing."""

    def __init__(self, bid_ask_spread: float = 0.001, commission: float = 0.0):
        self.bid_ask_spread = bid_ask_spread
        self.commission = commission
        total_bps = (float(bid_ask_spread) + float(commission)) * 10000.0
        self.model = TransactionCostModel(
            base_bps=total_bps,
            slippage_bps=0.0,
            volatility_multiplier=0.0,
        )

    def calculate_rebalancing_cost(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        portfolio_value: float,
        portfolio_volatility: float | None = None,
    ) -> float:
        turnover = calculate_turnover(current_weights, target_weights)
        return self.model.estimate_cost(
            turnover=turnover,
            portfolio_value=portfolio_value,
            portfolio_volatility=portfolio_volatility,
        )
