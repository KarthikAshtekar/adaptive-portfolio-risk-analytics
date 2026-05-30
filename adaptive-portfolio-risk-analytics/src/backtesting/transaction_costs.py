"""Transaction cost utilities."""

from __future__ import annotations

import numpy as np


class TransactionCostCalculator:
    """Calculate simple transaction costs for rebalancing."""

    def __init__(self, bid_ask_spread: float = 0.001, commission: float = 0.0):
        self.bid_ask_spread = bid_ask_spread
        self.commission = commission

    def calculate_rebalancing_cost(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        portfolio_value: float,
    ) -> float:
        turnover = float(np.abs(target_weights - current_weights).sum()) / 2.0
        return turnover * portfolio_value * (self.bid_ask_spread + self.commission)
