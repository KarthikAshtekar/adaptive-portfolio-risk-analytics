"""Stress-testing utilities (kept as extension support)."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


class StressTestingFramework:
    """Simple stress scenarios for portfolio diagnostics."""

    @staticmethod
    def historical_scenario(
        portfolio_weights: np.ndarray,
        asset_returns: pd.DataFrame,
        scenario_date: pd.Timestamp,
    ) -> float:
        scenario_returns = asset_returns.loc[scenario_date]
        return float(np.dot(portfolio_weights, scenario_returns.values))

    @staticmethod
    def reverse_stress_test(
        portfolio_weights: np.ndarray,
        target_loss: float,
        asset_correlations: np.ndarray,
    ) -> Dict:
        n_assets = len(portfolio_weights)
        if n_assets == 0:
            return {"required_uniform_move": 0.0, "asset_moves": np.array([])}

        avg_corr = float(np.nanmean(asset_correlations))
        scaling = 1.0 + max(0.0, avg_corr)
        required_move = target_loss / (np.sum(np.abs(portfolio_weights)) * scaling)
        asset_moves = np.full(n_assets, required_move)
        return {
            "required_uniform_move": required_move,
            "asset_moves": asset_moves,
            "assumed_average_correlation": avg_corr,
        }

    @staticmethod
    def correlation_stress_test(
        portfolio_weights: np.ndarray,
        volatilities: np.ndarray,
        correlation_change: float = 0.2,
    ) -> float:
        n_assets = len(portfolio_weights)
        stressed_corr = np.full((n_assets, n_assets), correlation_change)
        np.fill_diagonal(stressed_corr, 1.0)
        stressed_cov = np.outer(volatilities, volatilities) * stressed_corr
        stressed_var = float(portfolio_weights.T @ stressed_cov @ portfolio_weights)
        return float(np.sqrt(max(stressed_var, 0.0)))
