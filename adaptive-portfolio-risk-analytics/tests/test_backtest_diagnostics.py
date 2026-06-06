"""Tests for backtest diagnostics helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtesting import build_rebalance_summary, compare_cost_drag


def test_rebalance_summary_works() -> None:
    rebalance_log = pd.DataFrame(
        {
            "rebalance_reason": ["calendar", "threshold"],
            "turnover": [0.10, 0.20],
            "transaction_cost": [100.0, 150.0],
            "max_weight_drift": [0.04, 0.07],
        }
    )

    summary = build_rebalance_summary(rebalance_log)

    assert summary["total_rebalances"] == 2
    assert np.isclose(summary["total_turnover"], 0.30, atol=1e-8)
    assert summary["calendar_rebalances"] == 1
    assert summary["threshold_rebalances"] == 1
    assert summary["rebalance_reason_counts"] == {"calendar": 1, "threshold": 1}
    assert np.isclose(summary["average_turnover_by_reason"]["calendar"], 0.10)


def test_transaction_cost_summary_works() -> None:
    gross_values = pd.Series([1_000_000.0, 1_020_000.0, 1_040_000.0])
    net_values = pd.Series([1_000_000.0, 1_019_000.0, 1_037_500.0])

    summary = compare_cost_drag(gross_values, net_values)

    assert summary["gross_final_value"] == 1_040_000.0
    assert summary["net_final_value"] == 1_037_500.0
    assert summary["cost_drag"] == 2_500.0


def test_empty_logs_handled_gracefully() -> None:
    summary = build_rebalance_summary(pd.DataFrame())

    assert summary["total_rebalances"] == 0
    assert summary["total_transaction_cost"] == 0.0
    assert summary["rebalance_reason_counts"] == {}
