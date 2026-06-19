"""Tests for Phase 3D adaptive CPCV integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.validation.robustness as robustness
from src.experiments import FULL_SAMPLE_HMM_ERROR
from src.validation import run_cpcv_validation


def _returns() -> pd.DataFrame:
    rng = np.random.default_rng(20260619)
    index = pd.date_range("2021-01-04", periods=120, freq="B")
    return pd.DataFrame(
        rng.normal(0.0003, 0.01, size=(len(index), 3)),
        index=index,
        columns=["A", "B", "C"],
    )


def _config(strategy: str, strategy_type: str, source=None) -> dict[str, object]:
    return {
        "experiment_name": "adaptive_cpcv_unit",
        "strategy": strategy,
        "strategy_type": strategy_type,
        "regime_source": source,
        "policy_preset": "balanced" if source else None,
        "covariance_method": "sample",
        "rebalance_mode": "calendar",
        "threshold": 0.05,
        "transaction_cost_bps": 10.0,
        "slippage_bps": 5.0,
        "vol_targeting_enabled": False,
        "target_vol": None,
        "defensive_asset": "Synthetic Risk-Free",
        "train_window": 20,
        "training_window": 20,
        "initial_capital": 1_000_000.0,
    }


@pytest.fixture
def stub_test_blocks(monkeypatch):
    def fake_run_test_block(
        returns,
        split,
        test_index,
        config_row,
        defensive_returns=None,
    ):
        _ = (returns, split, defensive_returns)
        level = 0.001 if config_row.get("strategy_type") == "regime_adaptive" else 0.0005
        values = np.where(
            np.arange(len(test_index)) % 2 == 0,
            level,
            -0.5 * level,
        )
        return pd.Series(values, index=test_index)

    monkeypatch.setattr(robustness, "_run_test_block", fake_run_test_block)


def test_selected_objective_propagates_to_adaptive_cpcv(stub_test_blocks) -> None:
    result = run_cpcv_validation(
        _returns(),
        [
            _config("HRP", "fixed"),
            _config(
                "Regime-Adaptive Rule-Based — Balanced",
                "regime_adaptive",
                "rule_based_lagged",
            ),
        ],
        n_blocks=3,
        n_test_blocks=1,
        embargo_pct=0.0,
        objective="sharpe",
        max_configs=1,
        max_adaptive_configs=1,
    )

    assert result["robustness_ranking"]["metric"].eq("sharpe").all()
    assert result["robustness_ranking"]["strategy_type"].eq("regime_adaptive").any()


def test_calmar_fallback_applies_only_when_objective_missing(
    stub_test_blocks,
) -> None:
    result = run_cpcv_validation(
        _returns(),
        [_config("HRP", "fixed")],
        n_blocks=3,
        n_test_blocks=1,
        embargo_pct=0.0,
        objective=None,
    )

    assert result["robustness_ranking"]["metric"].eq("calmar").all()


def test_adaptive_config_limit_is_enforced(stub_test_blocks) -> None:
    configs = [
        _config("HRP", "fixed"),
        _config("Adaptive 1", "regime_adaptive", "rule_based_lagged"),
        _config("Adaptive 2", "regime_adaptive", "rule_based_lagged"),
        _config("Adaptive 3", "regime_adaptive", "rule_based_lagged"),
    ]

    result = run_cpcv_validation(
        _returns(),
        configs,
        n_blocks=3,
        n_test_blocks=1,
        embargo_pct=0.0,
        objective="calmar",
        max_configs=1,
        max_adaptive_configs=2,
    )
    adaptive_configs = (
        result["fold_results"]
        .loc[
            result["fold_results"]["strategy_type"].eq("regime_adaptive"),
            "config_id",
        ]
        .nunique()
    )

    assert adaptive_configs == 2


def test_full_sample_hmm_is_rejected_for_adaptive_cpcv() -> None:
    config = _config(
        "Regime-Adaptive HMM Full Sample",
        "regime_adaptive",
        "hmm_full_sample",
    )

    with pytest.raises(ValueError, match="historical-only"):
        run_cpcv_validation(
            _returns(),
            [config],
            n_blocks=3,
            n_test_blocks=1,
            objective="calmar",
        )

    assert "historical-only" in FULL_SAMPLE_HMM_ERROR
