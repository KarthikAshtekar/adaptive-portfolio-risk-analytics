"""Cross-path tests for Phase 3E defensive-return consistency."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.adaptive import run_regime_adaptive_backtest
from src.experiments.adaptive import execute_adaptive_experiment
from src.validation import run_cpcv_validation


def _returns(periods: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(20260620)
    index = pd.date_range("2020-01-02", periods=periods, freq="B")
    common = rng.normal(0.0002, 0.006, periods)
    return pd.DataFrame(
        {
            "A": common + rng.normal(0.0, 0.004, periods),
            "B": common + rng.normal(0.0, 0.006, periods),
            "C": common + rng.normal(0.0, 0.008, periods),
        },
        index=index,
    )


def _regimes(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series("Crisis", index=index, name="observed_regime")


def _config(rate: float = 0.04) -> dict[str, object]:
    return {
        "strategy": "Regime-Adaptive Rule-Based — Conservative",
        "strategy_type": "regime_adaptive",
        "regime_source": "rule_based_lagged",
        "policy_preset": "conservative",
        "training_window": 40,
        "train_window": 40,
        "defensive_asset": "Synthetic Risk-Free",
        "defensive_source": "synthetic",
        "defensive_annual_rate": rate,
        "defensive_ticker": None,
        "defensive_fallback": "synthetic",
        "transaction_cost_bps": 10.0,
        "slippage_bps": 5.0,
        "rebalance_frequency": "M",
        "initial_capital": 1_000_000.0,
    }


def test_direct_and_experiment_paths_use_identical_synthetic_returns() -> None:
    returns = _returns()
    regime_input = {
        "regimes": _regimes(returns.index),
        "use_lagged_regimes": True,
        "regime_method_name": "Rule-based observed regimes, lagged internally",
        "features": pd.DataFrame(index=returns.index),
    }
    direct = run_regime_adaptive_backtest(
        returns,
        regime_input["regimes"],
        training_window=40,
        defensive_source="synthetic",
        defensive_annual_rate=0.04,
    )
    experiment = execute_adaptive_experiment(
        returns,
        _config(),
        regime_input=regime_input,
    )["backtest"]

    pd.testing.assert_series_equal(
        direct["defensive_returns"],
        experiment["defensive_returns"],
    )
    assert direct["defensive_metadata"] == experiment["defensive_metadata"]


def test_default_direct_path_is_not_silent_zero_cash() -> None:
    returns = _returns()
    result = run_regime_adaptive_backtest(
        returns,
        _regimes(returns.index),
        training_window=40,
    )

    assert result["defensive_returns"].gt(0.0).all()
    assert result["defensive_metadata"]["defensive_source_used"] == "synthetic"
    assert result["performance_metrics"]["defensive_annual_rate"] == 0.04


def test_changing_synthetic_rate_changes_adaptive_final_value() -> None:
    returns = _returns()
    low = run_regime_adaptive_backtest(
        returns,
        _regimes(returns.index),
        training_window=40,
        defensive_source="synthetic",
        defensive_annual_rate=0.00,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )
    high = run_regime_adaptive_backtest(
        returns,
        _regimes(returns.index),
        training_window=40,
        defensive_source="synthetic",
        defensive_annual_rate=0.08,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )

    assert high["portfolio_values"].iloc[-1] > low["portfolio_values"].iloc[-1]


def test_cpcv_adaptive_folds_record_same_defensive_convention() -> None:
    returns = _returns(360)
    config = _config()

    result = run_cpcv_validation(
        returns,
        [config],
        n_blocks=3,
        n_test_blocks=1,
        embargo_pct=0.0,
        objective="calmar",
        max_adaptive_configs=1,
    )
    successful = result["fold_results"].loc[
        result["fold_results"]["status"].eq("success")
    ]

    assert not successful.empty
    assert successful["defensive_source_used"].eq("synthetic").all()
    assert successful["defensive_annual_rate"].eq(0.04).all()
    assert successful["defensive_fallback_used"].eq(False).all()
