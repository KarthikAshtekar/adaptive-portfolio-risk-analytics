"""Tests for the Phase 3C adaptive backtest wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.adaptive import get_policy_preset, run_regime_adaptive_backtest


def _returns(periods: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(20260618)
    index = pd.date_range("2022-01-03", periods=periods, freq="B")
    common = rng.normal(0.0003, 0.005, size=periods)
    return pd.DataFrame(
        {
            "A": common + rng.normal(0.0, 0.006, size=periods),
            "B": common + rng.normal(0.0, 0.008, size=periods),
            "C": common + rng.normal(0.0, 0.010, size=periods),
        },
        index=index,
    )


def _regimes(index: pd.DatetimeIndex) -> pd.Series:
    labels = np.select(
        [
            np.arange(len(index)) < 50,
            np.arange(len(index)) < 90,
            np.arange(len(index)) < 130,
        ],
        ["Calm", "Normal", "Stress"],
        default="Crisis",
    )
    return pd.Series(labels, index=index, name="observed_regime")


def test_adaptive_backtest_runs_and_returns_standard_outputs() -> None:
    returns = _returns()
    initial_value = 1_000_000.0
    result = run_regime_adaptive_backtest(
        returns,
        _regimes(returns.index),
        initial_value=initial_value,
        training_window=40,
        policy_map=get_policy_preset("Balanced default"),
    )

    assert {
        "portfolio_returns",
        "gross_portfolio_returns",
        "portfolio_values",
        "gross_portfolio_values",
        "weights",
        "diagnostics",
        "policy_table",
        "performance_metrics",
    }.issubset(result)
    assert not result["portfolio_returns"].empty
    assert not result["portfolio_values"].empty
    assert not result["weights"].empty
    assert np.isfinite(result["portfolio_returns"]).all()
    assert (result["portfolio_values"] > 0.0).all()
    assert initial_value * (1.0 + result["portfolio_returns"]).prod() == pytest.approx(
        result["portfolio_values"].iloc[-1]
    )
    assert initial_value * (1.0 + result["gross_portfolio_returns"]).prod() == pytest.approx(
        result["gross_portfolio_values"].iloc[-1]
    )


def test_adaptive_backtest_uses_lagged_regimes() -> None:
    returns = _returns()
    regimes = _regimes(returns.index)
    result = run_regime_adaptive_backtest(
        returns,
        regimes,
        training_window=40,
        use_lagged_regimes=True,
    )
    diagnostics = result["diagnostics"].set_index("date")
    transition_date = regimes.index[50]

    assert regimes.loc[transition_date] == "Normal"
    assert diagnostics.loc[transition_date, "regime"] == "Calm"


def test_adaptive_backtest_rejects_full_sample_hmm() -> None:
    returns = _returns()

    with pytest.raises(ValueError, match="historical-only"):
        run_regime_adaptive_backtest(
            returns,
            _regimes(returns.index),
            training_window=40,
            regime_method_name="HMM full-sample historical",
            use_lagged_regimes=False,
        )


def test_adaptive_backtest_accepts_walk_forward_decision_regimes() -> None:
    returns = _returns()
    decision_regimes = _regimes(returns.index).shift(1).fillna("Unknown")

    result = run_regime_adaptive_backtest(
        returns,
        decision_regimes,
        training_window=40,
        regime_method_name="HMM walk-forward decision regimes",
        use_lagged_regimes=False,
    )

    assert result["regime_method"] == "HMM walk-forward decision regimes"
    assert result["uses_lagged_regimes"] is False


def test_missing_defensive_returns_and_diagnostics_are_safe() -> None:
    returns = _returns()
    result = run_regime_adaptive_backtest(
        returns,
        _regimes(returns.index),
        defensive_returns=None,
        training_window=40,
    )

    expected = {
        "regime",
        "allocator",
        "covariance_method",
        "target_volatility",
        "risky_exposure",
        "defensive_weight",
        "turnover",
        "transaction_cost",
    }
    assert expected.issubset(result["diagnostics"].columns)
    assert result["diagnostics"]["defensive_weight"].between(0.0, 1.0).all()
    assert not result["diagnostics"].isna().all(axis=1).any()
