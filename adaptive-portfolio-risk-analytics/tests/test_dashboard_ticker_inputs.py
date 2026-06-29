"""Tests for dashboard ticker input helpers."""

from __future__ import annotations

import pandas as pd

import src.dashboard.app as dashboard_app
from src.dashboard.app import (
    asset_options_for_scope,
    merge_portfolio_tickers,
    parse_ticker_entries,
    selected_tickers_from_labels,
    ticker_label,
)


def test_parse_ticker_entries_normalizes_and_deduplicates_symbols() -> None:
    tickers = parse_ticker_entries(" aapl, RELIANCE.ns\nAAPL;btc-usd ")

    assert tickers == ["AAPL", "RELIANCE.NS", "BTC-USD"]


def test_merge_portfolio_tickers_uses_visible_selection_only() -> None:
    selected_labels = [ticker_label("HDFCBANK.NS"), ticker_label("TCS.NS")]

    tickers = merge_portfolio_tickers(selected_labels, ["AAPL", "TCS.NS"])

    assert tickers == ["HDFCBANK.NS", "TCS.NS"]


def test_custom_ticker_labels_are_selected_assets() -> None:
    tickers = selected_tickers_from_labels([ticker_label("AAPL"), ticker_label("TCS.NS")])

    assert tickers == ["AAPL", "TCS.NS"]


def test_asset_options_include_added_custom_tickers() -> None:
    options = asset_options_for_scope("Core Diversified", ["AAPL", "BTC-USD"])

    assert ticker_label("AAPL") in options
    assert ticker_label("BTC-USD") in options


def test_ticker_label_handles_custom_tickers() -> None:
    assert ticker_label("AAPL").endswith("Custom ticker")


def test_robustness_uses_current_sensitivity_objective(monkeypatch) -> None:
    captured = {}

    def fake_run_cpcv_validation(**kwargs):
        captured["objective"] = kwargs["objective"]
        return {
            "fold_results": pd.DataFrame(),
            "summary_table": pd.DataFrame(),
            "robustness_ranking": pd.DataFrame(),
            "split_diagnostics": pd.DataFrame(),
        }

    monkeypatch.setattr(
        dashboard_app,
        "run_cpcv_validation",
        fake_run_cpcv_validation,
    )
    sensitivity_payload = {
        "experiment_results_df": pd.DataFrame(
            {
                "strategy": ["HRP"],
                "sharpe": [1.5],
                "status": ["success"],
            }
        ),
        "objective_metric": "calmar",
        "returns_df": pd.DataFrame(
            {"A": [0.01]},
            index=pd.date_range("2024-01-01", periods=1),
        ),
        "train_window": 20,
        "initial_capital": 1_000_000.0,
    }

    result = dashboard_app.build_robustness_results(
        sensitivity_payload,
        n_blocks=4,
        n_test_blocks=1,
        embargo_pct=0.01,
        purge_window=0,
        max_configs=1,
        objective_metric="sharpe",
    )

    assert captured["objective"] == "sharpe"
    assert result["objective_metric"] == "sharpe"
    assert result["objective_label"] == "Net Sharpe"


def test_adaptive_robustness_uses_selected_objective_and_limit(monkeypatch) -> None:
    captured = {}

    def fake_run_cpcv_validation(**kwargs):
        captured.update(kwargs)
        return {
            "fold_results": pd.DataFrame(),
            "summary_table": pd.DataFrame(),
            "robustness_ranking": pd.DataFrame(),
            "split_diagnostics": pd.DataFrame(),
        }

    monkeypatch.setattr(
        dashboard_app,
        "run_cpcv_validation",
        fake_run_cpcv_validation,
    )
    sensitivity_payload = {
        "experiment_results_df": pd.DataFrame(
            [
                {
                    "strategy": "HRP",
                    "strategy_type": "fixed",
                    "sharpe": 1.0,
                    "status": "success",
                },
                {
                    "strategy": "Regime-Adaptive Rule-Based — Balanced",
                    "strategy_type": "regime_adaptive",
                    "regime_source": "rule_based_lagged",
                    "policy_preset": "balanced",
                    "sharpe": 1.5,
                    "status": "success",
                },
            ]
        ),
        "objective_metric": "calmar",
        "returns_df": pd.DataFrame(
            {"A": [0.01]},
            index=pd.date_range("2024-01-01", periods=1),
        ),
        "train_window": 20,
        "initial_capital": 1_000_000.0,
    }

    result = dashboard_app.build_robustness_results(
        sensitivity_payload,
        n_blocks=4,
        n_test_blocks=1,
        embargo_pct=0.01,
        purge_window=0,
        max_configs=1,
        objective_metric="sharpe",
        include_adaptive_in_cpcv=True,
        max_adaptive_configs=1,
    )

    assert captured["objective"] == "sharpe"
    assert captured["max_adaptive_configs"] == 1
    assert captured["experiment_configs"]["strategy_type"].eq(
        "regime_adaptive"
    ).any()
    assert result["include_adaptive_in_cpcv"] is True


def test_market_regime_results_preserve_current_dashboard_objective() -> None:
    rng = pd.Series(
        range(220),
        index=pd.date_range("2023-01-02", periods=220, freq="B"),
        dtype=float,
    )
    benchmark_returns = pd.Series(
        0.0002 + (rng % 11 - 5) * 0.0005,
        index=rng.index,
        name="benchmark",
    )
    returns_df = pd.DataFrame(
        {
            "A": benchmark_returns + 0.0001,
            "B": benchmark_returns - 0.0001,
        }
    )
    strategy_results = {
        "Equal Weight": {"portfolio_returns": benchmark_returns},
        "HRP": {"portfolio_returns": benchmark_returns + 0.0002},
    }

    result = dashboard_app.build_market_regime_results(
        returns_df=returns_df,
        strategy_results=strategy_results,
        benchmark_strategy="Equal Weight",
        lookback_vol=21,
        lookback_trend=63,
        lookback_corr=21,
        crisis_drawdown=-0.15,
        stress_drawdown=-0.08,
        use_lagged_decision_regime=True,
        objective_metric="sharpe",
    )

    assert result["objective_metric"] == "sharpe"
    assert result["use_lagged_decision_regime"] is True
    assert result["decision_regimes"].iloc[0] == "Unknown"


def test_hmm_unavailable_falls_back_without_breaking_dashboard(monkeypatch) -> None:
    monkeypatch.setattr(dashboard_app, "HMM_AVAILABLE", False)
    index = pd.date_range("2023-01-02", periods=140, freq="B")
    benchmark_returns = pd.Series(0.0005, index=index, name="benchmark")
    returns_df = pd.DataFrame(
        {
            "A": benchmark_returns + 0.0001,
            "B": benchmark_returns - 0.0001,
        }
    )
    strategy_results = {
        "Equal Weight": {"portfolio_returns": benchmark_returns},
    }

    result = dashboard_app.build_market_regime_results(
        returns_df=returns_df,
        strategy_results=strategy_results,
        benchmark_strategy="Equal Weight",
        lookback_vol=21,
        lookback_trend=63,
        lookback_corr=21,
        crisis_drawdown=-0.15,
        stress_drawdown=-0.08,
        use_lagged_decision_regime=True,
        objective_metric="calmar",
        regime_method="HMM walk-forward experimental",
    )

    assert result["hmm_available"] is False
    assert result["hmm_error"] == (
        "HMM regime detection requires the optional dependency `hmmlearn`."
    )
    assert not result["rule_based_regimes"].empty
    assert result["observed_regimes"].equals(result["rule_based_regimes"])
