"""Lag-safe regime-adaptive portfolio backtesting for Phase 3C."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics import PerformanceAnalytics, RiskAnalytics
from src.backtesting.rebalance_rules import (
    should_rebalance_calendar,
    should_rebalance_threshold,
)
from src.backtesting.transaction_costs import TransactionCostModel
from src.backtesting.turnover import calculate_turnover, summarize_turnover
from src.benchmarks import BenchmarkFactory
from src.covariance import CovarianceFactory

from .controller import RegimeAdaptiveController
from .policies import (
    RegimePolicy,
    policy_map_to_dataframe,
    validate_policy_map,
)


def _normalize_weights(weights, columns: pd.Index) -> pd.Series:
    if isinstance(weights, pd.Series):
        series = weights.reindex(columns).astype(float)
    else:
        series = pd.Series(np.asarray(weights, dtype=float), index=columns)
    series = series.clip(lower=0.0)
    total = float(series.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("allocator returned invalid weights")
    return series / total


def _align_defensive_returns(
    defensive_returns,
    index: pd.DatetimeIndex,
) -> pd.Series:
    if defensive_returns is None:
        return pd.Series(
            0.0,
            index=index,
            name="defensive_return",
            dtype=float,
        )
    if isinstance(defensive_returns, pd.DataFrame):
        if defensive_returns.shape[1] != 1:
            raise ValueError("defensive_returns must be a Series or single-column DataFrame")
        series = defensive_returns.iloc[:, 0].copy()
    elif isinstance(defensive_returns, pd.Series):
        series = defensive_returns.copy()
    else:
        raise TypeError("defensive_returns must be a pandas Series or single-column DataFrame")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("defensive_returns index must be a DatetimeIndex")
    return (
        pd.to_numeric(series, errors="coerce")
        .sort_index()
        .reindex(index)
        .ffill()
        .fillna(0.0)
        .rename("defensive_return")
    )


def _post_return_weights(
    weights: pd.Series,
    asset_returns: pd.Series,
    portfolio_return: float,
) -> pd.Series:
    denominator = 1.0 + float(portfolio_return)
    if denominator <= 0.0:
        return weights.copy()
    drifted = weights * (1.0 + asset_returns) / denominator
    drifted = drifted.clip(lower=0.0)
    total = float(drifted.sum())
    return drifted / total if total > 0.0 else weights.copy()


def _should_rebalance(
    *,
    policy: RegimePolicy,
    current_date: pd.Timestamp,
    previous_rebalance_date: pd.Timestamp | None,
    current_weights: pd.Series,
    target_weights: pd.Series,
    rebalance_frequency: str,
) -> tuple[bool, str | None, float]:
    calendar_flag = should_rebalance_calendar(
        current_date=current_date,
        previous_rebalance_date=previous_rebalance_date,
        frequency=rebalance_frequency,
    )
    threshold_flag = should_rebalance_threshold(
        current_weights,
        target_weights,
        threshold=policy.rebalance_threshold,
    )
    max_drift = float((current_weights - target_weights).abs().max())

    if policy.rebalance_mode == "calendar":
        return calendar_flag, "calendar" if calendar_flag else None, max_drift
    if policy.rebalance_mode == "threshold":
        return threshold_flag, "threshold" if threshold_flag else None, max_drift
    if policy.rebalance_mode == "calendar_or_threshold":
        if calendar_flag and threshold_flag:
            return True, "calendar_or_threshold", max_drift
        if calendar_flag:
            return True, "calendar", max_drift
        if threshold_flag:
            return True, "threshold", max_drift
        return False, None, max_drift
    raise ValueError(f"unsupported rebalance mode '{policy.rebalance_mode}'")


def run_regime_adaptive_backtest(
    returns,
    regimes,
    prices=None,
    defensive_returns=None,
    initial_value: float = 1.0,
    training_window: int = 252,
    rebalance_frequency: str = "M",
    transaction_cost_bps: float = 10,
    slippage_bps: float = 5,
    policy_map=None,
    regime_method_name: str = "rule_based",
    use_lagged_regimes: bool = True,
    periods_per_year: int = 252,
) -> dict[str, object]:
    """Run a first-version expanding decision loop for the adaptive strategy.

    Allocator and covariance targets refresh on calendar boundaries or regime
    changes. Exposure is recalculated daily from rolling realized volatility.
    Weights selected at date ``t`` are applied to returns at ``t+1``.
    """
    _ = prices
    method_name = str(regime_method_name).strip().lower()
    if "hmm" in method_name and ("full" in method_name or "historical" in method_name):
        raise ValueError(
            "Full-sample HMM regimes are historical-only and cannot drive " "the adaptive backtest."
        )
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if returns.empty:
        raise ValueError("returns must not be empty")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns index must be a DatetimeIndex")
    if not isinstance(regimes, pd.Series):
        raise TypeError("regimes must be a pandas Series")
    if not isinstance(regimes.index, pd.DatetimeIndex):
        raise ValueError("regimes index must be a DatetimeIndex")
    if int(training_window) < 20:
        raise ValueError("training_window must be at least 20")
    if float(initial_value) <= 0.0:
        raise ValueError("initial_value must be positive")
    if int(periods_per_year) <= 0:
        raise ValueError("periods_per_year must be positive")

    clean = returns.apply(pd.to_numeric, errors="coerce").dropna(how="any").sort_index()
    clean = clean[~clean.index.duplicated(keep="last")]
    if len(clean) <= int(training_window):
        raise ValueError("not enough return observations for training_window")

    policies = validate_policy_map(policy_map) if policy_map is not None else None
    controller = RegimeAdaptiveController(
        policy_map=policies,
        use_lagged_regimes=use_lagged_regimes,
    )
    regime_series = regimes.sort_index()
    defensive = _align_defensive_returns(defensive_returns, clean.index)
    full_columns = list(clean.columns) + ["Defensive"]
    cost_model = TransactionCostModel(
        base_bps=float(transaction_cost_bps),
        slippage_bps=float(slippage_bps),
    )

    current_weights = pd.Series(
        [1.0 / clean.shape[1]] * clean.shape[1] + [0.0],
        index=full_columns,
        dtype=float,
    )
    risky_sleeve_weights = pd.Series(
        1.0 / clean.shape[1],
        index=clean.columns,
        dtype=float,
    )
    target_weights = current_weights.copy()
    previous_rebalance_date: pd.Timestamp | None = None
    previous_target_date: pd.Timestamp | None = None
    previous_regime: str | None = None

    net_value = float(initial_value)
    gross_value = float(initial_value)
    value_dates = [clean.index[int(training_window) - 1]]
    net_values = [net_value]
    gross_values = [gross_value]
    portfolio_dates: list[pd.Timestamp] = []
    net_returns: list[float] = []
    gross_returns: list[float] = []
    applied_regimes: list[str] = []
    weights_records: list[pd.Series] = []
    diagnostics: list[dict[str, object]] = []
    turnover_values: list[float] = []
    turnover_dates: list[pd.Timestamp] = []

    for position in range(int(training_window) - 1, len(clean) - 1):
        current_date = clean.index[position]
        next_date = clean.index[position + 1]
        train_slice = clean.iloc[position - int(training_window) + 1 : position + 1]
        policy = controller.get_policy(current_date, regime_series)
        selected_regime = controller._regime_for_date(current_date, regime_series)

        refresh_target = (
            previous_target_date is None
            or selected_regime != previous_regime
            or should_rebalance_calendar(
                current_date=current_date,
                previous_rebalance_date=previous_target_date,
                frequency=rebalance_frequency,
            )
        )
        if refresh_target:
            covariance = CovarianceFactory.compute(
                train_slice,
                method=policy.covariance_method,
            )
            allocator = BenchmarkFactory.get_allocator(
                policy.allocator,
                covariance_method=policy.covariance_method,
            )
            risky_sleeve_weights = _normalize_weights(
                allocator.optimize(train_slice, covariance),
                train_slice.columns,
            )
            previous_target_date = current_date
            previous_regime = selected_regime

        risky_history = train_slice.mul(risky_sleeve_weights, axis=1).sum(axis=1)
        realized_volatility = float(risky_history.std(ddof=1) * np.sqrt(int(periods_per_year)))
        risky_exposure, defensive_weight = controller.calculate_risky_exposure(
            realized_volatility,
            policy.target_volatility,
            policy.risky_exposure_cap,
            floor=policy.defensive_weight_floor,
        )
        target_weights = pd.concat(
            [
                risky_sleeve_weights.mul(risky_exposure),
                pd.Series({"Defensive": defensive_weight}),
            ]
        ).reindex(full_columns)

        should_rebalance, rebalance_reason, max_drift = _should_rebalance(
            policy=policy,
            current_date=current_date,
            previous_rebalance_date=previous_rebalance_date,
            current_weights=current_weights,
            target_weights=target_weights,
            rebalance_frequency=rebalance_frequency,
        )
        if previous_rebalance_date is None:
            should_rebalance = True
            rebalance_reason = "initial"

        turnover = 0.0
        transaction_cost = 0.0
        value_before_cost = net_value
        if should_rebalance:
            turnover = calculate_turnover(current_weights, target_weights)
            transaction_cost = cost_model.estimate_cost(
                turnover=turnover,
                portfolio_value=value_before_cost,
                portfolio_volatility=realized_volatility,
            )
            net_value = max(0.0, net_value - transaction_cost)
            current_weights = target_weights.copy()
            previous_rebalance_date = current_date
            weights_records.append(current_weights.rename(current_date))
            turnover_values.append(turnover)
            turnover_dates.append(current_date)

        next_returns = pd.concat(
            [
                clean.loc[next_date],
                pd.Series({"Defensive": defensive.loc[next_date]}),
            ]
        ).reindex(full_columns)
        portfolio_return = float(current_weights.dot(next_returns))
        gross_value *= 1.0 + portfolio_return
        net_value_after_return = net_value * (1.0 + portfolio_return)
        net_period_return = (
            net_value_after_return / value_before_cost - 1.0 if value_before_cost > 0.0 else 0.0
        )
        net_value = net_value_after_return
        current_weights = _post_return_weights(
            current_weights,
            next_returns,
            portfolio_return,
        )

        portfolio_dates.append(next_date)
        gross_returns.append(portfolio_return)
        net_returns.append(net_period_return)
        applied_regimes.append(selected_regime)
        value_dates.append(next_date)
        gross_values.append(gross_value)
        net_values.append(net_value)
        diagnostics.append(
            {
                "date": current_date,
                "regime": selected_regime,
                "allocator": policy.allocator,
                "covariance_method": policy.covariance_method,
                "target_volatility": float(policy.target_volatility),
                "rebalance_mode": policy.rebalance_mode,
                "rebalance_threshold": float(policy.rebalance_threshold),
                "realized_volatility": realized_volatility,
                "risky_exposure": risky_exposure,
                "defensive_weight": defensive_weight,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "rebalanced": bool(should_rebalance),
                "rebalance_reason": rebalance_reason,
                "max_weight_drift": max_drift,
                "policy_notes": policy.notes,
            }
        )

    portfolio_returns = pd.Series(
        net_returns,
        index=portfolio_dates,
        name="Regime-Adaptive",
        dtype=float,
    )
    gross_portfolio_returns = pd.Series(
        gross_returns,
        index=portfolio_dates,
        name="gross_portfolio_return",
        dtype=float,
    )
    applied_regime_series = pd.Series(
        applied_regimes,
        index=portfolio_dates,
        name="decision_regime",
        dtype="object",
    )
    portfolio_values = pd.Series(
        net_values,
        index=value_dates,
        name="portfolio_value",
        dtype=float,
    )
    gross_portfolio_values = pd.Series(
        gross_values,
        index=value_dates,
        name="gross_portfolio_value",
        dtype=float,
    )
    drawdown = portfolio_values / portfolio_values.cummax() - 1.0
    weights = (
        pd.DataFrame(weights_records)
        if weights_records
        else pd.DataFrame(columns=full_columns, dtype=float)
    )
    diagnostics_frame = pd.DataFrame(diagnostics)
    turnover_series = pd.Series(
        turnover_values,
        index=turnover_dates,
        name="turnover",
        dtype=float,
    )
    turnover_summary = summarize_turnover(turnover_series)
    metrics = PerformanceAnalytics.summary_table(portfolio_returns)
    metrics.update(
        {
            "final_value": float(portfolio_values.iloc[-1]),
            "total_turnover": float(turnover_summary["total_turnover"]),
            "average_turnover": float(turnover_summary["average_turnover"]),
            "total_transaction_cost": float(diagnostics_frame["transaction_cost"].sum()),
            "number_of_rebalances": int(diagnostics_frame["rebalanced"].sum()),
        }
    )

    return {
        "portfolio_returns": portfolio_returns,
        "gross_portfolio_returns": gross_portfolio_returns,
        "portfolio_values": portfolio_values,
        "gross_portfolio_values": gross_portfolio_values,
        "drawdown": drawdown,
        "weights": weights,
        "weights_history": weights,
        "latest_weights": weights.iloc[-1] if not weights.empty else pd.Series(dtype=float),
        "diagnostics": diagnostics_frame,
        "policy_table": policy_map_to_dataframe(policies if policies is not None else None),
        "turnover_series": turnover_series,
        "turnover_summary": turnover_summary,
        "performance_metrics": metrics,
        "regime_method": regime_method_name,
        "uses_lagged_regimes": bool(use_lagged_regimes),
        "applied_regimes": applied_regime_series,
    }
