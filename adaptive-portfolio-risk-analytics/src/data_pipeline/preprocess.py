"""Data preprocessing for returns and risk analytics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class ReturnsRiskOutputs:
    """Container for Stage 2 return and volatility outputs."""

    simple_returns_df: pd.DataFrame
    returns_df: pd.DataFrame
    log_returns_df: pd.DataFrame
    volatility_summary_df: pd.DataFrame
    rolling_volatility_df: pd.DataFrame
    return_comparison_df: pd.DataFrame


class DataPreprocessor:
    """Preprocess market price data for portfolio analytics."""

    @staticmethod
    def handle_missing_values(data: pd.DataFrame, method: str = "forward_fill") -> pd.DataFrame:
        if data.empty:
            raise ValueError("data must not be empty")

        if method == "forward_fill":
            return data.ffill().bfill()
        if method == "interpolate":
            return data.interpolate(method="linear").ffill().bfill()
        if method == "drop":
            return data.dropna()
        raise ValueError(f"unknown missing value method: {method}")

    @staticmethod
    def detect_outliers(
        data: pd.DataFrame,
        method: str = "iqr",
        threshold: float = 3.0,
    ) -> pd.DataFrame:
        if data.empty:
            raise ValueError("data must not be empty")

        if method == "iqr":
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            return (data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))
        if method == "zscore":
            z = np.abs(stats.zscore(data, nan_policy="omit"))
            return pd.DataFrame(z > threshold, index=data.index, columns=data.columns)
        raise ValueError(f"unknown outlier detection method: {method}")

    @staticmethod
    def calculate_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
        if prices.empty:
            raise ValueError("prices must not be empty")

        clean = prices.replace([np.inf, -np.inf], np.nan).dropna(how="any")
        if clean.empty:
            raise ValueError("prices has no valid rows after cleanup")

        if method == "log":
            returns = np.log(clean / clean.shift(1))
        elif method == "simple":
            returns = clean.pct_change()
        else:
            raise ValueError(f"unknown returns method: {method}")

        returns = returns.dropna(how="any")
        if returns.empty:
            raise ValueError("returns series is empty after calculation")
        return returns

    @staticmethod
    def compare_return_methods(
        simple_returns: pd.DataFrame,
        log_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        DataPreprocessor._validate_returns_frame(simple_returns, "simple_returns")
        DataPreprocessor._validate_returns_frame(log_returns, "log_returns")

        aligned_simple, aligned_log = simple_returns.align(log_returns, join="inner")
        if aligned_simple.empty:
            raise ValueError("simple_returns and log_returns have no overlapping observations")

        comparison = pd.DataFrame(index=aligned_simple.columns)
        comparison.index.name = "asset"
        comparison["mean_simple_return"] = aligned_simple.mean()
        comparison["mean_log_return"] = aligned_log.mean()
        comparison["mean_abs_difference"] = (aligned_simple - aligned_log).abs().mean()
        comparison["max_abs_difference"] = (aligned_simple - aligned_log).abs().max()
        comparison["return_correlation"] = aligned_simple.corrwith(aligned_log)
        return comparison

    @staticmethod
    def calculate_volatility(
        returns: pd.DataFrame,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> pd.DataFrame:
        DataPreprocessor._validate_returns_frame(returns, "returns")
        _validate_periods_per_year(periods_per_year)

        daily_volatility = returns.std(ddof=1)
        annualized_volatility = daily_volatility * np.sqrt(periods_per_year)

        volatility_summary = pd.DataFrame(index=returns.columns)
        volatility_summary.index.name = "asset"
        volatility_summary["daily_volatility"] = daily_volatility
        volatility_summary["annualized_volatility"] = annualized_volatility
        volatility_summary["annualization_factor"] = periods_per_year
        return volatility_summary

    @staticmethod
    def calculate_rolling_volatility(
        returns: pd.DataFrame,
        windows: tuple[int, ...] = (30, 90),
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> pd.DataFrame:
        DataPreprocessor._validate_returns_frame(returns, "returns")
        _validate_periods_per_year(periods_per_year)

        invalid_windows = [window for window in windows if window <= 1]
        if invalid_windows:
            raise ValueError(f"rolling windows must be greater than 1: {invalid_windows}")

        rolling_frames: list[pd.DataFrame] = []
        rolling_scale = np.sqrt(periods_per_year)

        for window in windows:
            rolling_volatility = returns.rolling(window=window).std(ddof=1) * rolling_scale
            rolling_volatility.columns = pd.MultiIndex.from_product(
                [[f"{window}d"], rolling_volatility.columns],
                names=["window", "asset"],
            )
            rolling_frames.append(rolling_volatility)

        return pd.concat(rolling_frames, axis=1).sort_index(axis=1)

    @staticmethod
    def build_returns_risk_outputs(
        prices: pd.DataFrame,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
        rolling_windows: tuple[int, ...] = (30, 90),
    ) -> ReturnsRiskOutputs:
        simple_returns_df = DataPreprocessor.calculate_returns(prices, method="simple")
        log_returns_df = DataPreprocessor.calculate_returns(prices, method="log")
        volatility_summary_df = DataPreprocessor.calculate_volatility(
            log_returns_df,
            periods_per_year=periods_per_year,
        )
        rolling_volatility_df = DataPreprocessor.calculate_rolling_volatility(
            log_returns_df,
            windows=rolling_windows,
            periods_per_year=periods_per_year,
        )
        return_comparison_df = DataPreprocessor.compare_return_methods(
            simple_returns_df,
            log_returns_df,
        )

        return ReturnsRiskOutputs(
            simple_returns_df=simple_returns_df,
            returns_df=log_returns_df.copy(),
            log_returns_df=log_returns_df,
            volatility_summary_df=volatility_summary_df,
            rolling_volatility_df=rolling_volatility_df,
            return_comparison_df=return_comparison_df,
        )

    @staticmethod
    def normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
        if prices.empty:
            raise ValueError("prices must not be empty")
        base = prices.iloc[0]
        return prices / base

    @staticmethod
    def _validate_returns_frame(data: pd.DataFrame, name: str) -> None:
        if data.empty:
            raise ValueError(f"{name} must not be empty")
        if data.isna().any().any():
            raise ValueError(f"{name} must not contain missing values")


class DataValidator:
    """Validate data quality and statistical properties."""

    @staticmethod
    def check_completeness(data: pd.DataFrame, min_coverage: float = 0.95) -> bool:
        if data.empty:
            return False
        coverage = 1.0 - (data.isna().sum().sum() / data.size)
        return bool(coverage >= min_coverage)

    @staticmethod
    def check_stationarity(data: pd.Series, test: str = "adf") -> Tuple[float, bool]:
        if test != "adf":
            raise ValueError("only adf test is supported in Phase 1")

        from statsmodels.tsa.stattools import adfuller

        result = adfuller(data.dropna())
        p_value = float(result[1])
        return p_value, p_value < 0.05


def _validate_periods_per_year(periods_per_year: int) -> None:
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
