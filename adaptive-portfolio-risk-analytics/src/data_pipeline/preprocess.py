"""Data preprocessing for prices and returns."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats


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
    def normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
        if prices.empty:
            raise ValueError("prices must not be empty")
        base = prices.iloc[0]
        return prices / base


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
