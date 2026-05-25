"""
Data preprocessing module for cleaning and normalizing market data.

Handles: missing values, outliers, price adjustments, returns calculation
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats


class DataPreprocessor:
    """Preprocess market data for analysis."""

    @staticmethod
    def handle_missing_values(
        data: pd.DataFrame, method: str = "forward_fill"
    ) -> pd.DataFrame:
        """
        Handle missing values in market data.

        Parameters
        ----------
        data : pd.DataFrame
            Input data with potential missing values
        method : str
            Method: 'forward_fill', 'interpolate', 'drop'

        Returns
        -------
        pd.DataFrame
            Data with missing values handled

        TODO: Implement multiple imputation strategies
        TODO: Add validation for maximum allowed gaps
        """
        if method == "forward_fill":
            return data.fillna(method="ffill").fillna(method="bfill")
        elif method == "interpolate":
            return data.interpolate(method="linear")
        elif method == "drop":
            return data.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def detect_outliers(
        data: pd.DataFrame, method: str = "iqr", threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detect and flag outliers using IQR or z-score.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        method : str
            Method: 'iqr' or 'zscore'
        threshold : float
            Threshold for outlier detection

        Returns
        -------
        pd.DataFrame
            Boolean mask of outliers

        TODO: Implement robust outlier detection
        TODO: Add DBSCAN and isolation forest methods
        """
        if method == "iqr":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            return (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
        elif method == "zscore":
            return np.abs(stats.zscore(data)) > threshold
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def calculate_returns(
        prices: pd.DataFrame, method: str = "log"
    ) -> pd.DataFrame:
        """
        Calculate returns from price series.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data
        method : str
            Method: 'log' or 'simple'

        Returns
        -------
        pd.DataFrame
            Returns

        TODO: Add multi-period returns calculation
        """
        if method == "log":
            return np.log(prices).diff().dropna()
        elif method == "simple":
            return prices.pct_change().dropna()
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize prices (remove dividends, splits effects).

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (typically adjusted close)

        Returns
        -------
        pd.DataFrame
            Normalized prices

        TODO: Implement dividend and split adjustment
        TODO: Handle corporate actions
        """
        return prices.copy()


class DataValidator:
    """Validate data quality and completeness."""

    @staticmethod
    def check_completeness(data: pd.DataFrame, min_coverage: float = 0.95) -> bool:
        """
        Check data completeness.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        min_coverage : float
            Minimum required coverage (0-1)

        Returns
        -------
        bool
            True if data meets coverage threshold

        TODO: Add per-column validation
        TODO: Add time period validation
        """
        coverage = 1 - (data.isna().sum().sum() / data.size)
        return coverage >= min_coverage

    @staticmethod
    def check_stationarity(
        data: pd.Series, test: str = "adf"
    ) -> Tuple[float, bool]:
        """
        Test for stationarity (ADF test).

        Parameters
        ----------
        data : pd.Series
            Time series data
        test : str
            Test method: 'adf'

        Returns
        -------
        Tuple[float, bool]
            p-value and stationarity indicator (True = stationary)

        TODO: Add KPSS test
        TODO: Add Phillips-Perron test
        """
        from statsmodels.tsa.stattools import adfuller

        result = adfuller(data.dropna())
        p_value = result[1]
        is_stationary = p_value < 0.05
        return p_value, is_stationary
