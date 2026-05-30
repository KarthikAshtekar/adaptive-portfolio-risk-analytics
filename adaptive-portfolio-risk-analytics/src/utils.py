"""Utility functions and helpers for the platform."""

import numpy as np
import pandas as pd
from typing import Union
from pathlib import Path
import json
from datetime import datetime


def ensure_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert price series to returns.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data (will calculate log returns if prices, return if already returns)

    Returns
    -------
    pd.DataFrame
        Log returns
    """
    if (prices > 1).sum().sum() > 0:  # Likely prices, not returns
        return np.log(prices).diff().dropna()
    return prices.dropna()


def calculate_rolling_covariance(
    returns: pd.DataFrame, window: int = 252
) -> dict:
    """
    Calculate rolling covariance matrix.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns
    window : int
        Rolling window size in trading days

    Returns
    -------
    dict
        Dictionary mapping dates to covariance matrices
    """
    rolling_cov = {}
    for i in range(window, len(returns)):
        subset = returns.iloc[i - window:i]
        rolling_cov[returns.index[i]] = subset.cov()
    return rolling_cov


def validate_weights(weights: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Validate portfolio weights (sum to 1, non-negative).

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    tolerance : float
        Sum tolerance

    Returns
    -------
    bool
        True if valid weights
    """
    if np.any(weights < -tolerance):
        return False
    if not np.isclose(np.sum(weights), 1.0, atol=tolerance):
        return False
    return True


def save_json(data: dict, filepath: Union[str, Path]) -> None:
    """
    Save dictionary to JSON file.

    Parameters
    ----------
    data : dict
        Data to save
    filepath : str or Path
        Output filepath
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Union[str, Path]) -> dict:
    """
    Load JSON file to dictionary.

    Parameters
    ----------
    filepath : str or Path
        Input filepath

    Returns
    -------
    dict
        Loaded data
    """
    with open(filepath, "r") as f:
        return json.load(f)


def get_timestamp() -> str:
    """Return current timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_business_date(date: pd.Timestamp, frequency: str = "B") -> pd.Timestamp:
    """Get business day equivalent of date."""
    bday_index = pd.bdate_range(end=date, periods=1, freq=frequency)
    return bday_index[0]
