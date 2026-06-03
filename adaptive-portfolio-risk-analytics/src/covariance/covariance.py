from __future__ import annotations

import numpy as np
import pandas as pd


def compute_covariance_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the sample covariance matrix of asset returns.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Asset returns with shape (n_samples, n_assets).

    Returns
    -------
    pd.DataFrame
        Covariance matrix with shape (n_assets, n_assets).
    """
    return returns_df.cov()


def compute_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the correlation matrix of asset returns.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Asset returns with shape (n_samples, n_assets).

    Returns
    -------
    pd.DataFrame
        Correlation matrix with shape (n_assets, n_assets).
    """
    return returns_df.corr()


def validate_covariance_matrix(covariance_matrix: pd.DataFrame) -> bool:
    """
    Validate basic covariance matrix properties.

    Checks:
    - square matrix
    - symmetric matrix
    - strictly positive diagonal entries

    Parameters
    ----------
    covariance_matrix : pd.DataFrame

    Returns
    -------
    bool
    """
    if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
        return False

    values = covariance_matrix.values
    is_symmetric = np.allclose(values, values.T)
    positive_diagonal = (np.diag(values) > 0).all()
    return bool(is_symmetric and positive_diagonal)


def validate_correlation_matrix(correlation_matrix: pd.DataFrame) -> bool:
    """
    Validate correlation matrix properties.

    Checks:
    - square matrix
    - symmetric matrix
    - diagonal equals 1
    - values are bounded within [-1, 1]

    Parameters
    ----------
    correlation_matrix : pd.DataFrame

    Returns
    -------
    bool
    """
    if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
        return False

    values = correlation_matrix.values
    is_symmetric = np.allclose(values, values.T)

    # allow a small numerical tolerance for the diagonal and bounds checks
    atol = 1e-12
    diagonal_is_one = np.allclose(np.diag(values), 1.0, atol=atol)

    within_lower = (values >= (-1 - atol))
    within_upper = (values <= (1 + atol))
    within_bounds = (within_lower & within_upper).all()

    return bool(is_symmetric and diagonal_is_one and within_bounds)


def rank_correlations(correlation_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Rank all unique asset-pair correlations from highest to lowest.

    Parameters
    ----------
    correlation_matrix : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - Asset A
        - Asset B
        - Correlation
    """
    assets = list(correlation_matrix.columns)
    pairs: list[dict[str, object]] = []

    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            pairs.append(
                {
                    "Asset A": assets[i],
                    "Asset B": assets[j],
                    "Correlation": float(correlation_matrix.iloc[i, j]),
                }
            )

    return (
        pd.DataFrame(pairs)
        .sort_values(by="Correlation", ascending=False)
        .reset_index(drop=True)
    )


def compute_average_correlation(correlation_matrix: pd.DataFrame) -> float:
    """
    Compute the average of off-diagonal correlations.

    Parameters
    ----------
    correlation_matrix : pd.DataFrame

    Returns
    -------
    float
        Mean correlation of all unique asset pairs (upper triangular, k=1).
    """
    mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    mean_val = correlation_matrix.where(mask).stack().mean()
    return float(mean_val)
