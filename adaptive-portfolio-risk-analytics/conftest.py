"""pytest configuration and fixtures."""

import pytest
import numpy as np
import pandas as pd

# Tests should run via package/module resolution from project root (no sys.path hacks).


@pytest.fixture
def sample_returns():
    """Generate sample return data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=252, freq="B")
    data = np.random.randn(252, 5) * 0.02
    return pd.DataFrame(
        data,
        index=dates,
        columns=["Asset1", "Asset2", "Asset3", "Asset4", "Asset5"],
    )


@pytest.fixture
def sample_cov_matrix():
    """Generate sample covariance matrix."""
    np.random.seed(42)
    n_assets = 5
    cov = np.random.randn(n_assets, n_assets)
    cov = (cov + cov.T) / 2  # Make symmetric
    cov = cov + np.eye(n_assets) * 2  # Ensure positive definite
    return cov


@pytest.fixture
def sample_weights():
    """Generate sample portfolio weights."""
    n_assets = 5
    weights = np.random.dirichlet(np.ones(n_assets))
    return weights


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow")
