"""Core types and constants for the platform."""

from enum import Enum
from typing import Dict, Any
from dataclasses import dataclass
import numpy as np


class AllocationMethod(Enum):
    """Enumeration of available portfolio allocation methods."""

    EQUAL_WEIGHT = "equal_weight"
    MEAN_VARIANCE = "mean_variance"
    HRP = "hrp"
    HERC = "herc"
    INVERSE_VOLATILITY = "inverse_volatility"
    DYNAMIC = "dynamic"


class RegimeType(Enum):
    """Market regime classification."""

    BULL = "bull"
    BEAR = "bear"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    NEUTRAL = "neutral"


class RiskMetric(Enum):
    """Risk metrics available for calculation."""

    VAR = "var"
    CVAR = "cvar"
    VOLATILITY = "volatility"
    SHARPE = "sharpe"
    SORTINO = "sortino"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR = "calmar"


@dataclass
class PortfolioConfig:
    """Configuration for portfolio construction."""

    rebalance_frequency: str = "M"  # D, W, M, Q, Y
    lookback_window: int = 252
    min_weight: float = 0.0
    max_weight: float = 1.0
    target_volatility: Optional[float] = None
    transaction_cost: float = 0.001  # 10bps
    tax_rate: float = 0.0


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    start_date: str = "2015-01-01"
    end_date: str = "2024-01-01"
    initial_capital: float = 1_000_000.0
    rebalance_frequency: str = "M"
    cpcv_n_splits: int = 5
    embargo_pct: float = 0.01
    test_size_pct: float = 0.15


# Constants
DEFAULT_RISK_FREE_RATE = 0.02  # 2% annual
TRADING_DAYS_PER_YEAR = 252
BUSINESS_DAYS_PER_YEAR = 252

# Default covariance estimation parameters
LEDOIT_WOLF_SHRINKAGE = 0.1
GERBER_CORRELATION_TYPE = "RS"  # Rank-Sign

# Machine learning parameters
HIERARCHICAL_LINKAGE = "ward"
HIERARCHICAL_DISTANCE = "euclidean"

# Optional type for type hints
from typing import Optional
