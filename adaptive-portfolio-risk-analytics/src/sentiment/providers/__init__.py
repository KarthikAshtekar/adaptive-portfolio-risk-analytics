"""Provider interfaces for Phase 4A.5 ex-ante sentiment ingestion."""

from .alpha_vantage_provider import AlphaVantageNewsProvider
from .base import (
    NORMALIZED_SENTIMENT_COLUMNS,
    ProviderValidation,
    SentimentProvider,
    normalized_frame,
)
from .earnings_provider import EARNINGS_MANIFEST_COLUMNS, EarningsCallProvider
from .gdelt_provider import DEFAULT_GDELT_QUERIES, GDELTProvider
from .local_provider import LocalProvider
from .news_provider import NewsProvider
from .rbi_provider import RBIProvider

__all__ = [
    "AlphaVantageNewsProvider",
    "DEFAULT_GDELT_QUERIES",
    "EARNINGS_MANIFEST_COLUMNS",
    "EarningsCallProvider",
    "GDELTProvider",
    "LocalProvider",
    "NORMALIZED_SENTIMENT_COLUMNS",
    "NewsProvider",
    "ProviderValidation",
    "RBIProvider",
    "SentimentProvider",
    "normalized_frame",
]
