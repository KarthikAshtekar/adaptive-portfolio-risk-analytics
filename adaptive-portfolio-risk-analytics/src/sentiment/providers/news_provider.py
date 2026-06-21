"""Local news provider alias for reviewed article exports."""

from __future__ import annotations

from .local_provider import LocalProvider


class NewsProvider(LocalProvider):
    """Load reviewed normalized news records from a local CSV/DataFrame."""

    provider_name = "local_news"

    def __init__(self, source) -> None:
        super().__init__(source, provider_name=self.provider_name)
