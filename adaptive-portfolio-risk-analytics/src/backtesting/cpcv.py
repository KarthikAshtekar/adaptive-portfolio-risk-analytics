"""Legacy compatibility extension point for Phase 3A CPCV-style validation."""

from __future__ import annotations

import pandas as pd


class CPCVBacktester:
    """Direct callers should use the implemented :mod:`src.validation` package."""

    def split(self, X: pd.DataFrame, y: pd.Series | None = None):
        _ = (X, y)
        raise NotImplementedError(
            "Use src.validation.generate_cpcv_splits or "
            "src.validation.run_cpcv_validation for Phase 3A validation."
        )
