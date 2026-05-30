"""Phase 3 CPCV extension point."""

from __future__ import annotations

import pandas as pd


class CPCVBacktester:
    """Reserved for Phase 3 (not implemented)."""

    def split(self, X: pd.DataFrame, y: pd.Series | None = None):
        _ = (X, y)
        raise NotImplementedError("CPCV is reserved for Phase 3 and is not implemented.")
