"""Optional FinBERT fallback tests."""

from __future__ import annotations

import pandas as pd

from src.sentiment.finbert_scoring import score_with_finbert


def test_finbert_falls_back_when_model_is_unavailable() -> None:
    records = pd.DataFrame(
        [{"title": "Inflation shock risk", "text": "Banking stress may increase."}]
    )

    scored = score_with_finbert(
        records,
        pipeline_factory=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("model unavailable")
        ),
    )

    assert scored.loc[0, "scoring_method_used"] == "lexicon_fallback"
    assert scored.loc[0, "fallback_used"]
    assert "model unavailable" in scored.loc[0, "fallback_reason"]
    assert scored.loc[0, "model_name"] == "phase4a_lexicon"
