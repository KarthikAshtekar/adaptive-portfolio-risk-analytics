"""Ex-ante timestamp and market-reaction leakage tests."""

from __future__ import annotations

import pandas as pd

from src.sentiment.ex_ante_filters import (
    apply_publication_lag,
    flag_reaction_data_leakage,
    validate_ex_ante_records,
)


def test_ex_ante_validation_rejects_missing_timestamps() -> None:
    records = pd.DataFrame(
        [{"title": "Policy outlook", "publication_time": None, "retrieval_time": None}]
    )

    validated = validate_ex_ante_records(records)

    assert not validated.loc[0, "is_ex_ante_valid"]
    assert "missing publication_time" in validated.loc[
        0, "ex_ante_validation_errors"
    ]


def test_publication_lag_is_applied_after_publication_date() -> None:
    records = pd.DataFrame(
        [
            {
                "title": "Forward outlook",
                "publication_time": "2024-01-05T15:00:00Z",
                "retrieval_time": "2024-01-05T16:00:00Z",
            }
        ]
    )

    lagged = apply_publication_lag(validate_ex_ante_records(records), lag_days=1)

    assert lagged.loc[0, "decision_available_date"] == pd.Timestamp(
        "2024-01-06T00:00:00Z"
    )


def test_possible_market_reaction_language_is_flagged_not_deleted() -> None:
    records = pd.DataFrame(
        [{"title": "Nifty dropped after policy announcement", "text": "Volatility spiked."}]
    )

    flagged = flag_reaction_data_leakage(records)

    assert len(flagged) == 1
    assert flagged.loc[0, "possible_reaction_data"]
    assert "Nifty dropped" in flagged.loc[0, "reaction_warning_reason"]
