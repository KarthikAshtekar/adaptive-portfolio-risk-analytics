"""Phase 3A time-series-safe building blocks for CPCV-style validation."""

from __future__ import annotations

from itertools import combinations
from math import ceil
from typing import Iterable

import numpy as np
import pandas as pd


def _normalize_datetime_index(
    index: Iterable[object] | pd.Index,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Return a stable chronological index and its original integer positions."""
    if index is None:
        return pd.DatetimeIndex([]), np.array([], dtype=int)

    raw_index = pd.Index(index)
    if raw_index.empty:
        return pd.DatetimeIndex([]), np.array([], dtype=int)

    try:
        datetime_index = pd.DatetimeIndex(pd.to_datetime(raw_index, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise ValueError("index must contain datetime-like values") from exc

    if datetime_index.isna().any():
        raise ValueError("index must not contain missing datetime values")

    order = np.argsort(datetime_index.asi8, kind="mergesort")
    return datetime_index.take(order), order.astype(int)


def generate_time_blocks(index, n_blocks: int) -> list[dict[str, object]]:
    """Split an index into ordered, non-overlapping, near-equal time blocks.

    Duplicate timestamps are retained. ``positions`` refer to the observations'
    positions in the original input, while ``dates`` are returned in chronological
    order. When there are fewer observations than requested blocks, only non-empty
    blocks are created.
    """
    if int(n_blocks) <= 0:
        raise ValueError("n_blocks must be positive")

    sorted_index, original_positions = _normalize_datetime_index(index)
    if sorted_index.empty:
        return []

    block_count = min(int(n_blocks), len(sorted_index))
    block_positions = np.array_split(np.arange(len(sorted_index)), block_count)

    blocks: list[dict[str, object]] = []
    for block_id, sorted_positions in enumerate(block_positions):
        dates = sorted_index.take(sorted_positions)
        positions = original_positions[sorted_positions]
        blocks.append(
            {
                "block_id": block_id,
                "start_date": dates[0],
                "end_date": dates[-1],
                "positions": positions.tolist(),
                "dates": dates,
            }
        )
    return blocks


def _contiguous_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return []

    boundaries = np.flatnonzero(np.diff(selected) > 1)
    starts = np.concatenate(([selected[0]], selected[boundaries + 1]))
    ends = np.concatenate((selected[boundaries], [selected[-1]]))
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def apply_purge_and_embargo(
    index,
    test_index,
    purge_window: int = 0,
    embargo_pct: float = 0.01,
) -> pd.DatetimeIndex:
    """Return observations permitted for training after purge and embargo.

    Purging removes ``purge_window`` observations on both sides of every
    contiguous test interval. Embargo removes ``ceil(len(index) * embargo_pct)``
    observations after each interval. Test observations are always excluded.
    """
    if int(purge_window) < 0:
        raise ValueError("purge_window must be non-negative")
    if not 0.0 <= float(embargo_pct) <= 1.0:
        raise ValueError("embargo_pct must be between 0 and 1")

    sorted_index, _ = _normalize_datetime_index(index)
    if sorted_index.empty:
        return sorted_index

    sorted_test_index, _ = _normalize_datetime_index(test_index)
    if sorted_test_index.empty:
        return sorted_index

    test_mask = np.asarray(sorted_index.isin(sorted_test_index), dtype=bool)
    allowed_mask = ~test_mask
    embargo_size = ceil(len(sorted_index) * float(embargo_pct))
    purge_size = int(purge_window)

    for interval_start, interval_end in _contiguous_intervals(test_mask):
        purge_start = max(0, interval_start - purge_size)
        purge_end = min(len(sorted_index) - 1, interval_end + purge_size)
        allowed_mask[purge_start : purge_end + 1] = False

        embargo_start = interval_end + 1
        embargo_end = min(len(sorted_index), embargo_start + embargo_size)
        allowed_mask[embargo_start:embargo_end] = False

    return sorted_index[allowed_mask]


def generate_cpcv_splits(
    index,
    n_blocks: int = 6,
    n_test_blocks: int = 2,
    embargo_pct: float = 0.01,
    purge_window: int = 0,
) -> list[dict[str, object]]:
    """Generate deterministic CPCV-style train/test combinations.

    This is a pragmatic research splitter rather than a complete institutional
    CPCV path-construction implementation. Every combination of test blocks is
    generated, and the remaining observations are filtered through purge and
    embargo rules.
    """
    if int(n_test_blocks) <= 0:
        raise ValueError("n_test_blocks must be positive")

    blocks = generate_time_blocks(index, n_blocks)
    if not blocks:
        return []
    if int(n_test_blocks) > len(blocks):
        return []

    sorted_index, _ = _normalize_datetime_index(index)
    splits: list[dict[str, object]] = []

    for split_id, test_block_ids in enumerate(combinations(range(len(blocks)), int(n_test_blocks))):
        test_parts = [blocks[block_id]["dates"] for block_id in test_block_ids]
        test_index = test_parts[0]
        for part in test_parts[1:]:
            test_index = test_index.append(part)
        test_index = test_index.sort_values()

        train_index = apply_purge_and_embargo(
            sorted_index,
            test_index,
            purge_window=purge_window,
            embargo_pct=embargo_pct,
        )
        splits.append(
            {
                "split_id": split_id,
                "train_index": train_index,
                "test_index": test_index,
                "test_block_ids": list(test_block_ids),
                "train_start": train_index[0] if len(train_index) else None,
                "train_end": train_index[-1] if len(train_index) else None,
                "test_start": test_index[0] if len(test_index) else None,
                "test_end": test_index[-1] if len(test_index) else None,
                "n_train": len(train_index),
                "n_test": len(test_index),
            }
        )

    return splits
