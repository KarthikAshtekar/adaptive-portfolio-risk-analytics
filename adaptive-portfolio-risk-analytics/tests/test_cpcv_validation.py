"""Tests for CPCV-style time-series split generation."""

from __future__ import annotations

from math import comb

import pandas as pd

from src.validation import (
    apply_purge_and_embargo,
    generate_cpcv_splits,
    generate_time_blocks,
)


def test_time_blocks_cover_every_observation_exactly_once() -> None:
    index = pd.DatetimeIndex(
        [
            "2024-01-04",
            "2024-01-02",
            "2024-01-02",
            "2024-01-05",
            "2024-01-03",
            "2024-01-01",
        ]
    )

    blocks = generate_time_blocks(index, n_blocks=4)
    positions = [position for block in blocks for position in block["positions"]]

    assert sorted(positions) == list(range(len(index)))
    assert len(positions) == len(set(positions))


def test_time_blocks_are_chronological_and_near_equal() -> None:
    index = pd.date_range("2024-01-01", periods=11, freq="D")[::-1]

    blocks = generate_time_blocks(index, n_blocks=3)
    sizes = [len(block["dates"]) for block in blocks]

    assert max(sizes) - min(sizes) <= 1
    assert all(
        blocks[position]["end_date"] <= blocks[position + 1]["start_date"]
        for position in range(len(blocks) - 1)
    )


def test_cpcv_split_count_matches_block_combinations() -> None:
    index = pd.date_range("2023-01-01", periods=60, freq="D")

    splits = generate_cpcv_splits(
        index,
        n_blocks=6,
        n_test_blocks=2,
        embargo_pct=0.0,
    )

    assert len(splits) == comb(6, 2)


def test_test_indices_never_overlap_training_indices() -> None:
    index = pd.date_range("2023-01-01", periods=60, freq="D")

    splits = generate_cpcv_splits(index, n_blocks=6, n_test_blocks=2)

    for split in splits:
        assert set(split["train_index"]).isdisjoint(set(split["test_index"]))


def test_embargo_removes_observations_after_test_interval() -> None:
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    test_index = index[3:5]

    allowed = apply_purge_and_embargo(
        index,
        test_index,
        purge_window=0,
        embargo_pct=0.20,
    )

    assert index[5] not in allowed
    assert index[6] not in allowed
    assert index[7] in allowed


def test_purge_window_removes_observations_around_test_interval() -> None:
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    test_index = index[3:5]

    allowed = apply_purge_and_embargo(
        index,
        test_index,
        purge_window=1,
        embargo_pct=0.0,
    )

    assert index[2] not in allowed
    assert index[5] not in allowed
    assert index[1] in allowed
    assert index[6] in allowed


def test_empty_and_small_inputs_are_handled_safely() -> None:
    assert generate_time_blocks(pd.DatetimeIndex([]), n_blocks=6) == []
    assert generate_cpcv_splits(pd.DatetimeIndex([])) == []

    small_index = pd.date_range("2024-01-01", periods=2, freq="D")
    blocks = generate_time_blocks(small_index, n_blocks=6)
    splits = generate_cpcv_splits(
        small_index,
        n_blocks=6,
        n_test_blocks=2,
        embargo_pct=0.0,
    )

    assert len(blocks) == 2
    assert len(splits) == 1
