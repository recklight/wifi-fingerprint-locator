"""Split strategies: coverage, leakage, reproducibility and block boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from wifi_fingerprint_locator.errors import ConfigError
from wifi_fingerprint_locator.splits import (
    SPLIT_STRATEGIES,
    fold_indices,
    group_codes,
    holdout_indices,
)

GROUPS = np.repeat([f"p{k:02d}" for k in range(12)], 10)


@pytest.mark.parametrize("strategy", SPLIT_STRATEGIES)
def test_every_item_is_tested_exactly_once(strategy):
    folds = fold_indices(strategy, GROUPS, n_splits=3, rng=0)
    tested = np.concatenate([test for _, test in folds])
    assert np.array_equal(np.sort(tested), np.arange(GROUPS.size))
    for train, test in folds:
        assert np.intersect1d(train, test).size == 0
        assert train.size + test.size == GROUPS.size


@pytest.mark.parametrize("strategy", ["location_holdout", "location_block"])
def test_location_splits_never_share_a_location(strategy):
    for train, test in fold_indices(strategy, GROUPS, n_splits=4, rng=3):
        assert not set(GROUPS[train]) & set(GROUPS[test])


def test_packet_and_temporal_splits_test_known_locations():
    for strategy in ("packet_random", "temporal"):
        for train, test in fold_indices(strategy, GROUPS, n_splits=3, rng=1):
            assert set(GROUPS[test]) <= set(GROUPS[train])


def test_temporal_split_tests_contiguous_blocks_of_every_location():
    folds = fold_indices("temporal", GROUPS, n_splits=2)
    _, first_test = folds[0]
    assert set(GROUPS[first_test]) == set(GROUPS)
    assert first_test[:5].tolist() == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("strategy", SPLIT_STRATEGIES)
def test_the_same_seed_gives_the_same_folds(strategy):
    first = fold_indices(strategy, GROUPS, n_splits=3, rng=7)
    second = fold_indices(strategy, GROUPS, n_splits=3, rng=7)
    for (a_train, a_test), (b_train, b_test) in zip(first, second, strict=True):
        assert np.array_equal(a_train, b_train) and np.array_equal(a_test, b_test)


def test_shuffled_strategies_depend_on_the_seed():
    for strategy in ("packet_random", "location_holdout"):
        a = fold_indices(strategy, GROUPS, n_splits=3, rng=1)[0][1]
        b = fold_indices(strategy, GROUPS, n_splits=3, rng=2)[0][1]
        assert not np.array_equal(a, b), strategy


def test_location_block_follows_recording_order():
    folds = fold_indices("location_block", GROUPS, n_splits=3)
    assert folds[0][1].tolist() == list(range(40))
    assert folds[2][1].tolist() == list(range(80, 120))


def test_location_block_holdout_reproduces_contiguous_index_ranges():
    groups = np.repeat(np.arange(56), 200)
    split = holdout_indices("location_block", groups, fractions=(5 / 7, 1 / 7, 1 / 7))
    assert np.array_equal(split.train, np.arange(0, 8000))
    assert np.array_equal(split.validation, np.arange(8000, 9600))
    assert np.array_equal(split.test, np.arange(9600, 11200))


def test_packet_random_holdout_has_the_requested_sizes():
    groups = np.repeat(np.arange(56), 200)
    split = holdout_indices("packet_random", groups, rng=0)
    assert split.sizes() == {"train": 7840, "validation": 1680, "test": 1680}


def test_temporal_holdout_trains_on_early_packets():
    split = holdout_indices("temporal", GROUPS, fractions=(0.6, 0.2, 0.2))
    assert split.train[:6].tolist() == [0, 1, 2, 3, 4, 5]
    assert split.validation[:2].tolist() == [6, 7] and split.test[:2].tolist() == [8, 9]


def test_single_location_holdout_split_keeps_locations_whole():
    split = holdout_indices("location_holdout", GROUPS, rng=4)
    parts = [set(GROUPS[split.train]), set(GROUPS[split.validation]), set(GROUPS[split.test])]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])


def test_group_codes_follow_first_appearance():
    assert group_codes(["b", "a", "b", "c"]).tolist() == [0, 1, 0, 2]


@pytest.mark.parametrize(
    ("strategy", "n_splits", "fragment"),
    [
        ("location_holdout", 13, "reference locations"),
        ("temporal", 11, "smallest location"),
        ("packet_random", 1, "at least 2"),
        ("spiral", 3, "split_strategy"),
    ],
)
def test_impossible_fold_counts_are_reported(strategy, n_splits, fragment):
    with pytest.raises(ConfigError, match=fragment):
        fold_indices(strategy, GROUPS, n_splits=n_splits)


def test_invalid_fractions_are_rejected():
    with pytest.raises(ConfigError, match="adding up to 1"):
        holdout_indices("packet_random", GROUPS, fractions=(0.5, 0.5, 0.5))
    with pytest.raises(ConfigError, match="no training or test"):
        holdout_indices("location_block", GROUPS, fractions=(1.0, 0.0, 0.0))
