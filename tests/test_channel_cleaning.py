"""Cleaning policies: non-finite packets, stream-count mixtures and zero entries."""

from __future__ import annotations

import numpy as np
import pytest

from wifi_fingerprint_locator.channel_cleaning import (
    CleaningReport,
    clean_channels,
    repair_zero_entries,
)
from wifi_fingerprint_locator.errors import DatasetError

POSITIONS = np.arange(30.0)


def smooth_stack(n_packets=6, n_rx=3):
    ramp = (1.0 + 0.1 * POSITIONS) * np.exp(0.05j * POSITIONS)
    return np.tile(ramp, (n_packets, n_rx, 1)).astype(complex)


def test_zero_entries_are_interpolated_from_the_same_antenna_only():
    stack = smooth_stack()
    stack[:, 1] *= 10.0  # antenna 1 on a very different scale
    expected = stack.copy()
    stack[2, 0, [0, 7, 29]] = 0
    repaired, unrecoverable, count = repair_zero_entries(stack, POSITIONS)
    assert count == 3 and not unrecoverable.any()
    assert np.allclose(repaired[2, 0, 7], expected[2, 0, 7], rtol=1e-2)
    # Edges repeat the nearest non-zero value instead of extrapolating.
    assert repaired[2, 0, 0] == expected[2, 0, 1]
    assert repaired[2, 0, 29] == expected[2, 0, 28]
    assert np.array_equal(repaired[2, 1], expected[2, 1])


def test_an_antenna_with_almost_no_values_cannot_be_repaired():
    stack = smooth_stack(2)
    stack[1, 2, 1:] = 0
    _, unrecoverable, _ = repair_zero_entries(stack, POSITIONS)
    assert unrecoverable.tolist() == [False, True]


def test_minority_stream_count_is_dropped_survey_wide():
    keep, cleaned, report = clean_channels(smooth_stack(6), [1, 1, 2, 1, 2, 1], POSITIONS)
    assert keep.tolist() == [True, True, False, True, False, True]
    assert cleaned.shape == (4, 3, 30)
    assert report.kept_n_tx == 1 and report.n_dropped_stream_count == 2


def test_keep_policy_retains_mixed_stream_counts():
    keep, _, report = clean_channels(smooth_stack(3), [1, 2, 1], POSITIONS, n_tx_policy="keep")
    assert keep.all() and report.kept_n_tx == 0


def test_require_constant_refuses_a_mixture():
    with pytest.raises(DatasetError, match="transmit streams"):
        clean_channels(smooth_stack(3), [1, 2, 1], POSITIONS, n_tx_policy="require_constant")


def test_non_finite_packets_are_dropped_or_refused():
    stack = smooth_stack(4)
    stack[1, 0, 3] = np.nan
    stack[3] = np.inf
    keep, _, report = clean_channels(stack, [1] * 4, POSITIONS)
    assert keep.tolist() == [True, False, True, False] and report.n_dropped_nonfinite == 2
    with pytest.raises(DatasetError, match="non-finite"):
        clean_channels(stack, [1] * 4, POSITIONS, drop_nonfinite=False)


def test_zero_entry_policies():
    stack = smooth_stack(5)
    stack[[1, 3], 0, 4] = 0
    keep, _, report = clean_channels(stack, [1] * 5, POSITIONS, zero_entry_policy="drop")
    assert keep.tolist() == [True, False, True, False, True]
    assert report.n_dropped_zero_entries == 2
    keep, cleaned, _ = clean_channels(stack, [1] * 5, POSITIONS, zero_entry_policy="keep")
    assert keep.all() and np.count_nonzero(cleaned == 0) == 2
    keep, cleaned, report = clean_channels(stack, [1] * 5, POSITIONS)
    assert keep.all() and not np.any(cleaned == 0) and report.n_repaired_entries == 2


def test_the_input_is_never_modified():
    stack = smooth_stack(3)
    stack[0, 0, 5] = 0
    before = stack.copy()
    clean_channels(stack, [1, 1, 1], POSITIONS)
    assert np.array_equal(stack, before)


def test_reports_add_up():
    total = CleaningReport(10, 1, 2, 0, 3, 7, 1).combine(CleaningReport(5, 0, 0, 1, 0, 4, 1))
    assert total.n_input == 15 and total.n_output == 11 and total.kept_n_tx == 1


@pytest.mark.parametrize(
    ("keyword", "value"), [("n_tx_policy", "majority"), ("zero_entry_policy", "zero")]
)
def test_unknown_policies_are_rejected(keyword, value):
    with pytest.raises(ValueError, match="unknown"):
        clean_channels(smooth_stack(2), [1, 1], POSITIONS, **{keyword: value})
