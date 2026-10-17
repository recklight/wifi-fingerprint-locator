"""Subcarrier positions: the 802.11n grouping and the uniform approximation."""

from __future__ import annotations

import numpy as np
import pytest

from wifi_fingerprint_locator.subcarriers import (
    N_SUBCARRIER_GROUPS,
    SUBCARRIER_SPACING_HZ,
    reported_subcarrier_indices,
    subcarrier_offsets_hz,
    subcarrier_positions,
)


def test_twenty_megahertz_grouping_is_every_second_subcarrier_with_both_sides_of_dc():
    indices = reported_subcarrier_indices(20)
    assert indices.size == N_SUBCARRIER_GROUPS
    assert indices[:14].tolist() == list(range(-28, -1, 2))
    assert indices[14:16].tolist() == [-1, 1]
    assert indices[16:].tolist() == [*range(3, 28, 2), 28]
    assert 0 not in indices


def test_forty_megahertz_grouping_is_every_fourth_subcarrier():
    indices = reported_subcarrier_indices(40)
    assert indices.size == 30 and indices[0] == -58 and indices[-1] == 58
    assert set(np.diff(indices[:15]).tolist()) == {4}


def test_uniform_layout_spans_the_same_band_with_equal_steps():
    uniform = subcarrier_positions("uniform", 20)
    assert uniform[0] == -28.0 and uniform[-1] == 28.0
    assert np.allclose(np.diff(uniform), 56.0 / 29.0)
    assert np.isclose(uniform.mean(), 0.0)


def test_grouped_positions_are_the_reported_indices():
    assert np.array_equal(subcarrier_positions("grouped", 40), reported_subcarrier_indices(40))


def test_offsets_are_indices_times_the_subcarrier_spacing():
    assert np.allclose(
        subcarrier_offsets_hz(20), reported_subcarrier_indices(20) * SUBCARRIER_SPACING_HZ
    )


@pytest.mark.parametrize(("layout", "bandwidth"), [("grouped", 80), ("spiral", 20)])
def test_unknown_layouts_and_bandwidths_are_rejected(layout, bandwidth):
    with pytest.raises(ValueError, match=r"bandwidth|layout"):
        subcarrier_positions(layout, bandwidth)
