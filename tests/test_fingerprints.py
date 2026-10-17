"""Fingerprint representations: lengths, flattening order and the expected invariances."""

from __future__ import annotations

import numpy as np
import pytest

from wifi_fingerprint_locator.fingerprints import (
    REPRESENTATIONS,
    fingerprint_matrix,
    fingerprint_names,
)
from wifi_fingerprint_locator.subcarriers import subcarrier_positions

POSITIONS = subcarrier_positions("grouped")


def channel(rng, n_packets=4, n_rx=3):
    shape = (n_packets, n_rx, 30)
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


@pytest.mark.parametrize(
    ("name", "length"),
    [
        ("amplitude", 90),
        ("amplitude_db", 90),
        ("sanitized_phase", 90),
        ("amplitude_phase", 180),
        ("real_imag", 180),
        ("sanitized_real_imag", 180),
    ],
)
def test_lengths_and_names_match(rng, name, length):
    matrix = fingerprint_matrix(channel(rng), name, POSITIONS)
    assert matrix.shape == (4, length)
    assert len(fingerprint_names(name, 3)) == length


def test_registry_lists_every_representation():
    assert set(REPRESENTATIONS) == {
        "amplitude",
        "amplitude_db",
        "sanitized_phase",
        "amplitude_phase",
        "real_imag",
        "sanitized_real_imag",
    }


def test_flattening_is_antenna_major(rng):
    csi = channel(rng, 1)
    amplitude = fingerprint_matrix(csi, "amplitude", POSITIONS)[0]
    assert np.allclose(amplitude[30:60], np.abs(csi[0, 1]))
    assert fingerprint_names("amplitude", 3)[30] == "amplitude_rx1_sc00"


def test_bimodal_fingerprint_is_amplitude_then_phase(rng):
    csi = channel(rng)
    both = fingerprint_matrix(csi, "amplitude_phase", POSITIONS)
    assert np.allclose(both[:, :90], fingerprint_matrix(csi, "amplitude", POSITIONS))
    assert np.allclose(both[:, 90:], fingerprint_matrix(csi, "sanitized_phase", POSITIONS))


def test_random_packet_phase_offsets_change_real_imag_but_not_the_sanitized_forms():
    smooth = (1.5 + np.cos(POSITIONS / 6.0)) * np.exp(1j * 0.2 * np.sin(POSITIONS / 4.0))
    base = np.tile(smooth, (1, 3, 1))
    shifted = base * np.exp(1j * (0.15 * POSITIONS + 2.4))
    for name in ("amplitude", "sanitized_phase", "amplitude_phase", "sanitized_real_imag"):
        assert np.allclose(
            fingerprint_matrix(base, name, POSITIONS),
            fingerprint_matrix(shifted, name, POSITIONS),
            atol=1e-9,
        ), name
    assert not np.allclose(
        fingerprint_matrix(base, "real_imag", POSITIONS),
        fingerprint_matrix(shifted, "real_imag", POSITIONS),
    )


def test_amplitude_in_decibels_is_floored():
    csi = np.zeros((1, 1, 30), dtype=complex)
    csi[0, 0, 0] = 10.0
    values = fingerprint_matrix(csi, "amplitude_db", POSITIONS, amplitude_floor_db=-30.0)
    assert values[0, 0] == pytest.approx(20.0)
    assert values[0, 1] == pytest.approx(-30.0)


def test_unknown_representation_is_rejected(rng):
    with pytest.raises(ValueError, match="unknown representation"):
        fingerprint_matrix(channel(rng), "spectrogram", POSITIONS)
    with pytest.raises(ValueError, match="unknown representation"):
        fingerprint_names("spectrogram", 3)
