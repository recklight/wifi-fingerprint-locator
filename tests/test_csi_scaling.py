"""Rescaling of reported CSI to sqrt(SNR) units against hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest

from wifi_fingerprint_locator.beamform_report import ChannelSnapshot, RadioMeta
from wifi_fingerprint_locator.csi_scaling import (
    db_to_linear,
    linear_to_db,
    scale_csi,
    total_rss_dbm,
)


def snapshot(n_tx=1, rssi=(40, 40, 40), noise=-92, agc=40, value=10.0):
    meta = RadioMeta(0, n_tx, 3, rssi, noise, agc, (0, 1, 2), True, 0x0100)
    return ChannelSnapshot(np.full((n_tx, 3, 30), value + 0j), 0, meta)


def expected_amplitude(n_tx, rssi, noise, agc, value):
    """The scaling worked out step by step for a report whose entries all equal ``value``."""
    power = sum(10 ** (r / 10) for r in rssi if r != 0)
    rss_dbm = 10 * np.log10(power) - 44 - agc
    csi_power = n_tx * 3 * 30 * value**2
    scale = 10 ** (rss_dbm / 10) / (csi_power / 30)
    noise_dbm = -92 if noise == -127 else noise
    total_noise = 10 ** (noise_dbm / 10) + scale * 3 * n_tx
    gain = {1: 1.0, 2: np.sqrt(2.0), 3: np.sqrt(10**0.45)}[n_tx]
    return value * np.sqrt(scale / total_noise) * gain


def test_decibel_conversions_are_inverse():
    values = np.array([0.5, 1.0, 20.0])
    assert np.allclose(db_to_linear(linear_to_db(values)), values)


def test_total_rss_ignores_absent_chains():
    assert total_rss_dbm((40, 0, 0), 40) == pytest.approx(40 - 44 - 40)
    assert total_rss_dbm((40, 40, 0), 30) == pytest.approx(10 * np.log10(2e4) - 74)


def test_total_rss_of_a_silent_report_is_nan_not_infinite():
    assert np.isnan(total_rss_dbm((0, 0, 0), 40))


@pytest.mark.parametrize(
    ("n_tx", "rssi", "noise", "agc"),
    [
        (1, (40, 40, 40), -92, 40),
        (1, (35, 0, 50), -85, 44),
        (2, (40, 38, 42), -95, 38),
        (3, (30, 31, 32), -90, 45),
    ],
)
def test_scaling_matches_the_hand_computed_value(n_tx, rssi, noise, agc):
    scaled = scale_csi(snapshot(n_tx, rssi, noise, agc))
    assert np.allclose(np.abs(scaled), expected_amplitude(n_tx, rssi, noise, agc, 10.0))


def test_unknown_noise_floor_falls_back_to_the_configured_value():
    unknown = scale_csi(snapshot(noise=-127))
    assert np.allclose(unknown, scale_csi(snapshot(noise=-92)))
    other = scale_csi(snapshot(noise=-127), noise_floor_fallback_dbm=-80.0)
    assert not np.allclose(other, unknown)


def test_stream_power_compensation_can_be_switched_off():
    on = scale_csi(snapshot(n_tx=2))
    off = scale_csi(snapshot(n_tx=2), compensate_stream_power=False)
    assert np.allclose(on, off * np.sqrt(2.0))


def test_raw_report_scale_cancels_when_thermal_noise_dominates():
    weak = {"rssi": (5, 0, 0), "noise": -60, "agc": 40}
    small = scale_csi(snapshot(value=10.0, **weak))
    large = scale_csi(snapshot(value=40.0, **weak))
    assert np.allclose(small, large, rtol=1e-3)


def test_quantization_noise_limits_the_snr_of_coarse_reports():
    # With a strong signal the 8-bit quantization term dominates the noise, so
    # a report with larger raw values carries a higher SNR.
    assert (
        np.abs(scale_csi(snapshot(value=40.0))).mean()
        > np.abs(scale_csi(snapshot(value=10.0))).mean()
    )


def test_reports_without_power_scale_become_nan():
    assert np.isnan(scale_csi(snapshot(rssi=(0, 0, 0)))).all()
    assert np.isnan(scale_csi(snapshot(value=0.0))).all()
