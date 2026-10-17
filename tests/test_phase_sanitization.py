"""Linear phase sanitization: exact removal of linear terms and the two line estimators."""

from __future__ import annotations

import numpy as np
import pytest

from wifi_fingerprint_locator.phase_sanitization import (
    linear_phase_fit,
    sanitize_channel_phase,
    sanitize_phase,
    unwrap_phase,
)
from wifi_fingerprint_locator.subcarriers import subcarrier_positions

GROUPED = subcarrier_positions("grouped")
UNIFORM = subcarrier_positions("uniform")


@pytest.mark.parametrize("method", ["endpoint", "lstsq"])
@pytest.mark.parametrize("positions", [GROUPED, UNIFORM], ids=["grouped", "uniform"])
def test_injected_linear_terms_are_removed_exactly(rng, method, positions):
    curve = rng.normal(0.0, 0.4, (5, 3, 30))
    slope = rng.uniform(-0.5, 0.5, (5, 3, 1))
    offset = rng.uniform(-3.0, 3.0, (5, 3, 1))
    base = sanitize_phase(curve, positions, method=method)
    shifted = sanitize_phase(curve + slope * positions + offset, positions, method=method)
    assert np.allclose(base, shifted, atol=1e-12)


@pytest.mark.parametrize("method", ["endpoint", "lstsq"])
def test_a_purely_linear_phase_sanitizes_to_zero_with_both_estimators(method):
    phase = 0.37 * GROUPED - 1.2
    assert np.allclose(sanitize_phase(phase, GROUPED, method=method), 0.0, atol=1e-12)


def test_uniform_positions_reproduce_the_endpoint_slope_and_mean_offset_formula(rng):
    phase = np.cumsum(rng.normal(0.0, 0.3, 30))
    m = np.linspace(-28.0, 28.0, 30)
    slope = (phase[-1] - phase[0]) / (m[-1] - m[0])
    expected = phase - slope * m - phase.mean()
    assert np.allclose(sanitize_phase(phase, UNIFORM, method="endpoint"), expected, atol=1e-12)


def test_endpoint_result_has_equal_ends_and_zero_mean(rng):
    phase = np.cumsum(rng.normal(0.0, 0.3, (4, 30)), axis=1)
    sanitized = sanitize_phase(phase, GROUPED, method="endpoint")
    assert np.allclose(sanitized[:, 0], sanitized[:, -1])
    assert np.allclose(sanitized.mean(axis=1), 0.0, atol=1e-12)


def test_least_squares_matches_numpy_polyfit(rng):
    phase = np.cumsum(rng.normal(0.0, 0.3, 30))
    slope, intercept = linear_phase_fit(phase, GROUPED, method="lstsq")
    reference = np.polyfit(GROUPED, phase, 1)
    assert np.allclose([float(slope), float(intercept)], reference, rtol=1e-10)


def test_least_squares_is_less_sensitive_to_a_corrupted_edge_subcarrier(rng):
    true_slope = 0.2
    phase = true_slope * GROUPED + rng.normal(0.0, 0.01, 30)
    phase[-1] += 1.0
    endpoint, _ = linear_phase_fit(phase, GROUPED, method="endpoint")
    lstsq, _ = linear_phase_fit(phase, GROUPED, method="lstsq")
    assert abs(float(lstsq) - true_slope) < abs(float(endpoint) - true_slope)


def test_axis_argument_processes_any_layout(rng):
    phase = rng.normal(size=(30, 4))
    along_first = sanitize_phase(phase, GROUPED, axis=0)
    assert np.allclose(along_first, sanitize_phase(phase.T, GROUPED).T)


def test_channel_phase_is_unwrapped_before_sanitization():
    phase = 0.9 * GROUPED + 2.0
    csi = 4.0 * np.exp(1j * phase)
    unwrapped = unwrap_phase(csi)
    assert np.allclose(unwrapped - unwrapped[0], phase - phase[0])
    assert np.allclose(sanitize_channel_phase(csi, GROUPED), 0.0, atol=1e-10)


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError, match="positions"):
        sanitize_phase(np.zeros(30), np.arange(29.0))
    with pytest.raises(ValueError, match="distinct"):
        sanitize_phase(np.zeros(30), np.zeros(30))
    with pytest.raises(ValueError, match="unknown phase fit"):
        sanitize_phase(np.zeros(30), GROUPED, method="quadratic")
