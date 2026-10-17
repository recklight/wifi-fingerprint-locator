"""Linear phase sanitization of CSI across subcarriers.

The raw phase of commodity WiFi CSI is useless as a fingerprint: every packet
carries a random sampling-time offset (STO, plus the sampling-frequency offset)
that adds a phase slope across subcarriers, and a carrier-frequency and PLL
offset (CFO) that adds a constant. With :math:`m_k` the position of subcarrier
:math:`k` and :math:`\\phi_k` the unwrapped measured phase,

.. math::

    \\phi_k = \\theta_k - 2\\pi \\frac{m_k}{N}\\,\\delta + \\beta + Z,

and both :math:`\\delta` and :math:`\\beta` change from packet to packet. Removing a
straight line fitted across the subcarriers cancels both (Sen et al., 2012):

.. math::

    \\hat\\phi_k = \\phi_k - a\\,m_k - b.

There are two ways to estimate the line:

``endpoint``
    :math:`a = (\\phi_K - \\phi_1)/(m_K - m_1)` and
    :math:`b = \\bar\\phi - a\\,\\bar m`, the transform used by PhaseFi
    (Wang, Gao & Mao, 2015). The result has zero mean and equal end values.
``lstsq``
    The least-squares line through all subcarriers, as used to remove the STO
    slope in SpotFi (Kotaru et al., 2015). Less sensitive to noise on the two
    edge subcarriers.

Both reproduce a purely linear phase exactly, so adding any :math:`a m + b` to
the input leaves the output unchanged. The positions should be the true
subcarrier indices (:func:`~wifi_fingerprint_locator.subcarriers.subcarrier_positions`
with ``"grouped"``); equally spaced positions are a common approximation.

Entries with zero magnitude have no phase (``angle(0) = 0``) and corrupt the
unwrap; repair them first with
:func:`~wifi_fingerprint_locator.channel_cleaning.repair_zero_entries`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Line estimators accepted by :func:`sanitize_phase` and ``features.phase_fit``.
PHASE_FIT_METHODS: tuple[str, ...] = ("endpoint", "lstsq")


def _positions(positions: ArrayLike, n_subcarriers: int) -> NDArray[np.float64]:
    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 1 or values.size != n_subcarriers:
        raise ValueError(
            f"positions must be one-dimensional with {n_subcarriers} entries, got shape "
            f"{values.shape}"
        )
    if values.size < 2 or np.ptp(values) == 0:
        raise ValueError("positions must contain at least two distinct values")
    return values


def unwrap_phase(csi: ArrayLike, *, axis: int = -1) -> NDArray[np.float64]:
    """Phase of complex CSI, unwrapped along the subcarrier axis.

    Parameters
    ----------
    csi : array_like of complex
        Channel values; the subcarriers run along ``axis``.
    axis : int, optional
        Subcarrier axis.

    Returns
    -------
    numpy.ndarray
        Unwrapped phase in radians, same shape as ``csi``.

    Examples
    --------
    >>> phase = np.linspace(0.0, 12.0, 30)
    >>> bool(np.allclose(unwrap_phase(np.exp(1j * phase)), phase))
    True
    """
    return np.unwrap(np.angle(np.asarray(csi)), axis=axis)


def linear_phase_fit(
    phase: ArrayLike, positions: ArrayLike, *, method: str = "endpoint", axis: int = -1
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Slope and intercept of the line removed by :func:`sanitize_phase`.

    Parameters
    ----------
    phase : array_like
        Unwrapped phase; the subcarriers run along ``axis``.
    positions : array_like, shape (n_subcarriers,)
        Subcarrier positions :math:`m_k`.
    method : {"endpoint", "lstsq"}, optional
        Line estimator.
    axis : int, optional
        Subcarrier axis.

    Returns
    -------
    slope, intercept : numpy.ndarray
        Arrays shaped like ``phase`` with ``axis`` removed.

    Raises
    ------
    ValueError
        For an unknown method or positions that do not match the phase.

    References
    ----------
    .. [1] X. Wang, L. Gao, and S. Mao, "PhaseFi: Phase fingerprinting for
       indoor localization with a deep learning approach," in Proc. IEEE
       Global Communications Conference (GLOBECOM), 2015.
    .. [2] M. Kotaru, K. Joshi, D. Bharadia, and S. Katti, "SpotFi: Decimeter
       level localization using WiFi," in Proc. ACM SIGCOMM, 2015, pp. 269-282.

    Examples
    --------
    >>> m = np.array([-2.0, -1.0, 1.0, 2.0])
    >>> slope, intercept = linear_phase_fit(0.5 * m + 3.0, m, method="lstsq")
    >>> round(float(slope), 12), round(float(intercept), 12)
    (0.5, 3.0)
    """
    values = np.moveaxis(np.asarray(phase, dtype=np.float64), axis, -1)
    m = _positions(positions, values.shape[-1])
    m_mean = m.mean()
    phi_mean = values.mean(axis=-1)
    if method == "endpoint":
        slope = (values[..., -1] - values[..., 0]) / (m[-1] - m[0])
    elif method == "lstsq":
        centered = m - m_mean
        slope = (values @ centered) / float(centered @ centered)
    else:
        raise ValueError(f"unknown phase fit {method!r}; choose one of {list(PHASE_FIT_METHODS)}")
    intercept = phi_mean - slope * m_mean
    return slope, intercept


def sanitize_phase(
    phase: ArrayLike, positions: ArrayLike, *, method: str = "endpoint", axis: int = -1
) -> NDArray[np.float64]:
    """Remove the line fitted across subcarriers from an unwrapped phase.

    Parameters
    ----------
    phase : array_like
        Unwrapped phase (see :func:`unwrap_phase`); the subcarriers run along
        ``axis`` and every other axis is processed independently, so a whole
        ``(n_packets, n_rx, n_subcarriers)`` stack is handled at once.
    positions : array_like, shape (n_subcarriers,)
        Subcarrier positions :math:`m_k`.
    method : {"endpoint", "lstsq"}, optional
        Line estimator (see the module description).
    axis : int, optional
        Subcarrier axis.

    Returns
    -------
    numpy.ndarray
        Sanitized phase :math:`\\hat\\phi_k = \\phi_k - a m_k - b`, same shape as
        ``phase``.

    References
    ----------
    .. [1] X. Wang, L. Gao, and S. Mao, "PhaseFi: Phase fingerprinting for
       indoor localization with a deep learning approach," in Proc. IEEE
       Global Communications Conference (GLOBECOM), 2015.
    .. [2] S. Sen, B. Radunovic, R. R. Choudhury, and T. Minka, "You are facing
       the Mona Lisa: Spot localization using PHY layer information," in Proc.
       ACM MobiSys, 2012, pp. 183-196.
    .. [3] M. Kotaru, K. Joshi, D. Bharadia, and S. Katti, "SpotFi: Decimeter
       level localization using WiFi," in Proc. ACM SIGCOMM, 2015, pp. 269-282.

    Examples
    --------
    >>> m = np.linspace(-28.0, 28.0, 30)
    >>> curve = 0.3 * np.sin(m / 5.0)
    >>> base = sanitize_phase(curve, m)
    >>> shifted = sanitize_phase(curve + 0.7 * m - 2.0, m)
    >>> bool(np.allclose(base, shifted)), abs(float(base.mean())) < 1e-12
    (True, True)
    >>> bool(np.isclose(base[0], base[-1]))
    True
    """
    values = np.moveaxis(np.asarray(phase, dtype=np.float64), axis, -1)
    m = _positions(positions, values.shape[-1])
    slope, intercept = linear_phase_fit(values, m, method=method, axis=-1)
    sanitized = values - slope[..., None] * m - intercept[..., None]
    return np.moveaxis(sanitized, -1, axis)


def sanitize_channel_phase(
    csi: ArrayLike, positions: ArrayLike, *, method: str = "endpoint", axis: int = -1
) -> NDArray[np.float64]:
    """Unwrap and sanitize the phase of complex CSI in one step.

    Parameters
    ----------
    csi : array_like of complex
        Channel values; the subcarriers run along ``axis``.
    positions : array_like, shape (n_subcarriers,)
        Subcarrier positions.
    method : {"endpoint", "lstsq"}, optional
        Line estimator.
    axis : int, optional
        Subcarrier axis.

    Returns
    -------
    numpy.ndarray
        Sanitized phase in radians, same shape as ``csi``.

    Examples
    --------
    >>> m = np.linspace(-28.0, 28.0, 30)
    >>> csi = 3.0 * np.exp(1j * (0.4 * m + 1.0))
    >>> float(np.abs(sanitize_channel_phase(csi, m)).max()) < 1e-12
    True
    """
    return sanitize_phase(unwrap_phase(csi, axis=axis), positions, method=method, axis=axis)
