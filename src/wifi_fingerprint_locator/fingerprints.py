"""Fingerprint representations of a channel measurement.

A cleaned measurement is one transmit stream received on ``n_rx`` antennas
over 30 subcarriers, a complex array :math:`H_{r,k}`. A fingerprint flattens it
into a real vector, antenna by antenna (all subcarriers of the first antenna,
then the second, ...). Each representation has a name, and
``features.representation`` selects one:

==========================  =========  ====================================================
name                        length     content
==========================  =========  ====================================================
``amplitude``               30 n_rx    :math:`|H|`, linear (DeepFi; Wang et al., 2015, 2017)
``amplitude_db``            30 n_rx    :math:`20\\log_{10}\\max(|H|, \\epsilon)`
``sanitized_phase``         30 n_rx    :math:`\\hat\\phi` after linear sanitization (PhaseFi)
``amplitude_phase``         60 n_rx    ``amplitude`` followed by ``sanitized_phase``
``real_imag``               60 n_rx    :math:`\\Re H` followed by :math:`\\Im H`, raw phase
``sanitized_real_imag``     60 n_rx    real and imaginary parts of :math:`|H| e^{j\\hat\\phi}`
==========================  =========  ====================================================

``amplitude_phase`` uses amplitude and phase together, as BiLoc does (Wang,
Gao & Mao, 2017), but in a simpler form: BiLoc pairs the average amplitudes of
adjacent antennas with angles of arrival estimated from their calibrated phase
differences, while this fingerprint keeps the per-antenna amplitude and
sanitized phase.

``real_imag`` keeps the random per-packet phase offsets, so I expect it to be
the weakest representation; ``sanitized_real_imag`` is the ablation that
removes them while keeping the Cartesian form. The scales differ between the
blocks of ``amplitude_phase`` (amplitudes in :math:`\\sqrt{\\mathrm{SNR}}` units,
phases in radians); the models scale every feature separately, which makes a
block-wise scaling unnecessary.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .phase_sanitization import sanitize_channel_phase

_Extractor = Callable[
    [NDArray[np.complex128], NDArray[np.float64], str, float], NDArray[np.float64]
]


def _flatten(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return values.reshape(values.shape[0], -1)


def _amplitude(
    csi: NDArray[np.complex128], positions: NDArray[np.float64], phase_fit: str, floor_db: float
) -> NDArray[np.float64]:
    return _flatten(np.abs(csi))


def _amplitude_db(
    csi: NDArray[np.complex128], positions: NDArray[np.float64], phase_fit: str, floor_db: float
) -> NDArray[np.float64]:
    floor = 10.0 ** (floor_db / 20.0)
    return _flatten(20.0 * np.log10(np.maximum(np.abs(csi), floor)))


def _sanitized_phase(
    csi: NDArray[np.complex128], positions: NDArray[np.float64], phase_fit: str, floor_db: float
) -> NDArray[np.float64]:
    return _flatten(sanitize_channel_phase(csi, positions, method=phase_fit))


def _amplitude_phase(
    csi: NDArray[np.complex128], positions: NDArray[np.float64], phase_fit: str, floor_db: float
) -> NDArray[np.float64]:
    return np.hstack(
        [
            _amplitude(csi, positions, phase_fit, floor_db),
            _sanitized_phase(csi, positions, phase_fit, floor_db),
        ]
    )


def _real_imag(
    csi: NDArray[np.complex128], positions: NDArray[np.float64], phase_fit: str, floor_db: float
) -> NDArray[np.float64]:
    return np.hstack([_flatten(csi.real), _flatten(csi.imag)])


def _sanitized_real_imag(
    csi: NDArray[np.complex128], positions: NDArray[np.float64], phase_fit: str, floor_db: float
) -> NDArray[np.float64]:
    phase = sanitize_channel_phase(csi, positions, method=phase_fit)
    rebuilt = np.abs(csi) * np.exp(1j * phase)
    return np.hstack([_flatten(rebuilt.real), _flatten(rebuilt.imag)])


_EXTRACTORS: dict[str, _Extractor] = {
    "amplitude": _amplitude,
    "amplitude_db": _amplitude_db,
    "sanitized_phase": _sanitized_phase,
    "amplitude_phase": _amplitude_phase,
    "real_imag": _real_imag,
    "sanitized_real_imag": _sanitized_real_imag,
}
#: Fingerprint representations accepted by :func:`fingerprint_matrix` and
#: ``features.representation``.
REPRESENTATIONS: tuple[str, ...] = tuple(_EXTRACTORS)

_BLOCKS: dict[str, tuple[str, ...]] = {
    "amplitude": ("amplitude",),
    "amplitude_db": ("amplitude_db",),
    "sanitized_phase": ("phase",),
    "amplitude_phase": ("amplitude", "phase"),
    "real_imag": ("real", "imag"),
    "sanitized_real_imag": ("real", "imag"),
}


def fingerprint_matrix(
    csi: ArrayLike,
    representation: str,
    positions: ArrayLike,
    *,
    phase_fit: str = "endpoint",
    amplitude_floor_db: float = -20.0,
) -> NDArray[np.float64]:
    """Turn a stack of channel measurements into a fingerprint matrix.

    Parameters
    ----------
    csi : array_like of complex, shape (n_packets, n_rx, n_subcarriers)
        Cleaned channel stack (one transmit stream).
    representation : str
        One of :data:`REPRESENTATIONS`.
    positions : array_like, shape (n_subcarriers,)
        Subcarrier positions, used by the phase representations.
    phase_fit : {"endpoint", "lstsq"}, optional
        Line estimator of the phase sanitization.
    amplitude_floor_db : float, optional
        Smallest amplitude, in dB, kept by ``amplitude_db`` (avoids ``log(0)``).

    Returns
    -------
    numpy.ndarray, shape (n_packets, n_features)
        One fingerprint per row, antenna-major.

    Raises
    ------
    ValueError
        For an unknown representation or a stack of the wrong shape.

    References
    ----------
    .. [1] X. Wang, L. Gao, S. Mao, and S. Pandey, "DeepFi: Deep learning for
       indoor fingerprinting using channel state information," in Proc. IEEE
       Wireless Communications and Networking Conference (WCNC), 2015,
       pp. 1666-1671.
    .. [2] X. Wang, L. Gao, S. Mao, and S. Pandey, "CSI-based fingerprinting
       for indoor localization: A deep learning approach," IEEE Transactions
       on Vehicular Technology, vol. 66, no. 1, pp. 763-776, 2017.
    .. [3] X. Wang, L. Gao, and S. Mao, "PhaseFi: Phase fingerprinting for
       indoor localization with a deep learning approach," in Proc. IEEE
       Global Communications Conference (GLOBECOM), 2015.
    .. [4] X. Wang, L. Gao, and S. Mao, "BiLoc: Bi-modal deep learning for
       indoor localization with commodity 5GHz WiFi," IEEE Access, vol. 5,
       pp. 4209-4220, 2017.

    Examples
    --------
    >>> m = np.linspace(-28.0, 28.0, 30)
    >>> stack = np.ones((2, 3, 30)) * np.exp(1j * 0.1 * m)
    >>> [fingerprint_matrix(stack, name, m).shape[1] for name in REPRESENTATIONS]
    [90, 90, 90, 180, 180, 180]
    >>> float(np.abs(fingerprint_matrix(stack, "sanitized_phase", m)).max()) < 1e-12
    True
    """
    if representation not in _EXTRACTORS:
        raise ValueError(
            f"unknown representation {representation!r}; choose one of {list(REPRESENTATIONS)}"
        )
    values = np.asarray(csi, dtype=np.complex128)
    if values.ndim != 3:
        raise ValueError(f"csi must be shaped (n_packets, n_rx, n_subcarriers), got {values.shape}")
    m = np.asarray(positions, dtype=np.float64)
    if m.shape != (values.shape[-1],):
        raise ValueError(f"positions must have {values.shape[-1]} entries, got shape {m.shape}")
    return _EXTRACTORS[representation](values, m, phase_fit, float(amplitude_floor_db))


def fingerprint_names(representation: str, n_rx: int, n_subcarriers: int = 30) -> list[str]:
    """Column names of :func:`fingerprint_matrix`, e.g. ``amplitude_rx0_sc00``.

    Examples
    --------
    >>> names = fingerprint_names("amplitude_phase", 3)
    >>> len(names), names[0], names[-1]
    (180, 'amplitude_rx0_sc00', 'phase_rx2_sc29')
    """
    if representation not in _BLOCKS:
        raise ValueError(
            f"unknown representation {representation!r}; choose one of {list(REPRESENTATIONS)}"
        )
    return [
        f"{block}_rx{antenna}_sc{subcarrier:02d}"
        for block in _BLOCKS[representation]
        for antenna in range(n_rx)
        for subcarrier in range(n_subcarriers)
    ]
