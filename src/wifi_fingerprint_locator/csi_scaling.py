"""Conversion of raw reported CSI to absolute units of :math:`\\sqrt{\\mathrm{SNR}}`.

The NIC normalizes every report with its automatic gain control, so the raw
8-bit values carry no absolute power. The header does: the per-chain RSSI
(dB relative to the AGC setting) gives the total received power

.. math::

    P_\\mathrm{rx} = 10 \\log_{10} \\sum_{c:\\,\\mathrm{rssi}_c \\neq 0}
    10^{\\mathrm{rssi}_c / 10} - 44 - \\mathrm{AGC}\\quad[\\mathrm{dBm}],

and the reported noise floor :math:`N` (dBm) gives the noise power. With
:math:`s = 10^{P_\\mathrm{rx}/10} / (\\sum |H|^2 / 30)` the channel is rescaled to

.. math::

    \\tilde H = H \\sqrt{\\frac{s}{10^{N/10} + s\\,N_\\mathrm{rx} N_\\mathrm{tx}}},

where the second noise term models the 8-bit quantization error. A report
with two transmit streams is further multiplied by :math:`\\sqrt{2}` and one with
three by :math:`\\sqrt{10^{0.45}}` (the NIC approximates the 4.8 dB of three
streams by 4.5 dB), because each stream carries a share of the transmit
power. This follows the CSI Tool convention (Halperin et al., 2011).

A report whose RSSI fields are all zero, or whose CSI is all zero, has no
defined scale: the result is NaN and the cleaning stage drops it, instead of
letting ``log10(0) = -inf`` turn into infinities downstream.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .beamform_report import UNKNOWN_NOISE_DBM, ChannelSnapshot

#: Offset (dB) between the summed RSSI and the received power in dBm, before AGC.
RSSI_OFFSET_DB: float = 44.0


def db_to_linear(value_db: ArrayLike) -> NDArray[np.float64]:
    """Power ratio of a value in decibels, :math:`10^{x/10}`.

    Examples
    --------
    >>> db_to_linear([0.0, 10.0, -3.0]).round(4).tolist()
    [1.0, 10.0, 0.5012]
    """
    return np.power(10.0, np.asarray(value_db, dtype=np.float64) / 10.0)


def linear_to_db(value: ArrayLike) -> NDArray[np.float64]:
    """Decibels of a power ratio, :math:`10 \\log_{10} x`; zero maps to ``-inf``.

    Examples
    --------
    >>> linear_to_db([1.0, 100.0]).tolist()
    [0.0, 20.0]
    """
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(np.asarray(value, dtype=np.float64))


def total_rss_dbm(rssi_db: Sequence[int] | ArrayLike, agc_db: float) -> float:
    """Total received signal strength of a report, in dBm.

    Parameters
    ----------
    rssi_db : sequence of int
        Per-chain RSSI values; zeros mark absent chains and are ignored.
    agc_db : float
        AGC setting of the report.

    Returns
    -------
    float
        The received power, or NaN when every chain reads zero.

    References
    ----------
    .. [1] D. Halperin, W. Hu, A. Sheth, and D. Wetherall, "Tool release:
       Gathering 802.11n traces with channel state information," ACM SIGCOMM
       Computer Communication Review, vol. 41, no. 1, p. 53, 2011.

    Examples
    --------
    >>> round(total_rss_dbm((40, 40, 0), agc_db=40), 4)
    -40.9897
    >>> total_rss_dbm((0, 0, 0), agc_db=40)
    nan
    """
    values = np.asarray(rssi_db, dtype=np.float64)
    present = values[values != 0]
    if present.size == 0:
        return float("nan")
    return float(linear_to_db(db_to_linear(present).sum())) - RSSI_OFFSET_DB - float(agc_db)


def scale_csi(
    snapshot: ChannelSnapshot,
    *,
    noise_floor_fallback_dbm: float = -92.0,
    three_stream_gain_db: float = 4.5,
    compensate_stream_power: bool = True,
) -> NDArray[np.complex128]:
    """Rescale a report's CSI to units of :math:`\\sqrt{\\mathrm{SNR}}`.

    Parameters
    ----------
    snapshot : ChannelSnapshot
        The decoded report.
    noise_floor_fallback_dbm : float, optional
        Noise floor used when the report says it is unknown (-127 dBm).
    three_stream_gain_db : float, optional
        Power gain applied to three-stream reports.
    compensate_stream_power : bool, optional
        Apply the per-stream power compensation (:math:`\\sqrt{2}` for two
        streams, ``three_stream_gain_db`` for three).

    Returns
    -------
    numpy.ndarray, shape (n_tx, n_rx, 30)
        The rescaled channel; NaN everywhere when the scale is undefined
        (all RSSI fields or all CSI entries zero).

    References
    ----------
    .. [1] D. Halperin, W. Hu, A. Sheth, and D. Wetherall, "Tool release:
       Gathering 802.11n traces with channel state information," ACM SIGCOMM
       Computer Communication Review, vol. 41, no. 1, p. 53, 2011.

    Examples
    --------
    >>> from wifi_fingerprint_locator.beamform_report import RadioMeta
    >>> meta = RadioMeta(0, 1, 3, (40, 40, 40), -92, 40, (0, 1, 2), True, 0x0100)
    >>> scaled = scale_csi(ChannelSnapshot(np.full((1, 3, 30), 10.0 + 0j), 0, meta))
    >>> round(float(np.abs(scaled[0, 0, 0])), 3)
    5.772
    """
    meta = snapshot.meta
    csi = snapshot.csi
    rss_dbm = total_rss_dbm(meta.rssi_db, meta.agc_db)
    csi_power = float(np.sum(np.abs(csi) ** 2))
    if not np.isfinite(rss_dbm) or csi_power <= 0.0:
        return np.full(csi.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    scale = float(db_to_linear(rss_dbm)) / (csi_power / csi.shape[-1])
    noise_dbm = (
        noise_floor_fallback_dbm if meta.noise_dbm == UNKNOWN_NOISE_DBM else float(meta.noise_dbm)
    )
    thermal_noise = float(db_to_linear(noise_dbm))
    quantization_noise = scale * meta.n_rx * meta.n_tx
    scaled = csi * np.sqrt(scale / (thermal_noise + quantization_noise))
    if compensate_stream_power:
        if meta.n_tx == 2:
            scaled = scaled * np.sqrt(2.0)
        elif meta.n_tx == 3:
            scaled = scaled * np.sqrt(float(db_to_linear(three_stream_gain_db)))
    return scaled
