"""Positions of the 30 reported subcarrier groups on the OFDM subcarrier axis.

The Intel 5300 reports channel state for 30 subcarrier groups, using the
grouping defined for beamforming feedback in IEEE Std 802.11n-2009: every
second subcarrier on a 20 MHz channel (grouping :math:`N_g = 2`, indices
:math:`-28, -26, \\dots, -2, -1, 1, 3, \\dots, 27, 28`) and every fourth on a
40 MHz channel (:math:`N_g = 4`). Subcarriers are 312.5 kHz apart.

Linear phase sanitization fits a line against these positions, so the choice
matters. The ``grouped`` layout uses the true indices; the ``uniform`` layout
spreads 30 equally spaced points between the first and the last index, a
common approximation that ignores the irregular spacing around DC. It is an
option so that the two layouts can be compared.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

#: Number of subcarrier groups in every beamforming report.
N_SUBCARRIER_GROUPS: int = 30

#: OFDM subcarrier spacing of 802.11n, in hertz.
SUBCARRIER_SPACING_HZ: float = 312_500.0

#: Layouts accepted by :func:`subcarrier_positions` and ``features.subcarrier_layout``.
SUBCARRIER_LAYOUTS: tuple[str, ...] = ("grouped", "uniform")

_GROUPED_INDICES: dict[int, NDArray[np.int64]] = {
    # Ng = 2 on 20 MHz: -28, -26, ..., -2, -1, 1, 3, ..., 27, 28
    20: np.concatenate([np.arange(-28, -1, 2), [-1, 1], np.arange(3, 28, 2), [28]]).astype(
        np.int64
    ),
    # Ng = 4 on 40 MHz: -58, -54, ..., -2, 2, ..., 54, 58
    40: np.concatenate([np.arange(-58, 0, 4), np.arange(2, 59, 4)]).astype(np.int64),
}


def reported_subcarrier_indices(bandwidth_mhz: int = 20) -> NDArray[np.int64]:
    """OFDM subcarrier index of each reported group.

    Parameters
    ----------
    bandwidth_mhz : int, optional
        20 or 40.

    Returns
    -------
    numpy.ndarray, shape (30,)
        Integer indices in reporting order.

    Raises
    ------
    ValueError
        If the bandwidth is not 20 or 40 MHz.

    References
    ----------
    .. [1] IEEE Std 802.11n-2009, "IEEE Standard for Information technology --
       Local and metropolitan area networks -- Specific requirements --
       Part 11: Wireless LAN Medium Access Control (MAC) and Physical Layer
       (PHY) Specifications Amendment 5: Enhancements for Higher Throughput,"
       2009.

    Examples
    --------
    >>> indices = reported_subcarrier_indices(20)
    >>> len(indices), indices[[0, 14, 15, -1]].tolist()
    (30, [-28, -1, 1, 28])
    >>> reported_subcarrier_indices(40)[[0, -1]].tolist()
    [-58, 58]
    """
    try:
        return _GROUPED_INDICES[int(bandwidth_mhz)].copy()
    except KeyError:
        raise ValueError(f"bandwidth must be 20 or 40 MHz, got {bandwidth_mhz!r}") from None


def subcarrier_positions(layout: str = "grouped", bandwidth_mhz: int = 20) -> NDArray[np.float64]:
    """Subcarrier positions used as the abscissa of phase fits and interpolation.

    Parameters
    ----------
    layout : {"grouped", "uniform"}, optional
        ``grouped`` returns the true indices of :func:`reported_subcarrier_indices`;
        ``uniform`` returns 30 equally spaced points between its first and last
        entry.
    bandwidth_mhz : int, optional
        20 or 40.

    Returns
    -------
    numpy.ndarray, shape (30,)
        Positions in units of the subcarrier spacing.

    Raises
    ------
    ValueError
        For an unknown layout or bandwidth.

    Examples
    --------
    >>> grouped = subcarrier_positions("grouped")
    >>> uniform = subcarrier_positions("uniform")
    >>> float(grouped[15]), round(float(uniform[15]), 4), float(uniform[-1])
    (1.0, 0.9655, 28.0)
    >>> round(float(uniform.mean()), 12), round(float(grouped.mean()), 4)
    (0.0, 0.4333)
    """
    indices = reported_subcarrier_indices(bandwidth_mhz).astype(np.float64)
    if layout == "grouped":
        return indices
    if layout == "uniform":
        return np.linspace(indices[0], indices[-1], indices.size)
    raise ValueError(
        f"unknown subcarrier layout {layout!r}; choose one of {list(SUBCARRIER_LAYOUTS)}"
    )


def subcarrier_offsets_hz(bandwidth_mhz: int = 20) -> NDArray[np.float64]:
    """Frequency offset of each reported group from the channel center, in hertz.

    Examples
    --------
    >>> float(subcarrier_offsets_hz(20)[-1])
    8750000.0
    """
    return reported_subcarrier_indices(bandwidth_mhz).astype(np.float64) * SUBCARRIER_SPACING_HZ
