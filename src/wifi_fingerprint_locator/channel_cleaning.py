"""Validation and repair of a stack of channel measurements before feature extraction.

Three problems occur in real captures. Each has an explicit policy, and every
packet or entry it drops or repairs is counted:

* Non-finite values. A report whose RSSI fields are all zero has no
  defined power scale (see :mod:`~wifi_fingerprint_locator.csi_scaling`) and
  becomes NaN. Such packets are dropped (or, with ``drop_nonfinite=False``,
  rejected with an error).
* A changing number of transmit streams. The NIC switches between one and
  two spatial streams from packet to packet, and the per-stream power
  compensation then puts the packets on different amplitude scales.
  ``n_tx_policy`` keeps them all (``keep``), refuses a mixture
  (``require_constant``) or keeps only the most common stream count
  (``drop_minority``).
* Zero entries. The 8-bit quantizer occasionally reports an entry as
  exactly ``0 + 0j``. Its phase is undefined and corrupts the phase unwrap, and
  its amplitude is an outlier. ``zero_entry_policy`` interpolates the real and
  imaginary parts from the other subcarriers of the *same antenna*
  (``interpolate``), drops the packet (``drop``) or leaves it (``keep``).
  Interpolation never crosses antennas.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .errors import DatasetError

_LOGGER = logging.getLogger(__name__)

#: Policies accepted by ``cleaning.n_tx_policy``.
N_TX_POLICIES: tuple[str, ...] = ("keep", "require_constant", "drop_minority")
#: Policies accepted by ``cleaning.zero_entry_policy``.
ZERO_ENTRY_POLICIES: tuple[str, ...] = ("interpolate", "drop", "keep")


@dataclass(frozen=True)
class CleaningReport:
    """What :func:`clean_channels` dropped and repaired.

    Attributes
    ----------
    n_input : int
        Packets received.
    n_dropped_nonfinite : int
        Packets dropped for NaN or infinite values.
    n_dropped_stream_count : int
        Packets dropped by the ``drop_minority`` stream-count policy.
    n_dropped_zero_entries : int
        Packets dropped because of zero entries (``drop`` policy, or too few
        non-zero subcarriers left to interpolate from).
    n_repaired_entries : int
        Zero entries replaced by interpolation.
    n_output : int
        Packets kept.
    kept_n_tx : int
        The stream count kept by ``drop_minority``; 0 for the other policies.

    Examples
    --------
    >>> CleaningReport(10, 1, 0, 0, 2, 9, 0).as_dict()["n_output"]
    9
    """

    n_input: int
    n_dropped_nonfinite: int
    n_dropped_stream_count: int
    n_dropped_zero_entries: int
    n_repaired_entries: int
    n_output: int
    kept_n_tx: int

    def as_dict(self) -> dict[str, int]:
        """Return the counters as a plain dictionary."""
        return asdict(self)

    def combine(self, other: CleaningReport) -> CleaningReport:
        """Add the counters of two reports (``kept_n_tx`` is kept when they agree)."""
        kept = self.kept_n_tx if self.kept_n_tx == other.kept_n_tx else 0
        totals = {
            key: getattr(self, key) + getattr(other, key)
            for key in self.as_dict()
            if key != "kept_n_tx"
        }
        return CleaningReport(**totals, kept_n_tx=kept)


def repair_zero_entries(
    csi: ArrayLike, positions: ArrayLike
) -> tuple[NDArray[np.complex128], NDArray[np.bool_], int]:
    """Replace exact-zero entries by interpolation along the subcarrier axis.

    The real and imaginary parts are interpolated linearly against the
    subcarrier positions, from the non-zero entries of the same antenna of the
    same packet; at the band edges the nearest non-zero value is repeated.

    Parameters
    ----------
    csi : array_like of complex, shape (n_packets, n_rx, n_subcarriers)
        Channel stack.
    positions : array_like, shape (n_subcarriers,)
        Subcarrier positions.

    Returns
    -------
    repaired : numpy.ndarray of complex128
        Copy of ``csi`` with zero entries filled in.
    unrecoverable : numpy.ndarray of bool, shape (n_packets,)
        Packets in which some antenna has fewer than two non-zero entries
        left; their zeros are left in place.
    n_repaired : int
        Number of entries replaced.

    Examples
    --------
    >>> m = np.arange(5.0)
    >>> row = np.array([[[1 + 1j, 2 + 2j, 0j, 4 + 4j, 5 + 5j]]])
    >>> repaired, unrecoverable, count = repair_zero_entries(row, m)
    >>> complex(repaired[0, 0, 2]), bool(unrecoverable[0]), count
    ((3+3j), False, 1)
    """
    values = np.array(csi, dtype=np.complex128, copy=True)
    if values.ndim != 3:
        raise ValueError(f"csi must be shaped (n_packets, n_rx, n_subcarriers), got {values.shape}")
    m = np.asarray(positions, dtype=np.float64)
    if m.shape != (values.shape[-1],):
        raise ValueError(f"positions must have {values.shape[-1]} entries, got shape {m.shape}")
    zero = values == 0
    unrecoverable = np.zeros(values.shape[0], dtype=bool)
    n_repaired = 0
    # Only the few affected (packet, antenna) rows are visited.
    for packet, antenna in zip(*np.nonzero(zero.any(axis=-1)), strict=True):
        mask = zero[packet, antenna]
        if np.count_nonzero(~mask) < 2:
            unrecoverable[packet] = True
            continue
        row = values[packet, antenna]
        real = np.interp(m[mask], m[~mask], row.real[~mask])
        imag = np.interp(m[mask], m[~mask], row.imag[~mask])
        row[mask] = real + 1j * imag
        n_repaired += int(mask.sum())
    return values, unrecoverable, n_repaired


def clean_channels(
    csi: ArrayLike,
    n_tx: ArrayLike,
    positions: ArrayLike,
    *,
    n_tx_policy: str = "drop_minority",
    zero_entry_policy: str = "interpolate",
    drop_nonfinite: bool = True,
) -> tuple[NDArray[np.bool_], NDArray[np.complex128], CleaningReport]:
    """Apply the cleaning policies to a stack of channel measurements.

    Parameters
    ----------
    csi : array_like of complex, shape (n_packets, n_rx, n_subcarriers)
        Channel stack (one transmit stream already selected).
    n_tx : array_like of int, shape (n_packets,)
        Number of transmit streams each packet was reported with.
    positions : array_like, shape (n_subcarriers,)
        Subcarrier positions, the abscissa for interpolation.
    n_tx_policy : {"keep", "require_constant", "drop_minority"}, optional
        What to do with a mixture of stream counts.
    zero_entry_policy : {"interpolate", "drop", "keep"}, optional
        What to do with exact-zero entries.
    drop_nonfinite : bool, optional
        Drop packets with NaN or infinite values; when False they raise.

    Returns
    -------
    keep : numpy.ndarray of bool, shape (n_packets,)
        Which input packets survive.
    cleaned : numpy.ndarray of complex128, shape (n_kept, n_rx, n_subcarriers)
        The surviving packets, zero entries repaired when requested.
    report : CleaningReport
        Counts of everything dropped and repaired.

    Raises
    ------
    DatasetError
        For ``require_constant`` with mixed stream counts, or non-finite
        values with ``drop_nonfinite=False``.
    ValueError
        For an unknown policy or inconsistent shapes.

    Examples
    --------
    >>> m = np.arange(4.0)
    >>> stack = np.ones((4, 1, 4), dtype=complex)
    >>> stack[1, 0, 2] = 0
    >>> stack[3] = np.nan
    >>> keep, cleaned, report = clean_channels(stack, [1, 1, 2, 1], m)
    >>> keep.tolist(), cleaned.shape, report.n_repaired_entries, report.kept_n_tx
    ([True, True, False, False], (2, 1, 4), 1, 1)
    """
    if n_tx_policy not in N_TX_POLICIES:
        raise ValueError(
            f"unknown n_tx policy {n_tx_policy!r}; choose one of {list(N_TX_POLICIES)}"
        )
    if zero_entry_policy not in ZERO_ENTRY_POLICIES:
        raise ValueError(
            f"unknown zero-entry policy {zero_entry_policy!r}; "
            f"choose one of {list(ZERO_ENTRY_POLICIES)}"
        )
    values = np.asarray(csi, dtype=np.complex128)
    if values.ndim != 3:
        raise ValueError(f"csi must be shaped (n_packets, n_rx, n_subcarriers), got {values.shape}")
    streams = np.asarray(n_tx, dtype=np.int64)
    if streams.shape != (values.shape[0],):
        raise ValueError(f"n_tx must have one entry per packet, got shape {streams.shape}")

    keep = np.ones(values.shape[0], dtype=bool)
    finite = np.isfinite(values).all(axis=(1, 2))
    n_nonfinite = int(np.sum(~finite))
    if n_nonfinite and not drop_nonfinite:
        raise DatasetError(
            f"{n_nonfinite} packet(s) have non-finite channel values (for example all RSSI "
            "fields zero); set cleaning.drop_nonfinite=true to drop them"
        )
    keep &= finite

    kept_n_tx = 0
    n_stream_dropped = 0
    present = np.unique(streams[keep])
    if n_tx_policy == "require_constant" and present.size > 1:
        raise DatasetError(
            f"packets mix {present.tolist()} transmit streams; set cleaning.n_tx_policy to "
            "'drop_minority' or 'keep' to accept them"
        )
    if n_tx_policy == "drop_minority" and present.size:
        counts = np.array([np.sum(streams[keep] == value) for value in present])
        kept_n_tx = int(present[np.argmax(counts)])  # ties go to the smaller stream count
        minority = keep & (streams != kept_n_tx)
        n_stream_dropped = int(minority.sum())
        keep &= ~minority

    n_zero_dropped = 0
    n_repaired = 0
    cleaned = values[keep]
    if zero_entry_policy != "keep" and cleaned.size:
        has_zero = (cleaned == 0).any(axis=(1, 2))
        if zero_entry_policy == "drop":
            drop = has_zero
        else:
            cleaned, drop, n_repaired = repair_zero_entries(cleaned, positions)
        n_zero_dropped = int(drop.sum())
        kept_rows = np.flatnonzero(keep)
        keep[kept_rows[drop]] = False
        cleaned = cleaned[~drop]
    else:
        cleaned = np.array(cleaned, dtype=np.complex128, copy=True)

    report = CleaningReport(
        n_input=int(values.shape[0]),
        n_dropped_nonfinite=n_nonfinite,
        n_dropped_stream_count=n_stream_dropped,
        n_dropped_zero_entries=n_zero_dropped,
        n_repaired_entries=n_repaired,
        n_output=int(keep.sum()),
        kept_n_tx=kept_n_tx,
    )
    if n_nonfinite or n_stream_dropped or n_zero_dropped:
        _LOGGER.info(
            "cleaning dropped %d non-finite, %d minority-stream and %d zero-entry packet(s) "
            "of %d; %d zero entries repaired",
            n_nonfinite,
            n_stream_dropped,
            n_zero_dropped,
            report.n_input,
            n_repaired,
        )
    return keep, cleaned, report
