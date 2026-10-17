"""Reader and writer for 802.11n beamforming-report capture files.

File format
-----------
A capture is a flat stream of frames. Each frame starts with a big-endian
``uint16`` that counts the bytes that follow it (one code byte plus the
payload), then the code byte. Code ``0xBB`` marks a beamforming report; other
codes are skipped. A beamforming payload is a 20-byte little-endian header

=======  =====  ========================================================
offset   type   field
=======  =====  ========================================================
0        u32    NIC timestamp, microseconds, wraps at 2**32
4        u16    report counter
6        --     two reserved bytes
8        u8     number of receive chains ``n_rx``
9        u8     number of transmit chains ``n_tx``
10-12    u8     RSSI of receive chains a, b, c (dB relative to AGC)
13       i8     noise floor, dBm (-127 when unknown)
14       u8     AGC setting, dB
15       u8     antenna selection (three 2-bit fields)
16       u16    length of the packed CSI block in bytes
18       u16    rate and flags (MCS, bandwidth, guard interval)
=======  =====  ========================================================

followed by the bit-packed CSI. For each of the 30 subcarrier groups the
packer skips three bits, then writes ``n_tx * n_rx`` complex entries as an
8-bit signed real part followed by an 8-bit signed imaginary part, LSB first.
Entries are ordered with the transmit chain varying fastest. Because of the
three-bit skips, most values straddle a byte boundary: the value starting at
bit ``b`` is ``(p[b // 8] >> b % 8) | (p[b // 8 + 1] << (8 - b % 8))``.

The format is the one produced by the Linux 802.11n CSI Tool (Halperin et al.,
2011). Decoding is vectorized: the bit offsets of every I/Q value are computed
once per ``(n_tx, n_rx)`` and a whole block is decoded with a few array
operations instead of a per-bit loop.
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .errors import BeamformFormatError
from .subcarriers import N_SUBCARRIER_GROUPS

_LOGGER = logging.getLogger(__name__)

#: Frame code of a beamforming report.
BEAMFORM_REPORT_CODE: int = 0xBB
#: Width of the NIC timestamp counter, in microseconds.
TIMESTAMP_PERIOD_US: int = 2**32
#: Noise value the NIC reports when the noise floor is unknown.
UNKNOWN_NOISE_DBM: int = -127
#: Bit of the rate-and-flags word that marks a 40 MHz (HT40) frame.
HT40_FLAG: int = 0x0800

_HEADER = struct.Struct("<IH2xBBBBBbBBHH")
_FRAME_PREFIX = struct.Struct(">HB")

#: Size of the fixed payload header in bytes.
HEADER_SIZE: int = _HEADER.size

#: Bits skipped before each subcarrier group in the packed CSI block.
_GROUP_PAD_BITS = 3
#: Bits per complex entry (8-bit real + 8-bit imaginary).
_ENTRY_BITS = 16


@dataclass(frozen=True)
class RadioMeta:
    """Radio-level metadata of one beamforming report.

    Attributes
    ----------
    report_counter : int
        Running report counter maintained by the driver.
    n_tx, n_rx : int
        Number of transmit and receive chains in the CSI tensor.
    rssi_db : tuple of int
        Per-receive-chain RSSI ``(a, b, c)`` in dB relative to the AGC
        setting; 0 marks an absent chain.
    noise_dbm : int
        Noise floor reported by the NIC, in dBm; -127 when unknown.
    agc_db : int
        Automatic gain control setting, in dB.
    antenna_permutation : tuple of int
        For each reported receive column ``i``, the 0-based physical antenna
        it came from.
    permutation_valid : bool
        False when the antenna-selection field is not a permutation; the
        tensor is then left in reported order.
    rate_flags : int
        The 16-bit rate-and-flags word (MCS, bandwidth, guard interval).

    Examples
    --------
    >>> meta = RadioMeta(0, 1, 3, (40, 38, 42), -92, 40, (0, 1, 2), True, 0x0901)
    >>> meta.bandwidth_mhz
    40
    """

    report_counter: int
    n_tx: int
    n_rx: int
    rssi_db: tuple[int, int, int]
    noise_dbm: int
    agc_db: int
    antenna_permutation: tuple[int, ...]
    permutation_valid: bool
    rate_flags: int

    @property
    def bandwidth_mhz(self) -> int:
        """Channel bandwidth of the frame, 20 or 40 MHz."""
        return 40 if self.rate_flags & HT40_FLAG else 20


@dataclass(frozen=True, eq=False)
class ChannelSnapshot:
    """One beamforming report: a CSI tensor, its timestamp and its metadata.

    A report with one transmit chain is ``(1, n_rx, 30)``, never squeezed to
    ``(n_rx, 30)``, so the rank of the tensor never depends on the packet.

    Attributes
    ----------
    csi : numpy.ndarray, shape (n_tx, n_rx, 30)
        Read-only ``complex128`` copy of the channel, in physical antenna order.
    timestamp_us : int
        Raw 32-bit NIC timestamp in microseconds; see :func:`unwrap_timestamps`.
    meta : RadioMeta
        Radio metadata.

    Raises
    ------
    ValueError
        If the tensor shape disagrees with ``meta`` or the timestamp is out of
        the 32-bit range.

    Examples
    --------
    >>> meta = RadioMeta(7, 1, 3, (40, 38, 42), -92, 40, (0, 1, 2), True, 0x0100)
    >>> snapshot = ChannelSnapshot(np.ones((1, 3, 30)), 1_000, meta)
    >>> snapshot.csi.shape, snapshot.csi.dtype.name, snapshot.csi.flags.writeable
    ((1, 3, 30), 'complex128', False)
    """

    csi: NDArray[np.complex128]
    timestamp_us: int
    meta: RadioMeta

    def __post_init__(self) -> None:
        csi = np.array(self.csi, dtype=np.complex128, copy=True)
        if csi.ndim != 3:
            raise ValueError(
                f"CSI must be shaped (n_tx, n_rx, n_subcarriers); got {csi.ndim} dimensions"
            )
        if csi.shape[:2] != (self.meta.n_tx, self.meta.n_rx):
            raise ValueError(
                f"CSI shape {csi.shape} disagrees with n_tx={self.meta.n_tx}, n_rx={self.meta.n_rx}"
            )
        if not 0 <= int(self.timestamp_us) < TIMESTAMP_PERIOD_US:
            raise ValueError(f"timestamp {self.timestamp_us} is outside the 32-bit counter range")
        csi.setflags(write=False)
        object.__setattr__(self, "csi", csi)
        object.__setattr__(self, "timestamp_us", int(self.timestamp_us))


@dataclass
class ReadStatistics:
    """Counters collected while reading one capture file.

    Attributes
    ----------
    n_frames : int
        Frames encountered, whatever their code.
    n_reports : int
        Beamforming reports successfully decoded.
    n_other_codes : int
        Frames with a code other than ``0xBB``, skipped.
    n_truncated : int
        Frames cut off by the end of the file.
    n_malformed : int
        Reports whose CSI length disagrees with ``n_tx``/``n_rx`` or whose
        payload is too short, skipped.
    n_invalid_permutation : int
        Reports whose antenna selection is not a permutation; kept in
        reported order.

    Examples
    --------
    >>> ReadStatistics(n_frames=2, n_reports=2).as_dict()["n_reports"]
    2
    """

    n_frames: int = 0
    n_reports: int = 0
    n_other_codes: int = 0
    n_truncated: int = 0
    n_malformed: int = 0
    n_invalid_permutation: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return the counters as a plain dictionary."""
        return asdict(self)


def expected_csi_length(n_tx: int, n_rx: int) -> int:
    """Number of bytes the packed CSI block must occupy.

    Parameters
    ----------
    n_tx, n_rx : int
        Number of transmit and receive chains.

    Returns
    -------
    int
        ``ceil(30 * (16 * n_tx * n_rx + 3) / 8)``.

    Examples
    --------
    >>> expected_csi_length(2, 3), expected_csi_length(1, 3)
    (372, 192)
    """
    bits = N_SUBCARRIER_GROUPS * (_ENTRY_BITS * n_tx * n_rx + _GROUP_PAD_BITS)
    return (bits + 7) // 8


@lru_cache(maxsize=32)
def _iq_bit_offsets(n_entries: int) -> NDArray[np.int64]:
    """Start bit of every real and imaginary byte, shaped ``(30, n_entries, 2)``."""
    group = np.arange(N_SUBCARRIER_GROUPS, dtype=np.int64)[:, None]
    entry = np.arange(n_entries, dtype=np.int64)[None, :]
    real_start = _GROUP_PAD_BITS * (group + 1) + _ENTRY_BITS * (group * n_entries + entry)
    offsets = np.stack([real_start, real_start + 8], axis=-1)
    offsets.setflags(write=False)
    return offsets


def unpack_csi(block: bytes | NDArray[np.uint8], n_tx: int, n_rx: int) -> NDArray[np.complex128]:
    """Decode a packed CSI block into a complex tensor in reported order.

    Parameters
    ----------
    block : bytes or numpy.ndarray
        The packed CSI bytes (at least :func:`expected_csi_length` long).
    n_tx, n_rx : int
        Number of transmit and receive chains.

    Returns
    -------
    numpy.ndarray, shape (n_tx, n_rx, 30)
        Complex values whose real and imaginary parts are integers in
        ``[-128, 127]``.

    Raises
    ------
    BeamformFormatError
        If the block is shorter than the layout requires.

    References
    ----------
    .. [1] D. Halperin, W. Hu, A. Sheth, and D. Wetherall, "Tool release:
       Gathering 802.11n traces with channel state information," ACM SIGCOMM
       Computer Communication Review, vol. 41, no. 1, p. 53, 2011.

    Examples
    --------
    >>> csi = np.arange(-45, 45).reshape(1, 3, 30) * (1 - 1j)
    >>> bool(np.array_equal(unpack_csi(pack_csi(csi), 1, 3), csi))
    True
    """
    if n_tx < 1 or n_rx < 1:
        raise BeamformFormatError(f"n_tx and n_rx must be positive, got {n_tx}, {n_rx}")
    data = np.frombuffer(bytes(block), dtype=np.uint8)
    needed = expected_csi_length(n_tx, n_rx)
    if data.size < needed:
        raise BeamformFormatError(f"CSI block holds {data.size} bytes, layout needs {needed}")

    offsets = _iq_bit_offsets(n_tx * n_rx)
    byte_index = offsets >> 3
    shift = (offsets & 7).astype(np.uint16)
    # One zero byte of padding so the "next byte" lookup never runs off the end.
    padded = np.concatenate([data[:needed], np.zeros(1, dtype=np.uint8)]).astype(np.uint16)
    combined = (padded[byte_index] >> shift) | (padded[byte_index + 1] << (8 - shift))
    values = (combined & 0xFF).astype(np.uint8).view(np.int8).astype(np.float64)
    iq = values[..., 0] + 1j * values[..., 1]  # (30, n_entries), tx varies fastest
    return iq.reshape(N_SUBCARRIER_GROUPS, n_rx, n_tx).transpose(2, 1, 0).copy()


def pack_csi(csi: ArrayLike) -> bytes:
    """Encode a complex tensor into a packed CSI block; inverse of :func:`unpack_csi`.

    Parameters
    ----------
    csi : array_like, shape (n_tx, n_rx, 30)
        Complex values in reported order whose real and imaginary parts are
        integers in ``[-128, 127]``.

    Returns
    -------
    bytes
        The packed block, :func:`expected_csi_length` bytes long.

    Raises
    ------
    BeamformFormatError
        If the shape is wrong or a component is not an 8-bit signed integer.

    Examples
    --------
    >>> len(pack_csi(np.zeros((2, 3, 30))))
    372
    """
    tensor = np.asarray(csi)
    if tensor.ndim != 3 or tensor.shape[2] != N_SUBCARRIER_GROUPS:
        raise BeamformFormatError(f"CSI must be shaped (n_tx, n_rx, 30), got {tensor.shape}")
    n_tx, n_rx = tensor.shape[:2]
    iq = tensor.transpose(2, 1, 0).reshape(N_SUBCARRIER_GROUPS, n_rx * n_tx)
    parts = np.stack([np.real(iq), np.imag(iq)], axis=-1).astype(np.float64)
    rounded = np.rint(parts)
    if not np.allclose(parts, rounded) or rounded.min() < -128 or rounded.max() > 127:
        raise BeamformFormatError(
            "CSI components must be integers in [-128, 127] to fit the 8-bit packed format"
        )
    raw = rounded.astype(np.int8).view(np.uint8)  # (30, n_entries, 2)

    offsets = _iq_bit_offsets(n_tx * n_rx)
    n_bytes = expected_csi_length(n_tx, n_rx)
    bit_positions = offsets[..., None] + np.arange(8)
    bit_values = (raw[..., None] >> np.arange(8, dtype=np.uint8)) & 1
    bits = np.zeros(n_bytes * 8, dtype=np.uint8)
    bits[bit_positions.ravel()] = bit_values.ravel()
    return np.packbits(bits, bitorder="little").tobytes()


def decode_antenna_permutation(antenna_selection: int, n_rx: int) -> tuple[tuple[int, ...], bool]:
    """Decode the antenna-selection byte into a 0-based permutation.

    Parameters
    ----------
    antenna_selection : int
        The byte from the report header: three 2-bit fields, one per
        reported receive column.
    n_rx : int
        Number of receive chains actually reported.

    Returns
    -------
    permutation : tuple of int
        For each reported column, the 0-based physical antenna index.
    valid : bool
        True when the first ``n_rx`` fields form a permutation of
        ``range(n_rx)``.

    Examples
    --------
    >>> decode_antenna_permutation(0b100100, 3)
    ((0, 1, 2), True)
    >>> decode_antenna_permutation(0b000110, 3)
    ((2, 1, 0), True)
    >>> decode_antenna_permutation(0b000000, 3)
    ((0, 0, 0), False)
    """
    fields = tuple(int((antenna_selection >> (2 * i)) & 0b11) for i in range(n_rx))
    valid = sorted(fields) == list(range(n_rx))
    return fields, valid


def encode_antenna_selection(permutation: Iterable[int]) -> int:
    """Pack a 0-based permutation into the antenna-selection byte.

    Examples
    --------
    >>> encode_antenna_selection((2, 0, 1))
    18
    >>> decode_antenna_permutation(18, 3)
    ((2, 0, 1), True)
    """
    value = 0
    for i, antenna in enumerate(permutation):
        if not 0 <= int(antenna) <= 3:
            raise ValueError(f"antenna index must be in 0..3, got {antenna}")
        value |= int(antenna) << (2 * i)
    return value


def parse_beamform_payload(payload: bytes, *, apply_permutation: bool = True) -> ChannelSnapshot:
    """Decode one beamforming-report payload (the bytes after the code byte).

    Parameters
    ----------
    payload : bytes
        Header plus packed CSI.
    apply_permutation : bool, optional
        Reorder receive columns into physical antenna order. The receive chain
        behind "column 1" can change from packet to packet, so without it the
        antenna axis means something different in different rows of a
        fingerprint matrix.

    Returns
    -------
    ChannelSnapshot
        The decoded report.

    Raises
    ------
    BeamformFormatError
        If the payload is too short or its CSI length disagrees with the
        number of chains.

    Examples
    --------
    >>> meta = RadioMeta(3, 1, 3, (30, 31, 32), -90, 41, (2, 0, 1), True, 0x0105)
    >>> csi = np.arange(90).reshape(1, 3, 30) % 50 - 25 + 0j
    >>> decoded = parse_beamform_payload(encode_beamform_payload(ChannelSnapshot(csi, 5, meta)))
    >>> bool(np.array_equal(decoded.csi, csi)), decoded.meta == meta
    (True, True)
    """
    if len(payload) < HEADER_SIZE:
        raise BeamformFormatError(
            f"payload of {len(payload)} bytes is shorter than the {HEADER_SIZE}-byte header"
        )
    fields = _HEADER.unpack_from(payload)
    timestamp_us, report_counter, n_rx, n_tx = fields[:4]
    rssi_db = tuple(int(value) for value in fields[4:7])
    noise_dbm, agc_db, antenna_selection, csi_length, rate_flags = fields[7:]

    if not (1 <= n_tx <= 3 and 1 <= n_rx <= 3):
        raise BeamformFormatError(f"implausible chain counts n_tx={n_tx}, n_rx={n_rx}")
    expected = expected_csi_length(n_tx, n_rx)
    if csi_length != expected:
        raise BeamformFormatError(
            f"CSI length field is {csi_length} bytes but n_tx={n_tx}, n_rx={n_rx} "
            f"requires {expected}"
        )
    block = payload[HEADER_SIZE : HEADER_SIZE + csi_length]
    if len(block) < csi_length:
        raise BeamformFormatError(
            f"payload carries {len(block)} CSI bytes, header announces {csi_length}"
        )

    csi = unpack_csi(block, n_tx, n_rx)
    permutation, valid = decode_antenna_permutation(antenna_selection, n_rx)
    if apply_permutation and valid and n_rx > 1:
        # Reported column i came from physical antenna permutation[i].
        csi = csi[:, np.argsort(permutation), :]

    meta = RadioMeta(
        report_counter=report_counter,
        n_tx=n_tx,
        n_rx=n_rx,
        rssi_db=rssi_db,  # type: ignore[arg-type]
        noise_dbm=noise_dbm,
        agc_db=agc_db,
        antenna_permutation=permutation,
        permutation_valid=valid,
        rate_flags=rate_flags,
    )
    return ChannelSnapshot(csi=csi, timestamp_us=timestamp_us, meta=meta)


def encode_beamform_payload(snapshot: ChannelSnapshot) -> bytes:
    """Encode a snapshot into a beamforming-report payload.

    The snapshot's CSI is taken to be in physical antenna order (as produced
    by :func:`parse_beamform_payload` with ``apply_permutation=True``) and is
    reordered into reporting order according to its antenna permutation.

    Parameters
    ----------
    snapshot : ChannelSnapshot
        A report whose CSI components are integers in ``[-128, 127]``.

    Returns
    -------
    bytes
        Header plus packed CSI.

    Examples
    --------
    >>> meta = RadioMeta(0, 2, 3, (30, 31, 32), -90, 41, (0, 1, 2), True, 0x010A)
    >>> len(encode_beamform_payload(ChannelSnapshot(np.zeros((2, 3, 30)), 0, meta)))
    392
    """
    meta = snapshot.meta
    csi = snapshot.csi
    if meta.permutation_valid and meta.n_rx > 1:
        csi = csi[:, list(meta.antenna_permutation), :]
    block = pack_csi(csi)
    header = _HEADER.pack(
        snapshot.timestamp_us,
        meta.report_counter & 0xFFFF,
        meta.n_rx,
        meta.n_tx,
        *meta.rssi_db,
        meta.noise_dbm,
        meta.agc_db,
        encode_antenna_selection(meta.antenna_permutation),
        len(block),
        meta.rate_flags,
    )
    return header + block


def frame_bytes(code: int, payload: bytes) -> bytes:
    """Wrap a payload in the length-prefixed frame format.

    Examples
    --------
    >>> frame_bytes(0xC1, b"abc").hex()
    '0004c1616263'
    """
    if len(payload) + 1 > 0xFFFF:
        raise BeamformFormatError(f"payload of {len(payload)} bytes does not fit in one frame")
    return _FRAME_PREFIX.pack(len(payload) + 1, code) + payload


class BeamformReportReader:
    """Stream beamforming reports out of a capture file.

    Reports are decoded one frame at a time, so a long capture is never held
    in memory as a whole. In the default lenient mode malformed reports are
    logged and skipped instead of aborting the read; :attr:`statistics`
    counts every kind of problem. In strict mode the first malformed or
    truncated frame raises :class:`~wifi_fingerprint_locator.errors.BeamformFormatError`.

    Parameters
    ----------
    path : str or pathlib.Path
        Capture file.
    apply_permutation : bool, optional
        Reorder receive columns into physical antenna order (default True).
    strict : bool, optional
        Raise on the first malformed or truncated frame (default False).

    Raises
    ------
    FileNotFoundError
        If the capture file does not exist.

    Examples
    --------
    >>> import tempfile
    >>> meta = RadioMeta(0, 1, 3, (30, 31, 32), -90, 41, (0, 1, 2), True, 0x0100)
    >>> reports = [ChannelSnapshot(np.full((1, 3, 30), k + 0j), k, meta) for k in range(3)]
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "capture.dat"
    ...     _ = write_beamform_file(path, reports)
    ...     reader = BeamformReportReader(path)
    ...     n_read = sum(1 for _ in reader.iter_reports())
    >>> n_read, reader.statistics.n_reports
    (3, 3)
    """

    def __init__(
        self, path: str | Path, *, apply_permutation: bool = True, strict: bool = False
    ) -> None:
        self.path = Path(path)
        self.apply_permutation = apply_permutation
        self.strict = strict
        self.statistics = ReadStatistics()
        if not self.path.is_file():
            raise FileNotFoundError(f"capture file not found: {self.path}")

    def iter_reports(self) -> Iterator[ChannelSnapshot]:
        """Yield every decodable beamforming report in file order.

        Statistics are reset at the start of each iteration.
        """
        self.statistics = ReadStatistics()
        with self.path.open("rb") as stream:
            yield from self._iter_stream(stream)
        stats = self.statistics
        if stats.n_truncated or stats.n_malformed:
            _LOGGER.warning(
                "%s: skipped %d truncated and %d malformed report(s) out of %d frame(s)",
                self.path.name,
                stats.n_truncated,
                stats.n_malformed,
                stats.n_frames,
            )

    def _problem(self, message: str) -> None:
        if self.strict:
            raise BeamformFormatError(f"{self.path.name}: {message}")

    def _iter_stream(self, stream: BinaryIO) -> Iterator[ChannelSnapshot]:
        stats = self.statistics
        warned_permutation = False
        while True:
            prefix = stream.read(_FRAME_PREFIX.size)
            if not prefix:
                return
            if len(prefix) < _FRAME_PREFIX.size:
                stats.n_truncated += 1
                self._problem("the file ends inside a frame header")
                return
            frame_length, code = _FRAME_PREFIX.unpack(prefix)
            stats.n_frames += 1
            if frame_length < 1:
                stats.n_malformed += 1
                position = stream.tell() - _FRAME_PREFIX.size
                self._problem(f"zero-length frame header at byte {position}")
                _LOGGER.warning(
                    "%s: zero-length frame header at byte %d; stopping", self.path.name, position
                )
                return
            payload = stream.read(frame_length - 1)
            if len(payload) < frame_length - 1:
                stats.n_truncated += 1
                self._problem(f"frame {stats.n_frames} is cut off by the end of the file")
                return
            if code != BEAMFORM_REPORT_CODE:
                stats.n_other_codes += 1
                continue
            try:
                snapshot = parse_beamform_payload(payload, apply_permutation=self.apply_permutation)
            except BeamformFormatError as exc:
                stats.n_malformed += 1
                self._problem(f"frame {stats.n_frames}: {exc}")
                _LOGGER.warning("%s: frame %d skipped: %s", self.path.name, stats.n_frames, exc)
                continue
            if not snapshot.meta.permutation_valid:
                stats.n_invalid_permutation += 1
                if not warned_permutation:
                    warned_permutation = True
                    _LOGGER.warning(
                        "%s: antenna selection %s is not a permutation; such reports are "
                        "kept in reported order",
                        self.path.name,
                        snapshot.meta.antenna_permutation,
                    )
            stats.n_reports += 1
            yield snapshot

    def read_all(self) -> list[ChannelSnapshot]:
        """Decode the whole file into a list."""
        return list(self.iter_reports())


def read_beamform_file(
    path: str | Path, *, apply_permutation: bool = True, strict: bool = False
) -> tuple[list[ChannelSnapshot], ReadStatistics]:
    """Read every beamforming report of a capture file.

    Parameters
    ----------
    path : str or pathlib.Path
        Capture file.
    apply_permutation : bool, optional
        Reorder receive columns into physical antenna order (default True).
    strict : bool, optional
        Raise on the first malformed or truncated frame instead of skipping it.

    Returns
    -------
    snapshots : list of ChannelSnapshot
        Decoded reports in file order.
    statistics : ReadStatistics
        What was read and what was skipped.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    BeamformFormatError
        In strict mode, for the first malformed or truncated frame.

    Examples
    --------
    >>> import tempfile
    >>> meta = RadioMeta(0, 1, 3, (30, 31, 32), -90, 41, (0, 1, 2), True, 0x0100)
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "capture.dat"
    ...     _ = write_beamform_file(path, [ChannelSnapshot(np.ones((1, 3, 30)), 9, meta)],
    ...                             extra_frames=[(0xC1, b"other")])
    ...     reports, statistics = read_beamform_file(path)
    >>> len(reports), statistics.n_other_codes
    (1, 1)
    """
    reader = BeamformReportReader(path, apply_permutation=apply_permutation, strict=strict)
    snapshots = reader.read_all()
    _LOGGER.debug(
        "%s: %d report(s) decoded (%s)",
        reader.path.name,
        len(snapshots),
        ", ".join(f"{k}={v}" for k, v in reader.statistics.as_dict().items() if v),
    )
    return snapshots, reader.statistics


def write_beamform_file(
    path: str | Path,
    snapshots: Iterable[ChannelSnapshot],
    *,
    extra_frames: Iterable[tuple[int, bytes]] = (),
) -> int:
    """Write snapshots as a capture file readable by :func:`read_beamform_file`.

    ``synthesize`` writes its captures with it, and a filtered subset of a
    real capture can be saved the same way.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination; parent directories are created.
    snapshots : iterable of ChannelSnapshot
        Reports with integer-valued CSI.
    extra_frames : iterable of (int, bytes), optional
        Additional ``(code, payload)`` frames appended after the reports,
        e.g. non-beamforming frames for testing.

    Returns
    -------
    int
        Number of bytes written.

    Examples
    --------
    >>> import tempfile
    >>> meta = RadioMeta(0, 1, 3, (30, 31, 32), -90, 41, (0, 1, 2), True, 0x0100)
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     report = ChannelSnapshot(np.ones((1, 3, 30)), 0, meta)
    ...     write_beamform_file(Path(folder) / "a.dat", [report])
    215
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    chunks = [frame_bytes(BEAMFORM_REPORT_CODE, encode_beamform_payload(s)) for s in snapshots]
    chunks.extend(frame_bytes(code, payload) for code, payload in extra_frames)
    data = b"".join(chunks)
    destination.write_bytes(data)
    return len(data)


def unwrap_timestamps(timestamps_us: Sequence[int] | ArrayLike) -> NDArray[np.int64]:
    """Undo the 32-bit wrap-around of NIC timestamps.

    Parameters
    ----------
    timestamps_us : sequence of int
        Raw timestamps in file order.

    Returns
    -------
    numpy.ndarray of int64
        Monotone timestamps in microseconds, starting at the first raw value.

    Examples
    --------
    >>> unwrap_timestamps([2**32 - 10, 5, 20]).tolist()
    [4294967286, 4294967301, 4294967316]
    """
    raw = np.asarray(timestamps_us, dtype=np.int64)
    if raw.ndim != 1:
        raise ValueError(f"timestamps must be one-dimensional, got shape {raw.shape}")
    if raw.size == 0:
        return raw
    steps = np.diff(raw)
    wraps = np.concatenate([[0], np.cumsum(steps < 0)])
    return raw + wraps * TIMESTAMP_PERIOD_US


def summarize_capture(
    snapshots: Sequence[ChannelSnapshot], statistics: ReadStatistics | None = None
) -> dict[str, float | int]:
    """Quality summary of one capture: counts, chain usage, radio levels and timing.

    Parameters
    ----------
    snapshots : sequence of ChannelSnapshot
        Decoded reports in file order.
    statistics : ReadStatistics, optional
        Reader counters to include.

    Returns
    -------
    dict
        ``n_reports``, ``n_single_stream``, ``n_multi_stream``, ``n_rx``,
        ``bandwidth_mhz`` (0 when mixed), ``rssi_mean_db``, ``noise_min_dbm``,
        ``noise_max_dbm``, ``agc_min_db``, ``agc_max_db``, ``duration_s``,
        ``median_interval_s``, ``n_zero_entries``, then the reader counters.

    Examples
    --------
    >>> meta = RadioMeta(0, 1, 3, (30, 31, 32), -90, 41, (0, 1, 2), True, 0x0100)
    >>> reports = [ChannelSnapshot(np.ones((1, 3, 30)), k * 1_000_000, meta) for k in range(4)]
    >>> summary = summarize_capture(reports)
    >>> summary["n_reports"], summary["duration_s"], summary["median_interval_s"]
    (4, 3.0, 1.0)
    """
    n_tx = np.array([s.meta.n_tx for s in snapshots], dtype=np.int64)
    bandwidths = sorted({s.meta.bandwidth_mhz for s in snapshots})
    times = unwrap_timestamps([s.timestamp_us for s in snapshots])
    rssi = np.array([s.meta.rssi_db for s in snapshots], dtype=np.float64).reshape(-1, 3)
    present = rssi[rssi > 0]
    noise = np.array([s.meta.noise_dbm for s in snapshots], dtype=np.int64)
    agc = np.array([s.meta.agc_db for s in snapshots], dtype=np.int64)
    summary: dict[str, float | int] = {
        "n_reports": len(snapshots),
        "n_single_stream": int(np.sum(n_tx == 1)),
        "n_multi_stream": int(np.sum(n_tx > 1)),
        "n_rx": int(max((s.meta.n_rx for s in snapshots), default=0)),
        "bandwidth_mhz": bandwidths[0] if len(bandwidths) == 1 else 0,
        "rssi_mean_db": float(present.mean()) if present.size else float("nan"),
        "noise_min_dbm": int(noise.min()) if noise.size else 0,
        "noise_max_dbm": int(noise.max()) if noise.size else 0,
        "agc_min_db": int(agc.min()) if agc.size else 0,
        "agc_max_db": int(agc.max()) if agc.size else 0,
        "duration_s": float(times[-1] - times[0]) / 1e6 if times.size else 0.0,
        "median_interval_s": float(np.median(np.diff(times))) / 1e6 if times.size > 1 else 0.0,
        "n_zero_entries": int(sum(int(np.sum(s.csi == 0)) for s in snapshots)),
    }
    if statistics is not None:
        summary.update(statistics.as_dict())
    return summary
