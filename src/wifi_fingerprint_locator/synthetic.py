"""Synthetic fingerprint surveys with the on-disk format of real captures.

No measured data ships with the project, so tests, examples and the
``synthesize`` command build a survey from a small physical model and write it
exactly as a real one is stored: one beamforming-report capture file per
reference point plus a ``dataset.csv``.

Channel model
-------------
A transmitter with two antennas stands in a rectangular room; the receiver
with ``n_rx`` antennas (a linear array along x) is placed at each reference
point of a regular grid. The channel from transmit antenna :math:`t` to
receive antenna :math:`r` at subcarrier frequency :math:`f_k` is a sum of
propagation paths,

.. math::

    H_{t,r,k} = \\sum_p g_{t,r,p}\\,(1 + \\varepsilon_p)\\,
    e^{-j 2\\pi f_k L_{t,r,p} / c},
    \\qquad g = \\rho_p \\frac{\\lambda}{4\\pi L},

over the direct path, the four first-order wall reflections (image sources,
Allen & Berkley, 1979; reflectivity :math:`\\rho`) and paths via a few fixed
point scatterers. The scatterer gains fluctuate from packet to packet
(:math:`\\varepsilon_p`, people moving), and every packet gets a gain jitter
and additive noise. Received power falls with distance from the access point
and the multipath pattern changes with position, so the positions can be
learned, but neighboring reference points are not trivially similar.

Measurement impairments
-----------------------
Each reported packet carries the defects a fingerprinting pipeline must
handle: a random sampling-time offset (a phase slope across subcarriers), a
random common phase (carrier and PLL offset), a fixed random phase per receive
chain, 8-bit quantization with an arbitrary per-packet scale, RSSI, AGC and
noise-floor header values consistent with the true received power,
occasionally two transmit streams instead of one, and occasional entries
reported as exactly zero. Receive columns are reported in a permuted order
that the antenna-selection field undoes.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .beamform_report import (
    TIMESTAMP_PERIOD_US,
    ChannelSnapshot,
    RadioMeta,
    write_beamform_file,
)
from .config import SyntheticConfig
from .dataset import DATASET_INDEX_NAME, write_dataset_index
from .subcarriers import SUBCARRIER_SPACING_HZ, reported_subcarrier_indices

_LOGGER = logging.getLogger(__name__)

#: Speed of light in vacuum, m/s.
SPEED_OF_LIGHT_M_S: float = 299_792_458.0

_RSSI_OFFSET_DB = 44.0
_AGC_RANGE_DB = (38, 46)
_NOISE_RANGE_DBM = (-95, -85)
_REPORT_PEAK_RANGE = (25.0, 55.0)
_HT_FLAG = 0x0100
_SHORT_GUARD_FLAG = 0x2000
_N_TRANSMIT_ANTENNAS = 2


@dataclass(frozen=True, eq=False)
class SyntheticRoom:
    """Geometry of a synthetic survey.

    Attributes
    ----------
    bounds_m : tuple of float
        ``(x_min, x_max, y_min, y_max)`` of the walls.
    reference_points_m : numpy.ndarray, shape (n_points, 2)
        Grid positions, x-major (all y values of the first x, then the next).
    transmit_antennas_m : numpy.ndarray, shape (2, 2)
        Positions of the two transmit antennas.
    scatterers_m : numpy.ndarray, shape (n_scatterers, 2)
        Positions of the point scatterers.
    scatterer_gains : numpy.ndarray, shape (n_scatterers,)
        Amplitude gains of the scatterers.

    Examples
    --------
    >>> room = make_room(SyntheticConfig(grid_shape=(2, 3)), rng=0)
    >>> room.reference_points_m.shape, room.bounds_m
    ((6, 2), (-1.5, 2.5, -1.5, 3.5))
    """

    bounds_m: tuple[float, float, float, float]
    reference_points_m: NDArray[np.float64]
    transmit_antennas_m: NDArray[np.float64]
    scatterers_m: NDArray[np.float64]
    scatterer_gains: NDArray[np.float64]


def reference_grid(grid_shape: Sequence[int], spacing_m: float) -> NDArray[np.float64]:
    """Positions of a rectangular grid of reference points, x-major.

    Examples
    --------
    >>> reference_grid((2, 2), 0.5).tolist()
    [[0.0, 0.0], [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]]
    """
    n_x, n_y = (int(n) for n in grid_shape)
    xs, ys = np.meshgrid(np.arange(n_x) * spacing_m, np.arange(n_y) * spacing_m, indexing="ij")
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)


def make_room(
    config: SyntheticConfig, rng: np.random.Generator | int | None = None
) -> SyntheticRoom:
    """Draw the fixed geometry of a synthetic survey.

    Parameters
    ----------
    config : SyntheticConfig
        Grid, room and transmitter settings.
    rng : numpy.random.Generator or int, optional
        Randomness of the scatterer positions and gains.

    Returns
    -------
    SyntheticRoom
        The geometry.
    """
    rng = np.random.default_rng(rng)
    points = reference_grid(config.grid_shape, config.grid_spacing_m)
    margin = config.room_margin_m
    bounds = (
        -margin,
        float(points[:, 0].max()) + margin,
        -margin,
        float(points[:, 1].max()) + margin,
    )
    access_point = np.asarray(config.access_point_m, dtype=np.float64)
    transmit = np.vstack([access_point, access_point + np.array([config.antenna_spacing_m, 0.0])])
    inset = 0.2
    scatterers = np.column_stack(
        [
            rng.uniform(bounds[0] + inset, bounds[1] - inset, config.n_scatterers),
            rng.uniform(bounds[2] + inset, bounds[3] - inset, config.n_scatterers),
        ]
    )
    gains = rng.uniform(0.3, 0.8, config.n_scatterers)
    return SyntheticRoom(bounds, points, transmit, scatterers, gains)


def path_geometry(
    room: SyntheticRoom, receive_antennas_m: ArrayLike, wall_reflectivity: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Length and free-space amplitude of every propagation path.

    The first-order wall reflections are found with the image-source method:
    each wall mirrors the transmitter, and the reflected path is the straight
    line from that mirror image to the receiver.

    Parameters
    ----------
    room : SyntheticRoom
        Geometry.
    receive_antennas_m : array_like, shape (n_rx, 2)
        Receive antenna positions.
    wall_reflectivity : float
        Amplitude reflection coefficient of the walls.

    Returns
    -------
    lengths, gains : numpy.ndarray, shape (2, n_rx, 5 + n_scatterers)
        Path lengths in meters and amplitudes, paths ordered as direct, the
        walls ``x_min``, ``x_max``, ``y_min``, ``y_max``, then the scatterers.

    References
    ----------
    .. [1] J. B. Allen and D. A. Berkley, "Image method for efficiently simulating
       small-room acoustics," The Journal of the Acoustical Society of America,
       vol. 65, no. 4, pp. 943-950, 1979.

    Examples
    --------
    >>> room = make_room(SyntheticConfig(n_scatterers=2), rng=0)
    >>> lengths, gains = path_geometry(room, [[0.0, 0.0]], 0.5)
    >>> lengths.shape, round(float(lengths[0, 0, 0]), 4)
    ((2, 1, 7), 1.4142)
    """
    receivers = np.asarray(receive_antennas_m, dtype=np.float64).reshape(-1, 2)
    transmit = room.transmit_antennas_m
    x_min, x_max, y_min, y_max = room.bounds_m
    images = np.stack(
        [
            np.column_stack([2 * x_min - transmit[:, 0], transmit[:, 1]]),
            np.column_stack([2 * x_max - transmit[:, 0], transmit[:, 1]]),
            np.column_stack([transmit[:, 0], 2 * y_min - transmit[:, 1]]),
            np.column_stack([transmit[:, 0], 2 * y_max - transmit[:, 1]]),
        ],
        axis=1,
    )  # (n_tx, 4, 2)
    direct = np.linalg.norm(transmit[:, None, :] - receivers[None, :, :], axis=-1)
    walls = np.linalg.norm(images[:, None, :, :] - receivers[None, :, None, :], axis=-1)
    to_scatterer = np.linalg.norm(transmit[:, None, :] - room.scatterers_m[None, :, :], axis=-1)
    from_scatterer = np.linalg.norm(room.scatterers_m[None, :, :] - receivers[:, None, :], axis=-1)
    via_scatterer = to_scatterer[:, None, :] + from_scatterer[None, :, :]
    lengths = np.concatenate([direct[..., None], walls, via_scatterer], axis=-1)
    weights = np.concatenate([[1.0], np.full(4, float(wall_reflectivity)), room.scatterer_gains])
    return lengths, weights / (4.0 * np.pi * np.maximum(lengths, 0.05))


def make_capture(
    path: str | Path,
    room: SyntheticRoom,
    position_m: ArrayLike,
    config: SyntheticConfig,
    *,
    n_packets: int | None = None,
    rng: np.random.Generator | int | None = None,
) -> list[ChannelSnapshot]:
    """Simulate one capture at a reference point and write it as a capture file.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination capture file.
    room : SyntheticRoom
        Geometry from :func:`make_room`.
    position_m : array_like, shape (2,)
        Receiver position.
    config : SyntheticConfig
        Channel and impairment settings.
    n_packets : int, optional
        Packets to record; by default ``config.n_packets``.
    rng : numpy.random.Generator or int, optional
        Randomness.

    Returns
    -------
    list of ChannelSnapshot
        The written reports, CSI in physical antenna order.

    Examples
    --------
    >>> import tempfile
    >>> config = SyntheticConfig()
    >>> room = make_room(config, rng=0)
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     reports = make_capture(Path(folder) / "a.dat", room, (2.0, 1.0), config,
    ...                            n_packets=5, rng=1)
    >>> len(reports), reports[0].csi.shape[1:]
    (5, (3, 30))
    """
    rng = np.random.default_rng(rng)
    count = int(config.n_packets if n_packets is None else n_packets)
    n_rx = int(config.n_rx)
    position = np.asarray(position_m, dtype=np.float64).reshape(2)
    offsets = (np.arange(n_rx) - (n_rx - 1) / 2.0) * config.antenna_spacing_m
    receivers = position + np.column_stack([offsets, np.zeros(n_rx)])
    lengths, gains = path_geometry(room, receivers, config.wall_reflectivity)
    indices = reported_subcarrier_indices(20).astype(np.float64)
    frequencies = config.center_frequency_hz + indices * SUBCARRIER_SPACING_HZ
    propagation = np.exp(
        -2j * np.pi * frequencies[None, None, None, :] * lengths[..., None] / SPEED_OF_LIGHT_M_S
    )  # (n_tx, n_rx, n_paths, n_sub)

    n_paths = lengths.shape[-1]
    is_scatterer = np.arange(n_paths) >= 5
    fluctuation = (
        rng.normal(size=(count, n_paths)) + 1j * rng.normal(size=(count, n_paths))
    ) / np.sqrt(2.0)
    path_weights = 1.0 + config.scatterer_jitter * fluctuation * is_scatterer
    channel = np.einsum("np,trp,trpk->ntrk", path_weights, gains, propagation)
    channel *= 10.0 ** (rng.normal(0.0, config.gain_jitter_db, count) / 20.0)[:, None, None, None]
    signal_power = np.mean(np.abs(channel) ** 2, axis=(1, 2, 3))
    noise_scale = np.sqrt(signal_power / 10.0 ** (config.snr_db / 10.0) / 2.0)
    channel = channel + noise_scale[:, None, None, None] * (
        rng.normal(size=channel.shape) + 1j * rng.normal(size=channel.shape)
    )

    n_tx = np.where(rng.random(count) < config.two_stream_fraction, 2, 1)
    agc = rng.integers(_AGC_RANGE_DB[0], _AGC_RANGE_DB[1], count)
    stream_mask = (np.arange(_N_TRANSMIT_ANTENNAS)[None, :] < n_tx[:, None]).astype(np.float64)
    chain_power = (
        np.einsum("nt,ntr->nr", stream_mask, np.mean(np.abs(channel) ** 2, axis=-1)) / n_tx[:, None]
    )
    chain_dbm = config.tx_power_dbm + 10.0 * np.log10(np.maximum(chain_power, 1e-30))
    rssi = np.clip(np.rint(chain_dbm + _RSSI_OFFSET_DB + agc[:, None]), 1, 255).astype(int)

    timing_offset = rng.uniform(-1.0, 1.0, count) * config.max_timing_offset_ns * 1e-9
    common_phase = rng.uniform(0.0, 2.0 * np.pi, count)
    chain_phase = rng.uniform(0.0, 2.0 * np.pi, n_rx)
    impairment = np.exp(
        1j
        * (
            common_phase[:, None, None]
            + chain_phase[None, :, None]
            - 2.0
            * np.pi
            * indices[None, None, :]
            * SUBCARRIER_SPACING_HZ
            * timing_offset[:, None, None]
        )
    )  # (n_packets, n_rx, n_sub)
    reported = channel * impairment[:, None, :, :]

    noise_base = int(rng.integers(_NOISE_RANGE_DBM[0], _NOISE_RANGE_DBM[1] + 1))
    noise = noise_base + rng.integers(-1, 2, count)
    peak = rng.uniform(*_REPORT_PEAK_RANGE, count)
    mcs = np.where(n_tx == 1, rng.integers(0, 8, count), rng.integers(8, 16, count))
    short_guard = rng.random(count) < 0.5
    zero_entries = rng.random(reported.shape) < config.zero_entry_rate
    permutation = tuple(int(a) for a in rng.permutation(n_rx))
    steps = config.packet_interval_s * 1e6 * (1.0 + 0.02 * rng.standard_normal(count))
    start = int(rng.integers(0, TIMESTAMP_PERIOD_US))
    timestamps = (
        start + np.rint(np.cumsum(steps) - steps[0]).astype(np.int64)
    ) % TIMESTAMP_PERIOD_US

    snapshots = []
    for k in range(count):
        streams = reported[k, : n_tx[k]]
        scale = peak[k] / max(np.abs(streams.real).max(), np.abs(streams.imag).max(), 1e-30)
        quantized = np.clip(np.rint(streams.real * scale), -128, 127) + 1j * np.clip(
            np.rint(streams.imag * scale), -128, 127
        )
        quantized[zero_entries[k, : n_tx[k]]] = 0.0
        meta = RadioMeta(
            report_counter=k & 0xFFFF,
            n_tx=int(n_tx[k]),
            n_rx=n_rx,
            rssi_db=tuple(int(v) for v in np.pad(rssi[k], (0, 3 - n_rx))),  # type: ignore[arg-type]
            noise_dbm=int(noise[k]),
            agc_db=int(agc[k]),
            antenna_permutation=permutation,
            permutation_valid=True,
            rate_flags=int(_HT_FLAG | mcs[k] | (_SHORT_GUARD_FLAG if short_guard[k] else 0)),
        )
        snapshots.append(ChannelSnapshot(quantized, int(timestamps[k]), meta))
    write_beamform_file(path, snapshots)
    return snapshots


def write_synthetic_dataset(
    folder: str | Path,
    config: SyntheticConfig | None = None,
    *,
    rng: np.random.Generator | int | None = None,
) -> Path:
    """Write a synthetic survey: capture files and a dataset index.

    Parameters
    ----------
    folder : str or pathlib.Path
        Destination; created if needed. Captures go to
        ``items/item_<NNN>.dat``, the index to ``dataset.csv``.
    config : SyntheticConfig, optional
        Survey settings; by default ``SyntheticConfig()``.
    rng : numpy.random.Generator or int, optional
        Randomness; the same value always gives the same bytes.

    Returns
    -------
    pathlib.Path
        The written ``dataset.csv``.

    Examples
    --------
    >>> import tempfile
    >>> config = SyntheticConfig(grid_shape=(2, 2), n_packets=3)
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     index = write_synthetic_dataset(folder, config, rng=0)
    ...     rows = pd.read_csv(index)
    >>> rows.columns.tolist(), rows["path"].tolist()[:2]
    (['path', 'x_m', 'y_m', 'group', 'campaign'], ['items/item_000.dat', 'items/item_001.dat'])
    """
    settings = SyntheticConfig() if config is None else config
    rng = np.random.default_rng(rng)
    destination = Path(folder)
    destination.mkdir(parents=True, exist_ok=True)
    room = make_room(settings, rng)
    rows = []
    counter = 0
    for campaign in range(settings.n_campaigns):
        for point_index, point in enumerate(room.reference_points_m):
            relative = f"items/item_{counter:03d}.dat"
            make_capture(destination / relative, room, point, settings, rng=rng)
            rows.append(
                {
                    "path": relative,
                    "x_m": float(point[0]),
                    "y_m": float(point[1]),
                    "group": f"point_{point_index:03d}",
                    "campaign": f"campaign_{campaign}",
                }
            )
            counter += 1
    index = write_dataset_index(destination / DATASET_INDEX_NAME, pd.DataFrame(rows))
    _LOGGER.info(
        "synthetic survey: %d capture(s) of %d packet(s) written to %s",
        counter,
        settings.n_packets,
        destination.as_posix(),
    )
    return index
