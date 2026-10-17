"""Assembly of a site survey: capture files -> cleaned channel stack -> fingerprints.

A fingerprint survey visits a set of reference points and records a capture
file at each. :func:`read_survey` reads every capture listed in the dataset
index and, for each file,

1. decodes the beamforming reports (:mod:`~wifi_fingerprint_locator.beamform_report`),
2. keeps the packet window ``survey.packet_start`` ... ``+ survey.n_packets``,
3. rescales each report to :math:`\\sqrt{\\mathrm{SNR}}` units
   (:mod:`~wifi_fingerprint_locator.csi_scaling`),
4. selects one transmit stream and the requested receive antennas,

then applies the cleaning policies to the whole survey at once
(:mod:`~wifi_fingerprint_locator.channel_cleaning`), so that the stream-count
majority is a survey-wide decision. The result, :class:`SurveyChannels`, holds
complex channels; :func:`survey_features` turns it into a
:class:`~wifi_fingerprint_locator.dataset.FeatureSet` for any representation,
so the comparison of representations reads the captures only once.

:func:`index_capture_folder` builds a dataset index from a folder of captures
whose file names encode grid indices.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .beamform_report import read_beamform_file
from .channel_cleaning import CleaningReport, clean_channels
from .csi_scaling import scale_csi
from .dataset import (
    TARGET_COLUMNS,
    FeatureSet,
    load_dataset_index,
    location_name,
    relative_paths,
)
from .errors import DatasetError
from .fingerprints import fingerprint_matrix
from .subcarriers import subcarrier_positions

if TYPE_CHECKING:
    from .config import Config, DataConfig, ScalingConfig, SurveyConfig

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class CaptureChannels:
    """The selected channels of one capture file, before cleaning.

    Attributes
    ----------
    path : str
        The capture file.
    csi : numpy.ndarray of complex128, shape (n_packets, n_rx, 30)
        One transmit stream, scaled.
    segments : numpy.ndarray of int64, shape (n_packets,)
        Position of each packet among the file's reports (0-based).
    n_tx : numpy.ndarray of int64, shape (n_packets,)
        Transmit streams each packet was reported with.
    bandwidth_mhz : int
        Channel bandwidth of the capture.

    Examples
    --------
    >>> capture = CaptureChannels("a.dat", np.ones((2, 3, 30), complex), np.arange(2),
    ...                           np.ones(2, dtype=np.int64), 20)
    >>> capture.n_packets
    2
    """

    path: str
    csi: NDArray[np.complex128]
    segments: NDArray[np.int64]
    n_tx: NDArray[np.int64]
    bandwidth_mhz: int

    @property
    def n_packets(self) -> int:
        """Number of packets."""
        return int(self.csi.shape[0])


@dataclass(frozen=True, eq=False)
class SurveyChannels:
    """Cleaned channels of a whole survey, one row per packet.

    Attributes
    ----------
    csi : numpy.ndarray of complex128, shape (n_items, n_rx, 30)
        Cleaned channels.
    coordinates : numpy.ndarray, shape (n_items, 2)
        Reference-point position of every packet, in meters.
    groups : numpy.ndarray of str
        Reference-point name of every packet.
    paths : numpy.ndarray of str
        Capture file of every packet, relative to the dataset index.
    segments : numpy.ndarray of int64
        Position of every packet in its capture file.
    bandwidth_mhz : int
        Channel bandwidth shared by all captures.
    report : CleaningReport
        What cleaning dropped and repaired.

    Examples
    --------
    >>> survey = SurveyChannels(np.ones((1, 3, 30), complex), np.zeros((1, 2)), np.array(["p"]),
    ...                         np.array(["a.dat"]), np.zeros(1, np.int64), 20,
    ...                         CleaningReport(1, 0, 0, 0, 0, 1, 1))
    >>> survey.n_items, survey.n_rx
    (1, 3)
    """

    csi: NDArray[np.complex128]
    coordinates: NDArray[np.float64]
    groups: NDArray[np.str_]
    paths: NDArray[np.str_]
    segments: NDArray[np.int64]
    bandwidth_mhz: int
    report: CleaningReport

    @property
    def n_items(self) -> int:
        """Number of packets."""
        return int(self.csi.shape[0])

    @property
    def n_rx(self) -> int:
        """Number of receive antennas."""
        return int(self.csi.shape[1])


def read_capture_channels(
    path: str | Path,
    survey: SurveyConfig,
    scaling: ScalingConfig,
    *,
    use_window: bool = True,
) -> CaptureChannels:
    """Decode, window, scale and select the channels of one capture file.

    Parameters
    ----------
    path : str or pathlib.Path
        Capture file.
    survey : SurveyConfig
        Packet window, stream and antenna selection, reader options.
    scaling : ScalingConfig
        Rescaling options.
    use_window : bool, optional
        Apply the packet window; ``predict`` turns it off and uses every report.

    Returns
    -------
    CaptureChannels
        The selected channels.

    Raises
    ------
    DatasetError
        When the capture holds no usable report, mixes channel bandwidths, or
        lacks the requested antennas.

    Examples
    --------
    >>> import tempfile
    >>> from wifi_fingerprint_locator.config import Config
    >>> from wifi_fingerprint_locator.synthetic import make_room, make_capture
    >>> config = Config()
    >>> room = make_room(config.synthetic, rng=0)
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "capture.dat"
    ...     _ = make_capture(path, room, (1.0, 2.0), config.synthetic, n_packets=12, rng=1)
    ...     capture = read_capture_channels(path, config.survey, config.scaling)
    >>> capture.csi.shape, capture.segments[:3].tolist()
    ((12, 3, 30), [0, 1, 2])
    """
    source = Path(path)
    snapshots, _ = read_beamform_file(
        source, apply_permutation=survey.apply_permutation, strict=survey.strict
    )
    if not snapshots:
        raise DatasetError(f"{source} holds no decodable beamforming report")
    start = survey.packet_start if use_window else 0
    stop = None if (not use_window or survey.n_packets is None) else start + survey.n_packets
    window = snapshots[start:stop]
    if use_window and survey.n_packets is not None and len(window) < survey.n_packets:
        _LOGGER.warning(
            "%s holds %d report(s); the window asks for %d from report %d, using %d",
            source.name,
            len(snapshots),
            survey.n_packets,
            start,
            len(window),
        )
    if not window:
        raise DatasetError(
            f"{source} has no report in the window starting at survey.packet_start={start}"
        )
    bandwidths = sorted({s.meta.bandwidth_mhz for s in window})
    if len(bandwidths) > 1:
        raise DatasetError(
            f"{source} mixes channel bandwidths {bandwidths} MHz; subcarrier group k is a "
            "different frequency in each, so they cannot share a fingerprint"
        )
    channels, segments, streams = [], [], []
    n_short = 0
    for position, snapshot in enumerate(window, start=start):
        meta = snapshot.meta
        antennas = range(meta.n_rx) if survey.rx_indices is None else survey.rx_indices
        if meta.n_tx <= survey.tx_index or max(antennas) >= meta.n_rx:
            n_short += 1
            continue
        if scaling.to_snr_units:
            scaled = scale_csi(
                snapshot,
                noise_floor_fallback_dbm=scaling.noise_floor_fallback_dbm,
                three_stream_gain_db=scaling.three_stream_gain_db,
                compensate_stream_power=scaling.compensate_stream_power,
            )
        else:
            scaled = snapshot.csi
        channels.append(scaled[survey.tx_index][list(antennas)])
        segments.append(position)
        streams.append(meta.n_tx)
    if n_short:
        _LOGGER.warning(
            "%s: %d report(s) lack transmit stream %d or a requested antenna; skipped",
            source.name,
            n_short,
            survey.tx_index,
        )
    if not channels:
        raise DatasetError(
            f"{source}: no report carries transmit stream survey.tx_index={survey.tx_index} "
            f"and antennas survey.rx_indices={survey.rx_indices}"
        )
    shapes = {channel.shape for channel in channels}
    if len(shapes) > 1:
        raise DatasetError(f"{source} mixes receive-antenna counts {sorted(shapes)}")
    return CaptureChannels(
        path=str(source),
        csi=np.stack(channels).astype(np.complex128),
        segments=np.asarray(segments, dtype=np.int64),
        n_tx=np.asarray(streams, dtype=np.int64),
        bandwidth_mhz=bandwidths[0],
    )


def _positions(config: Config, bandwidth_mhz: int) -> NDArray[np.float64]:
    return subcarrier_positions(config.features.subcarrier_layout, bandwidth_mhz)


def clean_captures(
    captures: Sequence[CaptureChannels], config: Config
) -> tuple[list[NDArray[np.bool_]], NDArray[np.complex128], CleaningReport, int]:
    """Clean several captures together.

    Parameters
    ----------
    captures : sequence of CaptureChannels
        Captures to clean, all of one bandwidth and antenna count.
    config : Config
        ``cleaning`` and ``features.subcarrier_layout`` are used.

    Returns
    -------
    keep : list of numpy.ndarray of bool
        For every capture, which of its packets survive.
    csi : numpy.ndarray of complex128
        The surviving packets of all captures, in order.
    report : CleaningReport
        Counts of what was dropped and repaired.
    bandwidth_mhz : int
        The shared bandwidth.

    Raises
    ------
    DatasetError
        When the captures differ in bandwidth or antenna count.

    Examples
    --------
    >>> from wifi_fingerprint_locator.config import Config
    >>> ones = CaptureChannels("a.dat", np.ones((3, 3, 30), complex), np.arange(3),
    ...                        np.array([1, 2, 1]), 20)
    >>> keep, csi, report, bandwidth = clean_captures([ones], Config())
    >>> keep[0].tolist(), csi.shape, report.kept_n_tx, bandwidth
    ([True, False, True], (2, 3, 30), 1, 20)
    """
    bandwidths = sorted({capture.bandwidth_mhz for capture in captures})
    if len(bandwidths) != 1:
        raise DatasetError(
            f"the captures mix channel bandwidths {bandwidths} MHz; analyze each bandwidth "
            "separately"
        )
    antenna_counts = sorted({capture.csi.shape[1] for capture in captures})
    if len(antenna_counts) != 1:
        raise DatasetError(f"the captures mix receive-antenna counts {antenna_counts}")
    stacked = np.concatenate([capture.csi for capture in captures])
    streams = np.concatenate([capture.n_tx for capture in captures])
    keep, cleaned, report = clean_channels(
        stacked,
        streams,
        _positions(config, bandwidths[0]),
        n_tx_policy=config.cleaning.n_tx_policy,
        zero_entry_policy=config.cleaning.zero_entry_policy,
        drop_nonfinite=config.cleaning.drop_nonfinite,
    )
    bounds = np.cumsum([0, *(capture.n_packets for capture in captures)])
    per_capture = [keep[lo:hi] for lo, hi in pairwise(bounds)]
    return per_capture, cleaned, report, bandwidths[0]


def read_survey(dataset: str | Path | pd.DataFrame, config: Config) -> SurveyChannels:
    """Read, scale and clean every capture of a dataset index.

    Parameters
    ----------
    dataset : str, pathlib.Path or pandas.DataFrame
        The dataset index, or a frame returned by
        :func:`~wifi_fingerprint_locator.dataset.load_dataset_index`.
    config : Config
        ``survey``, ``scaling``, ``cleaning`` and ``features`` are used.

    Returns
    -------
    SurveyChannels
        Cleaned channels of all captures, in index order.

    Raises
    ------
    DatasetError
        When a capture is unusable or loses every packet in cleaning.

    Examples
    --------
    >>> import tempfile
    >>> from wifi_fingerprint_locator.config import Config
    >>> from wifi_fingerprint_locator.synthetic import write_synthetic_dataset
    >>> config = Config().override({"synthetic.grid_shape": [2, 2], "synthetic.n_packets": 6,
    ...                             "survey.n_packets": 6})
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     index = write_synthetic_dataset(folder, config.synthetic, rng=0)
    ...     survey = read_survey(index, config)
    >>> survey.csi.shape[1:], sorted(set(survey.groups.tolist()))[:2], survey.report.n_input
    ((3, 30), ['point_000', 'point_001'], 24)
    """
    index = dataset if isinstance(dataset, pd.DataFrame) else load_dataset_index(dataset)
    root = index.attrs.get("root", ".")
    captures = [
        read_capture_channels(path, config.survey, config.scaling) for path in index["path"]
    ]
    keep, cleaned, report, bandwidth = clean_captures(captures, config)
    names = relative_paths(list(index["path"]), root)
    emptied = [name for name, mask in zip(names, keep, strict=True) if not mask.any()]
    if emptied:
        raise DatasetError(
            f"cleaning removed every packet of {len(emptied)} capture(s), e.g. {emptied[0]}; "
            "check cleaning.n_tx_policy and cleaning.zero_entry_policy"
        )
    counts = np.array([mask.sum() for mask in keep])
    coordinates = np.repeat(index[list(TARGET_COLUMNS)].to_numpy(dtype=np.float64), counts, axis=0)
    _LOGGER.info(
        "survey: %d capture(s), %d of %d packet(s) kept",
        len(captures),
        report.n_output,
        report.n_input,
    )
    return SurveyChannels(
        csi=cleaned,
        coordinates=coordinates,
        groups=np.repeat(index["group"].astype(str).to_numpy(), counts),
        paths=np.repeat(np.asarray(names), counts),
        segments=np.concatenate(
            [capture.segments[mask] for capture, mask in zip(captures, keep, strict=True)]
        ),
        bandwidth_mhz=bandwidth,
        report=report,
    )


def survey_features(
    survey: SurveyChannels, config: Config, representation: str | None = None
) -> FeatureSet:
    """Fingerprints of a survey in one representation.

    Parameters
    ----------
    survey : SurveyChannels
        Cleaned channels.
    config : Config
        ``features`` is used; the stored configuration is ``config`` with
        ``features.representation`` set to the representation used.
    representation : str, optional
        Overrides ``config.features.representation``.

    Returns
    -------
    FeatureSet
        One fingerprint per packet.

    Examples
    --------
    >>> from wifi_fingerprint_locator.channel_cleaning import CleaningReport
    >>> from wifi_fingerprint_locator.config import Config
    >>> survey = SurveyChannels(np.ones((2, 3, 30), complex), np.zeros((2, 2)),
    ...                         np.array(["p", "p"]), np.array(["a.dat", "a.dat"]),
    ...                         np.arange(2), 20, CleaningReport(2, 0, 0, 0, 0, 2, 1))
    >>> features = survey_features(survey, Config(), "amplitude_phase")
    >>> features.X.shape, features.config["features"]["representation"]
    ((2, 180), 'amplitude_phase')
    """
    if representation is not None and representation != config.features.representation:
        config = config.override({"features.representation": representation})
    settings = config.features
    X = fingerprint_matrix(
        survey.csi,
        settings.representation,
        _positions(config, survey.bandwidth_mhz),
        phase_fit=settings.phase_fit,
        amplitude_floor_db=settings.amplitude_floor_db,
    )
    return FeatureSet(
        X=X,
        y=survey.coordinates,
        groups=survey.groups,
        paths=survey.paths,
        config=config.to_dict(),
        segments=survey.segments,
    )


def index_capture_folder(folder: str | Path, data: DataConfig) -> pd.DataFrame:
    """Build dataset-index rows from capture files named after grid indices.

    Every file below ``folder`` whose name fully matches ``data.file_pattern``
    becomes a row; the named groups ``x`` and ``y`` are grid indices, turned
    into meters as ``(index - data.grid_origin) * data.grid_spacing_m``.

    Parameters
    ----------
    folder : str or pathlib.Path
        Folder to scan (recursively).
    data : DataConfig
        ``file_pattern``, ``grid_spacing_m`` and ``grid_origin`` are used.

    Returns
    -------
    pandas.DataFrame
        Columns ``path`` (relative to ``folder``), ``x_m``, ``y_m``, ``group``
        and ``campaign``, sorted by campaign, x and y.

    Raises
    ------
    FileNotFoundError
        If the folder does not exist.
    DatasetError
        If no file matches the pattern.

    Examples
    --------
    >>> import tempfile
    >>> from wifi_fingerprint_locator.config import DataConfig
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     for name in ("run2-1.dat", "run1-1.dat", "notes.txt"):
    ...         _ = (Path(folder) / name).write_bytes(b"")
    ...     rows = index_capture_folder(folder, DataConfig(grid_spacing_m=0.5))
    >>> rows["path"].tolist(), rows["x_m"].tolist()
    (['run1-1.dat', 'run2-1.dat'], [0.0, 0.5])
    """
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"capture folder not found: {root}")
    pattern = re.compile(data.file_pattern)
    rows = []
    for path in sorted(root.rglob("*")):
        match = pattern.fullmatch(path.name) if path.is_file() else None
        if match is None:
            continue
        x_index, y_index = int(match["x"]), int(match["y"])
        x_m = (x_index - data.grid_origin) * data.grid_spacing_m
        y_m = (y_index - data.grid_origin) * data.grid_spacing_m
        campaign = match.groupdict().get("campaign") or ""
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "x_m": float(x_m),
                "y_m": float(y_m),
                "group": location_name(x_m, y_m),
                "campaign": campaign,
                "_order": (campaign, x_index, y_index),
            }
        )
    if not rows:
        raise DatasetError(f"no file below {root} matches data.file_pattern={data.file_pattern!r}")
    rows.sort(key=lambda row: row["_order"])
    frame = pd.DataFrame(rows).drop(columns="_order")
    _LOGGER.info("%d capture file(s) indexed below %s", len(frame), root)
    return frame
