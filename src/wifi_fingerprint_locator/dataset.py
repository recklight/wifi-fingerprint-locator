"""Dataset index (``dataset.csv``) and the fingerprint file (``features.npz``).

The dataset index lists one capture file per row:

=============  ========  ========================================================
column         required  meaning
=============  ========  ========================================================
``path``       yes       capture file, relative to the folder of the CSV
``x_m``        yes       x coordinate of the reference point, meters
``y_m``        yes       y coordinate of the reference point, meters
``group``      no        reference-point name; defaults to one derived from x and y
``campaign``   no        measurement session, kept for bookkeeping
=============  ========  ========================================================

Several files may share a reference point (repeated campaigns); they then
share a group, so a location-wise split keeps them together. File names carry
no meaning: positions and groups come from the columns only.

``prepare`` turns the captures into a :class:`FeatureSet`, one row per packet,
stored as ``features.npz`` together with the configuration that produced it.
"""

from __future__ import annotations

import json
import os
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .errors import ConfigError, DatasetError

if TYPE_CHECKING:
    from .config import Config

#: File name of the dataset index written by ``synthesize`` and ``index``.
DATASET_INDEX_NAME: str = "dataset.csv"
#: Regression targets: coordinates of the reference point, in meters.
TARGET_COLUMNS: tuple[str, ...] = ("x_m", "y_m")
#: Columns every dataset index must have.
REQUIRED_COLUMNS: tuple[str, ...] = ("path", *TARGET_COLUMNS)

_FEATURE_KEYS = ("X", "y", "groups", "paths", "config_json", "segments")
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def location_name(x_m: float, y_m: float) -> str:
    """Default group name of a reference point.

    Examples
    --------
    >>> location_name(1.5, 3.0)
    'x1.5_y3'
    """
    return f"x{float(x_m):g}_y{float(y_m):g}"


def load_dataset_index(path: str | Path) -> pd.DataFrame:
    """Read and validate a dataset index.

    Parameters
    ----------
    path : str or pathlib.Path
        The ``dataset.csv`` file.

    Returns
    -------
    pandas.DataFrame
        One row per capture with ``path`` resolved to an absolute path, the
        coordinates as floats and ``group`` filled in. ``frame.attrs["root"]``
        holds the folder of the CSV and ``frame["relative_path"]`` the path as
        written in the CSV (POSIX form).

    Raises
    ------
    FileNotFoundError
        If the index does not exist.
    DatasetError
        If a required column is missing, a coordinate is not a finite number,
        or listed capture files do not exist.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     root = Path(folder)
    ...     _ = (root / "a.dat").write_bytes(b"")
    ...     _ = (root / "dataset.csv").write_text("path,x_m,y_m\\na.dat,0,1\\n", encoding="utf-8")
    ...     frame = load_dataset_index(root / "dataset.csv")
    >>> frame["group"].tolist(), frame["relative_path"].tolist()
    (['x0_y1'], ['a.dat'])
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"dataset index not found: {source}. Point data.dataset (or --dataset) at your own "
            "dataset.csv, or run 'wifi-fingerprint-locator synthesize' first"
        )
    try:
        frame = pd.read_csv(source, dtype={"path": str, "group": str, "campaign": str})
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DatasetError(f"{source} is not a readable CSV file: {exc}") from exc
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetError(
            f"{source} lacks column(s) {missing}; found {list(frame.columns)}; "
            f"required are {list(REQUIRED_COLUMNS)}"
        )
    if frame.empty:
        raise DatasetError(f"{source} lists no capture files")
    for column in TARGET_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
            raise DatasetError(f"{source}: column {column!r} must hold finite numbers")
        frame[column] = values.astype(np.float64)
    if frame["path"].isna().any():
        raise DatasetError(f"{source}: column 'path' has empty entries")
    root = source.resolve().parent
    relative = [Path(item).as_posix() for item in frame["path"]]
    absolute = [(root / item).resolve() for item in relative]
    absent = [rel for rel, full in zip(relative, absolute, strict=True) if not full.is_file()]
    if absent:
        shown = ", ".join(absent[:5])
        more = f" ... and {len(absent) - 5} more" if len(absent) > 5 else ""
        raise DatasetError(
            f"{len(absent)} capture file(s) listed in {source} do not exist: {shown}{more}"
        )
    if "group" not in frame.columns:
        frame["group"] = [
            location_name(x, y) for x, y in zip(frame["x_m"], frame["y_m"], strict=True)
        ]
    else:
        blank = frame["group"].isna() | (frame["group"].astype(str).str.strip() == "")
        frame.loc[blank, "group"] = [
            location_name(x, y)
            for x, y in zip(frame.loc[blank, "x_m"], frame.loc[blank, "y_m"], strict=True)
        ]
        frame["group"] = frame["group"].astype(str)
    frame["relative_path"] = relative
    frame["path"] = [str(item) for item in absolute]
    frame.attrs["root"] = str(root)
    return frame


def write_dataset_index(path: str | Path, rows: pd.DataFrame) -> Path:
    """Write a dataset index (UTF-8, LF line endings) and return its path.

    Examples
    --------
    >>> import tempfile
    >>> rows = pd.DataFrame({"path": ["a.dat"], "x_m": [0.0], "y_m": [1.0]})
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     target = write_dataset_index(Path(folder) / "dataset.csv", rows)
    ...     target.read_text(encoding="utf-8").splitlines()[0]
    'path,x_m,y_m'
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(destination, index=False, lineterminator="\n")
    return destination


@dataclass(frozen=True, eq=False)
class FeatureSet:
    """Fingerprints of a survey, one row per packet.

    Attributes
    ----------
    X : numpy.ndarray, shape (n_items, n_features)
        Fingerprints.
    y : numpy.ndarray, shape (n_items, 2)
        Reference-point coordinates ``(x_m, y_m)`` in meters.
    groups : numpy.ndarray of str, shape (n_items,)
        Reference-point name of every packet.
    paths : numpy.ndarray of str, shape (n_items,)
        Capture file of every packet, relative to the dataset index.
    config : dict
        The complete configuration ``prepare`` used (``Config.to_dict()``).
    segments : numpy.ndarray of int64, shape (n_items,), optional
        Position of every packet in its capture file, counted from 0.

    Examples
    --------
    >>> features = FeatureSet(np.zeros((2, 3)), np.zeros((2, 2)), np.array(["a", "a"]),
    ...                       np.array(["f.dat", "f.dat"]), {}, np.array([0, 1]))
    >>> features.n_items, features.n_features
    (2, 3)
    """

    X: NDArray[np.float64]
    y: NDArray[np.float64]
    groups: NDArray[np.str_]
    paths: NDArray[np.str_]
    config: dict[str, Any]
    segments: NDArray[np.int64] | None = None

    def __post_init__(self) -> None:
        X = np.asarray(self.X, dtype=np.float64)
        y = np.asarray(self.y, dtype=np.float64)
        if X.ndim != 2:
            raise DatasetError(f"X must be two-dimensional, got shape {X.shape}")
        if y.shape != (X.shape[0], len(TARGET_COLUMNS)):
            raise DatasetError(f"y must be shaped ({X.shape[0]}, 2), got {y.shape}")
        groups = np.asarray(self.groups).astype(str)
        paths = np.asarray(self.paths).astype(str)
        for name, values in (("groups", groups), ("paths", paths)):
            if values.shape != (X.shape[0],):
                raise DatasetError(f"{name} must have one entry per row, got shape {values.shape}")
        segments = None if self.segments is None else np.asarray(self.segments, dtype=np.int64)
        if segments is not None and segments.shape != (X.shape[0],):
            raise DatasetError(f"segments must have one entry per row, got shape {segments.shape}")
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "paths", paths)
        object.__setattr__(self, "segments", segments)

    @property
    def n_items(self) -> int:
        """Number of packets."""
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        """Length of a fingerprint."""
        return int(self.X.shape[1])

    def items_table(self) -> pd.DataFrame:
        """The ``items.csv`` table: ``path``, ``x_m``, ``y_m``, ``group``, ``segment``."""
        table = pd.DataFrame(
            {
                "path": self.paths,
                TARGET_COLUMNS[0]: self.y[:, 0],
                TARGET_COLUMNS[1]: self.y[:, 1],
                "group": self.groups,
            }
        )
        if self.segments is not None:
            table["segment"] = self.segments
        return table


def save_features(path: str | Path, features: FeatureSet) -> Path:
    """Write a :class:`FeatureSet` as compressed ``.npz`` and return its path.

    Examples
    --------
    >>> import tempfile
    >>> features = FeatureSet(np.ones((2, 3)), np.zeros((2, 2)), np.array(["a", "b"]),
    ...                       np.array(["f.dat", "g.dat"]), {"seed": 0}, np.array([0, 0]))
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     loaded = load_features(save_features(Path(folder) / "features.npz", features))
    >>> loaded.X.shape, loaded.groups.tolist(), loaded.config
    ((2, 3), ['a', 'b'], {'seed': 0})
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "X": features.X,
        "y": features.y,
        "groups": features.groups.astype(str),
        "paths": features.paths.astype(str),
        "config_json": np.array(json.dumps(features.config, sort_keys=False)),
    }
    if features.segments is not None:
        arrays["segments"] = features.segments
    # Written member by member with a fixed time stamp (np.savez_compressed stamps the
    # current time), so the same fingerprints always give the same bytes.
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, value in arrays.items():
            info = zipfile.ZipInfo(f"{key}.npy", date_time=_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            with archive.open(info, "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asanyarray(value), allow_pickle=False)
    return destination


def _differing_keys(left: Any, right: Any, prefix: str) -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        keys = sorted(set(left) | set(right))
        found: list[str] = []
        for key in keys:
            found.extend(_differing_keys(left.get(key), right.get(key), f"{prefix}.{key}"))
        return found
    return [] if left == right else [prefix]


def load_features(path: str | Path, config: Config | None = None) -> FeatureSet:
    """Read a ``features.npz`` written by :func:`save_features`.

    Parameters
    ----------
    path : str or pathlib.Path
        The file.
    config : Config, optional
        When given, the sections listed in ``FEATURE_SECTIONS`` must equal the
        ones ``prepare`` used.

    Returns
    -------
    FeatureSet
        The stored fingerprints.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    DatasetError
        If it lacks a key or is not a fingerprint file.
    ConfigError
        If the feature settings differ from the stored ones.
    """
    from .config import FEATURE_SECTIONS

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"features file not found: {source}. Run 'prepare' first")
    try:
        loaded = np.load(source, allow_pickle=False)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise DatasetError(f"{source} is not a features file: {exc}") from exc
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise DatasetError(f"{source} holds a single array, not the archive 'prepare' writes")
    try:
        with loaded as archive:
            contents = {key: archive[key] for key in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise DatasetError(f"{source} is not a features file: {exc}") from exc
    missing = [key for key in _FEATURE_KEYS[:5] if key not in contents]
    if missing:
        raise DatasetError(f"{source} lacks key(s) {missing}; it was not written by 'prepare'")
    stored = json.loads(str(contents["config_json"]))
    if config is not None:
        current = config.to_dict()
        differing = [
            key
            for section in FEATURE_SECTIONS
            for key in _differing_keys(stored.get(section), current.get(section), section)
        ]
        if differing:
            raise ConfigError(
                "feature settings differ from the ones 'prepare' used: "
                f"{', '.join(differing)}; run 'prepare' again or use the same --config"
            )
    return FeatureSet(
        X=contents["X"],
        y=contents["y"],
        groups=contents["groups"],
        paths=contents["paths"],
        config=stored,
        segments=contents.get("segments"),
    )


def relative_paths(paths: Sequence[str | Path], root: str | Path) -> list[str]:
    """POSIX paths relative to ``root`` where possible, absolute otherwise.

    Examples
    --------
    >>> relative_paths(["/data/a/x.dat"], "/data")
    ['a/x.dat']
    """
    result = []
    for item in paths:
        try:
            result.append(Path(os.path.relpath(item, root)).as_posix())
        except ValueError:  # another drive on Windows
            result.append(Path(item).as_posix())
    return result
