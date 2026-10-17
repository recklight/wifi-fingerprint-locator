"""Figures and the tables that hold the same numbers.

Every ``plot_<thing>`` returns a :class:`matplotlib.figure.Figure` built with
the object-oriented API (``matplotlib.pyplot`` is never imported, so no display
is needed), and every one has a ``<thing>_table`` that returns the plotted
values as a DataFrame; the commands write both, the table as a CSV with the
same stem as the image.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from numpy.typing import ArrayLike

from .metrics import empirical_cdf, summarize_errors

#: Line and marker colors, in series order.
SERIES_COLORS: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0", "#c9a227")
#: Marker shapes, in series order, as a second encoding next to color.
SERIES_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v")


def new_figure(width: float = 6.0, height: float = 4.0, dpi: int = 150) -> Figure:
    """A constrained-layout figure that needs no display.

    Examples
    --------
    >>> new_figure(4, 3, 100).get_size_inches().tolist()
    [4.0, 3.0]
    """
    return Figure(figsize=(width, height), dpi=dpi, layout="constrained")


def save_figure(figure: Figure, path_stem: str | Path, figure_format: str = "png") -> Path:
    """Save ``figure`` as ``<path_stem>.<figure_format>`` and return the path.

    The extension is appended, so dots in the stem are kept.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     save_figure(new_figure(), Path(folder) / "run.v1", "svg").name
    'run.v1.svg'
    """
    stem = Path(path_stem)
    destination = stem.with_name(f"{stem.name}.{figure_format}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format=figure_format)
    return destination


def _common_top(curves: Mapping[str, ArrayLike], step_m: float) -> float:
    top = max(float(np.max(np.asarray(errors, dtype=np.float64))) for errors in curves.values())
    return max(step_m, float(np.ceil(top / step_m - 1e-9) * step_m))


def error_cdf_table(curves: Mapping[str, ArrayLike], step_m: float = 0.2) -> pd.DataFrame:
    """Empirical CDF of several error sets on shared thresholds.

    Parameters
    ----------
    curves : mapping of str to array_like
        Name of each curve and its Euclidean errors in meters.
    step_m : float, optional
        Threshold spacing.

    Returns
    -------
    pandas.DataFrame
        Columns ``curve``, ``threshold_m`` and ``fraction``.

    Examples
    --------
    >>> table = error_cdf_table({"a": [0.1, 0.5], "b": [0.3]}, step_m=0.2)
    >>> table.groupby("curve")["fraction"].max().tolist()
    [1.0, 1.0]
    """
    if not curves:
        raise ValueError("no error curves to tabulate")
    top = _common_top(curves, step_m)
    frames = []
    for name, errors in curves.items():
        thresholds, fractions = empirical_cdf(errors, step_m=step_m, max_m=top)
        frames.append(
            pd.DataFrame({"curve": name, "threshold_m": thresholds, "fraction": fractions})
        )
    return pd.concat(frames, ignore_index=True)


def plot_error_cdf(
    curves: Mapping[str, ArrayLike],
    *,
    step_m: float = 0.2,
    title: str = "Localization error CDF",
    dpi: int = 150,
) -> Figure:
    """Empirical CDF of the Euclidean error, one curve per entry.

    Examples
    --------
    >>> figure = plot_error_cdf({"packets": [0.1, 0.4, 0.8]})
    >>> figure.axes[0].get_xlabel()
    'Euclidean error (m)'
    """
    table = error_cdf_table(curves, step_m)
    figure = new_figure(6.0, 4.0, dpi)
    axes = figure.add_subplot()
    for index, (name, part) in enumerate(table.groupby("curve", sort=False)):
        median = summarize_errors(np.asarray(curves[name]))["median_error_m"]
        axes.step(
            part["threshold_m"],
            part["fraction"],
            where="post",
            color=SERIES_COLORS[index % len(SERIES_COLORS)],
            marker=SERIES_MARKERS[index % len(SERIES_MARKERS)],
            markersize=3,
            markevery=max(1, len(part) // 12),
            label=f"{name} (median {median:.2f} m)",
        )
    axes.set_xlabel("Euclidean error (m)")
    axes.set_ylabel("Fraction of estimates")
    axes.set_ylim(0.0, 1.02)
    axes.set_title(title)
    axes.grid(alpha=0.3)
    axes.legend(loc="lower right", fontsize="small")
    return figure


def residuals_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Error vector of every estimate.

    Returns
    -------
    pandas.DataFrame
        ``path``, ``segment``, ``fold``, ``dx_m``, ``dy_m`` and ``error_m``.

    Examples
    --------
    >>> rows = pd.DataFrame({"path": ["a"], "segment": [0], "fold": [0], "x_m": [1.0],
    ...                      "y_m": [1.0], "predicted_x_m": [1.5], "predicted_y_m": [1.0]})
    >>> residuals_table(rows)[["dx_m", "dy_m", "error_m"]].values.tolist()
    [[0.5, 0.0, 0.5]]
    """
    dx = predictions["predicted_x_m"].to_numpy(dtype=np.float64) - predictions["x_m"].to_numpy(
        dtype=np.float64
    )
    dy = predictions["predicted_y_m"].to_numpy(dtype=np.float64) - predictions["y_m"].to_numpy(
        dtype=np.float64
    )
    table = pd.DataFrame(
        {
            "path": predictions["path"].to_numpy(),
            "segment": predictions["segment"].to_numpy() if "segment" in predictions else 0,
            "fold": predictions["fold"].to_numpy() if "fold" in predictions else -1,
            "dx_m": dx,
            "dy_m": dy,
        }
    )
    table["error_m"] = np.hypot(dx, dy)
    return table


def plot_residuals(predictions: pd.DataFrame, *, dpi: int = 150) -> Figure:
    """Scatter of the error vectors, with circles at the median and 90th-percentile error.

    Examples
    --------
    >>> rows = pd.DataFrame({"path": ["a", "b"], "segment": [0, 0], "fold": [0, 1],
    ...                      "x_m": [1.0, 0.0], "y_m": [1.0, 0.0],
    ...                      "predicted_x_m": [1.5, 0.0], "predicted_y_m": [1.0, -0.2]})
    >>> plot_residuals(rows).axes[0].get_title()
    'Error vectors (estimate - truth)'
    """
    table = residuals_table(predictions)
    summary = summarize_errors(table["error_m"].to_numpy())
    figure = new_figure(5.0, 5.0, dpi)
    axes = figure.add_subplot()
    axes.scatter(table["dx_m"], table["dy_m"], s=6, alpha=0.4, color=SERIES_COLORS[0])
    for index, (key, label) in enumerate(
        (("median_error_m", "median"), ("p90_error_m", "90th percentile")), start=1
    ):
        radius = summary[key]
        axes.add_patch(
            Circle(
                (0.0, 0.0),
                radius,
                fill=False,
                color=SERIES_COLORS[index],
                label=f"{label} {radius:.2f} m",
            )
        )
    limit = max(float(np.max(np.abs(table[["dx_m", "dy_m"]].to_numpy()))), summary["p90_error_m"])
    limit = limit * 1.1 if limit > 0 else 1.0
    axes.set_xlim(-limit, limit)
    axes.set_ylim(-limit, limit)
    axes.set_aspect("equal")
    axes.axhline(0.0, color="0.6", linewidth=0.8)
    axes.axvline(0.0, color="0.6", linewidth=0.8)
    axes.set_xlabel("x error (m)")
    axes.set_ylabel("y error (m)")
    axes.set_title("Error vectors (estimate - truth)")
    axes.legend(loc="upper right", fontsize="small")
    return figure


def location_errors_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Error statistics of every reference point.

    Returns
    -------
    pandas.DataFrame
        ``group``, ``x_m``, ``y_m``, ``n_items``, ``mean_error_m``,
        ``median_error_m``, ``mean_dx_m`` and ``mean_dy_m``, one row per
        reference point in order of first appearance.

    Examples
    --------
    >>> rows = pd.DataFrame({"group": ["p", "p"], "x_m": [0.0, 0.0], "y_m": [0.0, 0.0],
    ...                      "predicted_x_m": [1.0, 3.0], "predicted_y_m": [0.0, 0.0]})
    >>> location_errors_table(rows)[["n_items", "mean_error_m", "mean_dx_m"]].values.tolist()
    [[2.0, 2.0, 2.0]]
    """
    frame = pd.DataFrame(
        {
            "group": predictions["group"].to_numpy(),
            "x_m": predictions["x_m"].to_numpy(dtype=np.float64),
            "y_m": predictions["y_m"].to_numpy(dtype=np.float64),
            "dx_m": predictions["predicted_x_m"].to_numpy(dtype=np.float64)
            - predictions["x_m"].to_numpy(dtype=np.float64),
            "dy_m": predictions["predicted_y_m"].to_numpy(dtype=np.float64)
            - predictions["y_m"].to_numpy(dtype=np.float64),
        }
    )
    frame["error_m"] = np.hypot(frame["dx_m"], frame["dy_m"])
    table = (
        frame.groupby("group", sort=False)
        .agg(
            x_m=("x_m", "first"),
            y_m=("y_m", "first"),
            n_items=("error_m", "size"),
            mean_error_m=("error_m", "mean"),
            median_error_m=("error_m", "median"),
            mean_dx_m=("dx_m", "mean"),
            mean_dy_m=("dy_m", "mean"),
        )
        .reset_index()
    )
    return table.astype({"n_items": np.int64})


def plot_location_errors(predictions: pd.DataFrame, *, dpi: int = 150) -> Figure:
    """Map of the survey: mean error at every reference point and the mean error vector.

    Examples
    --------
    >>> rows = pd.DataFrame({"group": ["p", "q"], "x_m": [0.0, 1.0], "y_m": [0.0, 0.0],
    ...                      "predicted_x_m": [0.5, 1.0], "predicted_y_m": [0.0, 0.4]})
    >>> plot_location_errors(rows).axes[0].get_xlabel()
    'x (m)'
    """
    table = location_errors_table(predictions)
    figure = new_figure(6.0, 5.0, dpi)
    axes = figure.add_subplot()
    points = axes.scatter(
        table["x_m"],
        table["y_m"],
        c=table["mean_error_m"],
        cmap="viridis",
        s=120,
        edgecolors="0.3",
    )
    axes.quiver(
        table["x_m"],
        table["y_m"],
        table["mean_dx_m"],
        table["mean_dy_m"],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=SERIES_COLORS[1],
        width=0.004,
    )
    figure.colorbar(points, ax=axes, label="Mean error (m)")
    axes.set_aspect("equal")
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_title("Mean error per reference point")
    axes.margins(0.15)
    return figure


def comparison_cdf_table(errors: pd.DataFrame, step_m: float = 0.2) -> pd.DataFrame:
    """CDF of every representation under every strategy, on shared thresholds.

    Parameters
    ----------
    errors : pandas.DataFrame
        ``representation``, ``strategy`` and ``error_m`` (see
        :class:`~wifi_fingerprint_locator.comparison.ComparisonResult`).
    step_m : float, optional
        Threshold spacing.

    Returns
    -------
    pandas.DataFrame
        ``strategy``, ``representation``, ``threshold_m`` and ``fraction``.

    Examples
    --------
    >>> errors = pd.DataFrame({"representation": ["a", "a", "b"], "strategy": "s",
    ...                        "error_m": [0.1, 0.3, 0.5]})
    >>> round(float(comparison_cdf_table(errors)["threshold_m"].max()), 6)
    0.6
    """
    curves = {
        (strategy, name): part["error_m"].to_numpy(dtype=np.float64)
        for (strategy, name), part in errors.groupby(["strategy", "representation"], sort=False)
    }
    top = _common_top({f"{s}/{n}": v for (s, n), v in curves.items()}, step_m)
    frames = []
    for (strategy, name), values in curves.items():
        thresholds, fractions = empirical_cdf(values, step_m=step_m, max_m=top)
        frames.append(
            pd.DataFrame(
                {
                    "strategy": strategy,
                    "representation": name,
                    "threshold_m": thresholds,
                    "fraction": fractions,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def plot_comparison_cdf(errors: pd.DataFrame, *, step_m: float = 0.2, dpi: int = 150) -> Figure:
    """One panel per split strategy, one CDF curve per representation.

    Examples
    --------
    >>> errors = pd.DataFrame({"representation": ["a", "b", "a", "b"],
    ...                        "strategy": ["s", "s", "t", "t"], "error_m": [0.1, 0.3, 0.5, 0.2]})
    >>> [axes.get_title() for axes in plot_comparison_cdf(errors).axes]
    ['s', 't']
    """
    table = comparison_cdf_table(errors, step_m)
    strategies = list(dict.fromkeys(table["strategy"]))
    representations = list(dict.fromkeys(table["representation"]))
    n_panels = len(strategies)
    n_columns = min(n_panels, 2)
    n_rows = int(np.ceil(n_panels / n_columns))
    figure = new_figure(5.0 * n_columns, 3.6 * n_rows, dpi)
    panels = figure.subplots(n_rows, n_columns, squeeze=False, sharex=True, sharey=True)
    for panel_index, strategy in enumerate(strategies):
        axes = panels[panel_index // n_columns][panel_index % n_columns]
        for index, name in enumerate(representations):
            part = table[(table["strategy"] == strategy) & (table["representation"] == name)]
            if part.empty:
                continue
            axes.step(
                part["threshold_m"],
                part["fraction"],
                where="post",
                color=SERIES_COLORS[index % len(SERIES_COLORS)],
                marker=SERIES_MARKERS[index % len(SERIES_MARKERS)],
                markersize=3,
                markevery=max(1, len(part) // 12),
                label=name,
            )
        axes.set_title(strategy)
        axes.set_ylim(0.0, 1.02)
        axes.grid(alpha=0.3)
        axes.set_xlabel("Euclidean error (m)")
        axes.set_ylabel("Fraction of estimates")
        axes.legend(loc="lower right", fontsize="small")
    for panel_index in range(n_panels, n_rows * n_columns):
        panels[panel_index // n_columns][panel_index % n_columns].set_visible(False)
    return figure


def phase_sanitization_table(
    raw_phase: ArrayLike, sanitized_phase: ArrayLike, positions: ArrayLike
) -> pd.DataFrame:
    """Phases of several packets before and after sanitization, long format.

    Parameters
    ----------
    raw_phase, sanitized_phase : array_like, shape (n_packets, n_subcarriers)
        Unwrapped and sanitized phase of one antenna.
    positions : array_like, shape (n_subcarriers,)
        Subcarrier positions.

    Returns
    -------
    pandas.DataFrame
        ``packet``, ``subcarrier``, ``raw_phase_rad`` and ``sanitized_phase_rad``.

    Examples
    --------
    >>> phase_sanitization_table(np.zeros((2, 3)), np.ones((2, 3)), [-1, 0, 1]).shape
    (6, 4)
    """
    raw = np.atleast_2d(np.asarray(raw_phase, dtype=np.float64))
    clean = np.atleast_2d(np.asarray(sanitized_phase, dtype=np.float64))
    m = np.asarray(positions, dtype=np.float64)
    if raw.shape != clean.shape or raw.shape[1] != m.size:
        raise ValueError(
            f"phases {raw.shape} and {clean.shape} must match each other and {m.size} positions"
        )
    n_packets = raw.shape[0]
    return pd.DataFrame(
        {
            "packet": np.repeat(np.arange(n_packets), m.size),
            "subcarrier": np.tile(m, n_packets),
            "raw_phase_rad": raw.ravel(),
            "sanitized_phase_rad": clean.ravel(),
        }
    )


def plot_phase_sanitization(
    raw_phase: ArrayLike, sanitized_phase: ArrayLike, positions: ArrayLike, *, dpi: int = 150
) -> Figure:
    """Unwrapped phase of several packets before and after sanitization.

    Examples
    --------
    >>> figure = plot_phase_sanitization(np.zeros((2, 3)), np.ones((2, 3)), [-1, 0, 1])
    >>> [axes.get_title() for axes in figure.axes]
    ['Unwrapped phase', 'Sanitized phase']
    """
    table = phase_sanitization_table(raw_phase, sanitized_phase, positions)
    figure = new_figure(9.0, 3.6, dpi)
    left, right = figure.subplots(1, 2, sharex=True)
    for packet, part in table.groupby("packet"):
        color = SERIES_COLORS[int(packet) % len(SERIES_COLORS)]
        left.plot(part["subcarrier"], part["raw_phase_rad"], color=color, linewidth=1.0)
        right.plot(part["subcarrier"], part["sanitized_phase_rad"], color=color, linewidth=1.0)
    for axes, title in ((left, "Unwrapped phase"), (right, "Sanitized phase")):
        axes.set_title(title)
        axes.set_xlabel("Subcarrier index")
        axes.set_ylabel("Phase (rad)")
        axes.grid(alpha=0.3)
    return figure
