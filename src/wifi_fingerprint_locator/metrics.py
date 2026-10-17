"""Localization error metrics.

The quantity that matters to a user is the *Euclidean* distance between the
estimated and the true position, :math:`e_i = \\lVert \\hat p_i - p_i \\rVert_2`,
in meters, on test data only. :func:`localization_scores` reports

* the standard regression scores (``mae`` and ``rmse`` over both coordinates
  pooled; ``r2``, ``pearson_r`` and ``spearman_rho`` computed per axis and
  averaged over the two axes),
* the distribution of :math:`e_i` (mean, median, RMS, the 67th, 80th, 90th and
  95th percentiles and the maximum) and the per-axis mean absolute errors,
* ``grid_accuracy`` and ``grid_uar``: each estimate snapped to the nearest
  surveyed reference point, scored as a classification of the reference point
  (accuracy, and the unweighted average recall over reference points).

:func:`empirical_cdf` gives the fraction of errors at or below each threshold,
the usual way localization accuracy is plotted.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

#: Percentiles of the Euclidean error reported by :func:`summarize_errors`.
ERROR_PERCENTILES: tuple[int, ...] = (67, 80, 90, 95)


def _positions(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must be shaped (n_items, 2), got {array.shape}")
    return array


def position_errors(y_true: ArrayLike, y_pred: ArrayLike) -> NDArray[np.float64]:
    """Euclidean distance between true and estimated positions.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n_items, 2)
        Positions in meters.

    Returns
    -------
    numpy.ndarray, shape (n_items,)
        Distances in meters.

    Examples
    --------
    >>> position_errors([[0.0, 0.0], [1.0, 1.0]], [[3.0, 4.0], [1.0, 1.0]]).tolist()
    [5.0, 0.0]
    """
    truth = _positions(y_true, "y_true")
    estimate = _positions(y_pred, "y_pred")
    if truth.shape != estimate.shape:
        raise ValueError(f"y_true {truth.shape} and y_pred {estimate.shape} differ in shape")
    return np.hypot(*(estimate - truth).T)


def summarize_errors(errors: ArrayLike) -> dict[str, float]:
    """Mean, median, RMS, percentiles and maximum of Euclidean errors.

    Parameters
    ----------
    errors : array_like, shape (n_items,)
        Euclidean errors in meters.

    Returns
    -------
    dict
        ``mean_error_m``, ``median_error_m``, ``rmse_error_m``,
        ``p67_error_m``, ``p80_error_m``, ``p90_error_m``, ``p95_error_m`` and
        ``max_error_m``.

    Examples
    --------
    >>> summary = summarize_errors([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> summary["mean_error_m"], summary["median_error_m"], summary["max_error_m"]
    (2.0, 2.0, 4.0)
    """
    values = np.asarray(errors, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("cannot summarize an empty set of errors")
    summary = {
        "mean_error_m": float(values.mean()),
        "median_error_m": float(np.median(values)),
        "rmse_error_m": float(np.sqrt(np.mean(values**2))),
    }
    for level, value in zip(
        ERROR_PERCENTILES, np.percentile(values, ERROR_PERCENTILES), strict=True
    ):
        summary[f"p{level}_error_m"] = float(value)
    summary["max_error_m"] = float(values.max())
    return summary


def empirical_cdf(
    errors: ArrayLike, *, step_m: float = 0.2, max_m: float | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fraction of errors at or below each threshold.

    Parameters
    ----------
    errors : array_like, shape (n_items,)
        Euclidean errors in meters.
    step_m : float, optional
        Spacing of the thresholds.
    max_m : float, optional
        Largest threshold; by default the largest error rounded up to a step.

    Returns
    -------
    thresholds : numpy.ndarray
        ``0, step_m, 2 step_m, ...``.
    fractions : numpy.ndarray
        Non-decreasing values in ``[0, 1]``; the last is 1 when ``max_m`` is
        at least the largest error.

    Examples
    --------
    >>> thresholds, fractions = empirical_cdf([0.1, 0.3, 0.3, 0.9], step_m=0.2)
    >>> thresholds.round(1).tolist(), fractions.tolist()
    ([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.25, 0.75, 0.75, 0.75, 1.0])
    """
    values = np.sort(np.asarray(errors, dtype=np.float64).ravel())
    if values.size == 0:
        raise ValueError("cannot build a CDF from an empty set of errors")
    if not step_m > 0:
        raise ValueError(f"step_m must be strictly positive, got {step_m!r}")
    top = float(values[-1]) if max_m is None else float(max_m)
    n_steps = int(np.ceil(top / step_m - 1e-9))
    thresholds = np.arange(n_steps + 1, dtype=np.float64) * step_m
    fractions = np.searchsorted(values, thresholds + 1e-12, side="right") / values.size
    return thresholds, fractions.astype(np.float64)


def nearest_reference(points: ArrayLike, references: ArrayLike) -> NDArray[np.int64]:
    """Index of the nearest reference point for every point.

    Examples
    --------
    >>> nearest_reference([[0.1, 0.2], [0.9, 1.2]], [[0.0, 0.0], [1.0, 1.0]]).tolist()
    [0, 1]
    """
    query = _positions(points, "points")
    grid = _positions(references, "references")
    distances = np.sum((query[:, None, :] - grid[None, :, :]) ** 2, axis=-1)
    return np.argmin(distances, axis=1).astype(np.int64)


def _mean_correlation(
    truth: NDArray[np.float64], estimate: NDArray[np.float64], method: str
) -> float:
    values = []
    for axis in range(truth.shape[1]):
        a, b = truth[:, axis], estimate[:, axis]
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            continue
        values.append(stats.pearsonr(a, b)[0] if method == "pearson" else stats.spearmanr(a, b)[0])
    return float(np.mean(values)) if values else float("nan")


def localization_scores(
    y_true: ArrayLike, y_pred: ArrayLike, references: ArrayLike | None = None
) -> dict[str, float]:
    """All scalar metrics of a set of position estimates.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n_items, 2)
        True and estimated positions in meters.
    references : array_like, shape (n_references, 2), optional
        Surveyed reference points for ``grid_accuracy`` and ``grid_uar``;
        by default the distinct true positions.

    Returns
    -------
    dict
        In order: ``mae``, ``rmse``, ``r2``, ``pearson_r``, ``spearman_rho``,
        the entries of :func:`summarize_errors`, ``mae_x_m``, ``mae_y_m``,
        ``grid_accuracy`` and ``grid_uar``. Undefined values (for instance a
        correlation over a constant coordinate) are NaN.

    Examples
    --------
    >>> truth = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    >>> scores = localization_scores(truth, truth + [[0.1, 0.0]])
    >>> round(scores["mae"], 6), round(scores["mean_error_m"], 6), scores["grid_accuracy"]
    (0.05, 0.1, 1.0)
    """
    truth = _positions(y_true, "y_true")
    estimate = _positions(y_pred, "y_pred")
    errors = position_errors(truth, estimate)
    scores = {
        "mae": float(mean_absolute_error(truth, estimate)),
        "rmse": float(np.sqrt(mean_squared_error(truth, estimate))),
        "r2": float("nan"),
        "pearson_r": _mean_correlation(truth, estimate, "pearson"),
        "spearman_rho": _mean_correlation(truth, estimate, "spearman"),
    }
    if truth.shape[0] > 1 and np.all(np.ptp(truth, axis=0) > 0):
        scores["r2"] = float(r2_score(truth, estimate))
    scores.update(summarize_errors(errors))
    scores["mae_x_m"] = float(np.mean(np.abs(estimate[:, 0] - truth[:, 0])))
    scores["mae_y_m"] = float(np.mean(np.abs(estimate[:, 1] - truth[:, 1])))
    grid = np.unique(truth, axis=0) if references is None else _positions(references, "references")
    true_point = nearest_reference(truth, grid)
    estimated_point = nearest_reference(estimate, grid)
    scores["grid_accuracy"] = float(np.mean(true_point == estimated_point))
    with warnings.catch_warnings():
        # Estimates may land on reference points absent from this test set.
        warnings.simplefilter("ignore", UserWarning)
        scores["grid_uar"] = float(balanced_accuracy_score(true_point, estimated_point))
    return scores


#: Names of the scalar metrics, in the order :func:`localization_scores` returns them.
METRIC_NAMES: tuple[str, ...] = (
    "mae",
    "rmse",
    "r2",
    "pearson_r",
    "spearman_rho",
    "mean_error_m",
    "median_error_m",
    "rmse_error_m",
    *(f"p{level}_error_m" for level in ERROR_PERCENTILES),
    "max_error_m",
    "mae_x_m",
    "mae_y_m",
    "grid_accuracy",
    "grid_uar",
)


def summarize_folds(fold_scores: Sequence[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and sample standard deviation (``ddof=1``) of every metric across folds.

    Examples
    --------
    >>> summarize_folds([{"mae": 1.0}, {"mae": 3.0}])
    {'mae': {'mean': 2.0, 'std': 1.4142135623730951}}
    """
    if not fold_scores:
        raise ValueError("no fold scores to summarize")
    summary = {}
    for name in fold_scores[0]:
        values = np.array([scores[name] for scores in fold_scores], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[name] = {
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "std": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
        }
    return summary
