"""Cross-validated evaluation of a position estimator.

:func:`cross_validate` divides the fingerprints with the configured split
strategy (:mod:`~wifi_fingerprint_locator.splits`), trains a fresh model from
:func:`~wifi_fingerprint_locator.models.build_model` on every training part
and estimates the positions of the held-out packets. All scaling sits inside
the model pipeline, so it is learned from the training part only. Every
packet gets exactly one out-of-fold estimate, and every metric is computed on
test data only.

Metrics are reported per fold (with mean and ``ddof=1`` standard deviation
over folds) and pooled over all out-of-fold estimates, at two levels:

* packets -- one estimate per packet;
* recordings -- the estimates of all packets of one capture file averaged,
  which is what a user standing still for the length of a capture gets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .dataset import TARGET_COLUMNS, FeatureSet
from .metrics import localization_scores, summarize_folds
from .models import build_model
from .splits import LOCATION_STRATEGIES, fold_indices

if TYPE_CHECKING:
    from .config import Config

_LOGGER = logging.getLogger(__name__)

#: Column names of the estimated coordinates.
PREDICTED_COLUMNS: tuple[str, ...] = tuple(f"predicted_{name}" for name in TARGET_COLUMNS)


@dataclass(frozen=True, eq=False)
class CrossValidationResult:
    """Out-of-fold estimates and their metrics.

    Attributes
    ----------
    predictions : pandas.DataFrame
        ``predictions.csv``: ``path``, ``segment``, ``group``, ``fold``,
        ``x_m``, ``y_m``, ``predicted_x_m``, ``predicted_y_m``, ``error_m``,
        one row per packet in the order of the features.
    folds : pandas.DataFrame
        ``folds.csv``: ``fold``, ``n_train``, ``n_test`` and the scalar metrics.
    metrics : dict
        ``metrics.json``.
    recording_predictions : pandas.DataFrame
        ``recording_predictions.csv``: estimates averaged per capture file.
    """

    predictions: pd.DataFrame
    folds: pd.DataFrame
    metrics: dict[str, Any]
    recording_predictions: pd.DataFrame

    def as_dict(self) -> dict[str, Any]:
        """The metrics, as written to ``metrics.json``."""
        return self.metrics

    @property
    def errors(self) -> NDArray[np.float64]:
        """Euclidean error of every out-of-fold estimate, in meters."""
        return self.predictions["error_m"].to_numpy(dtype=np.float64)


def aggregate_recordings(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average the estimates of every capture file.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Packet-level estimates with ``path``, ``group``, ``fold``, the target
        columns and the predicted columns.

    Returns
    -------
    pandas.DataFrame
        One row per capture in order of first appearance: ``path``, ``group``,
        ``fold`` (-1 when the packets of a capture were tested in several
        folds), ``x_m``, ``y_m``, ``predicted_x_m``, ``predicted_y_m``,
        ``error_m`` and ``n_packets``.

    Examples
    --------
    >>> rows = pd.DataFrame({"path": ["a", "a", "b"], "group": ["p", "p", "q"],
    ...                      "fold": [0, 1, 1], "x_m": [0.0, 0.0, 1.0], "y_m": [0.0, 0.0, 1.0],
    ...                      "predicted_x_m": [1.0, -1.0, 1.0], "predicted_y_m": [0.0, 0.0, 2.0]})
    >>> table = aggregate_recordings(rows)
    >>> table["fold"].tolist(), table["error_m"].tolist(), table["n_packets"].tolist()
    ([-1, 1], [0.0, 1.0], [2, 1])
    """
    grouped = predictions.groupby("path", sort=False)
    table = grouped.agg(
        group=("group", "first"),
        fold_min=("fold", "min"),
        fold_max=("fold", "max"),
        **{name: (name, "first") for name in TARGET_COLUMNS},
        **{name: (name, "mean") for name in PREDICTED_COLUMNS},
        n_packets=("group", "size"),
    ).reset_index()
    table["fold"] = np.where(table["fold_min"] == table["fold_max"], table["fold_min"], -1)
    truth = table[list(TARGET_COLUMNS)].to_numpy(dtype=np.float64)
    estimate = table[list(PREDICTED_COLUMNS)].to_numpy(dtype=np.float64)
    table["error_m"] = np.hypot(*(estimate - truth).T)
    columns = ["path", "group", "fold", *TARGET_COLUMNS, *PREDICTED_COLUMNS, "error_m", "n_packets"]
    return table[columns].astype({"fold": np.int64, "n_packets": np.int64})


def _recording_level(
    predictions: pd.DataFrame, references: NDArray[np.float64]
) -> tuple[pd.DataFrame, dict[str, float]]:
    table = aggregate_recordings(predictions)
    scores = localization_scores(
        table[list(TARGET_COLUMNS)].to_numpy(dtype=np.float64),
        table[list(PREDICTED_COLUMNS)].to_numpy(dtype=np.float64),
        references,
    )
    return table, scores


def cross_validate(features: FeatureSet, config: Config) -> CrossValidationResult:
    """k-fold cross-validation of ``config.model`` under ``config.evaluation.split_strategy``.

    Under ``location_block`` and ``location_holdout`` the MLP draws its
    early-stopping validation set from whole reference points of the training
    part; under the other strategies, and in ``train``, from random packets.

    Parameters
    ----------
    features : FeatureSet
        Fingerprints with positions and reference-point groups.
    config : Config
        ``model``, ``evaluation`` and ``seed`` are used.

    Returns
    -------
    CrossValidationResult
        Out-of-fold estimates, per-fold metrics and their summary.

    Raises
    ------
    ConfigError
        When the data do not allow ``evaluation.n_splits`` folds of the strategy.
    """
    strategy = config.evaluation.split_strategy
    n_splits = config.evaluation.n_splits
    folds = fold_indices(strategy, features.groups, n_splits=n_splits, rng=config.seed)
    references = np.unique(features.y, axis=0)
    estimate = np.full_like(features.y, np.nan)
    fold_of = np.full(features.n_items, -1, dtype=np.int64)
    segments = (
        features.segments
        if features.segments is not None
        else np.zeros(features.n_items, dtype=np.int64)
    )

    # Under the location splits the test points are unseen, so the MLP holds
    # whole reference points out for early stopping as well.
    group_validation = config.model.name == "mlp" and strategy in LOCATION_STRATEGIES

    packet_rows: list[dict[str, Any]] = []
    recording_rows: list[dict[str, Any]] = []
    for fold, (train, test) in enumerate(folds):
        model = build_model(config)
        fit_params = {"regressor__groups": features.groups[train]} if group_validation else {}
        model.fit(features.X[train], features.y[train], **fit_params)
        fold_estimate = np.asarray(model.predict(features.X[test]), dtype=np.float64)
        estimate[test] = fold_estimate.reshape(test.size, -1)
        fold_of[test] = fold
        scores = localization_scores(features.y[test], estimate[test], references)
        packet_rows.append(
            {"fold": fold, "n_train": int(train.size), "n_test": int(test.size), **scores}
        )
        fold_table = _prediction_table(features, segments, estimate, fold_of, test)
        _, recording_scores = _recording_level(fold_table, references)
        recording_rows.append(
            {
                "fold": fold,
                "n_train": int(np.unique(features.paths[train]).size),
                "n_test": int(np.unique(features.paths[test]).size),
                **recording_scores,
            }
        )
        _LOGGER.info(
            "fold %d/%d (%s): %d train, %d test packets, mean error %.3f m",
            fold + 1,
            len(folds),
            strategy,
            train.size,
            test.size,
            scores["mean_error_m"],
        )

    everything = np.arange(features.n_items)
    predictions = _prediction_table(features, segments, estimate, fold_of, everything)
    recording_table, recording_pooled = _recording_level(predictions, references)
    pooled = localization_scores(features.y, estimate, references)

    def _summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        return summarize_folds(
            [
                {k: v for k, v in row.items() if k not in ("fold", "n_train", "n_test")}
                for row in rows
            ]
        )

    metrics: dict[str, Any] = {
        "task": "regression",
        "n_items": features.n_items,
        "n_splits": n_splits,
        "seed": config.seed,
        "folds": packet_rows,
        "summary": _summary(packet_rows),
        "pooled": pooled,
        "recordings": {
            "n_items": len(recording_table),
            "folds": recording_rows,
            "summary": _summary(recording_rows),
            "pooled": recording_pooled,
        },
    }
    return CrossValidationResult(
        predictions=predictions,
        folds=pd.DataFrame(packet_rows),
        metrics=metrics,
        recording_predictions=recording_table,
    )


def _prediction_table(
    features: FeatureSet,
    segments: NDArray[np.int64],
    estimate: NDArray[np.float64],
    fold_of: NDArray[np.int64],
    rows: NDArray[np.int64],
) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "path": features.paths[rows],
            "segment": segments[rows],
            "group": features.groups[rows],
            "fold": fold_of[rows],
            TARGET_COLUMNS[0]: features.y[rows, 0],
            TARGET_COLUMNS[1]: features.y[rows, 1],
            PREDICTED_COLUMNS[0]: estimate[rows, 0],
            PREDICTED_COLUMNS[1]: estimate[rows, 1],
        }
    )
    table["error_m"] = np.hypot(
        estimate[rows, 0] - features.y[rows, 0], estimate[rows, 1] - features.y[rows, 1]
    )
    return table
