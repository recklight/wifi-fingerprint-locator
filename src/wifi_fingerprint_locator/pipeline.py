"""High-level steps shared by the command line and the examples.

* :func:`prepare_features` -- dataset index -> :class:`~wifi_fingerprint_locator.dataset.FeatureSet`
  (:func:`prepare_survey` also returns the cleaning report);
* :func:`train_model` -- fit the configured model on all fingerprints;
* :func:`predict_inputs` -- estimate positions for new capture files, using
  only the settings stored in a model bundle;
* :func:`run_comparison` -- the representation x split-strategy matrix.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .channel_cleaning import CleaningReport
from .comparison import ComparisonResult, compare_representations
from .config import Config
from .dataset import FeatureSet
from .evaluation import PREDICTED_COLUMNS
from .fingerprints import fingerprint_matrix
from .models import build_model
from .site_survey import (
    clean_captures,
    read_capture_channels,
    read_survey,
    survey_features,
)
from .subcarriers import subcarrier_positions

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

_LOGGER = logging.getLogger(__name__)


def prepare_survey(dataset: str | Path, config: Config) -> tuple[FeatureSet, CleaningReport]:
    """Like :func:`prepare_features`, also returning what cleaning dropped and repaired.

    Parameters
    ----------
    dataset : str or pathlib.Path
        ``dataset.csv``.
    config : Config
        Run configuration.

    Returns
    -------
    features : FeatureSet
        One fingerprint per kept packet.
    report : CleaningReport
        Counts of dropped and repaired packets and entries.
    """
    survey = read_survey(dataset, config)
    features = survey_features(survey, config)
    _LOGGER.info(
        "%d fingerprint(s) of %d value(s) (%s) from %d reference point(s)",
        features.n_items,
        features.n_features,
        config.features.representation,
        np.unique(features.groups).size,
    )
    return features, survey.report


def prepare_features(dataset: str | Path, config: Config) -> FeatureSet:
    """Read the captures of a dataset index and build fingerprints.

    Parameters
    ----------
    dataset : str or pathlib.Path
        ``dataset.csv``.
    config : Config
        ``survey``, ``scaling``, ``cleaning`` and ``features`` are used, and
        the whole configuration is stored with the fingerprints.

    Returns
    -------
    FeatureSet
        One fingerprint per kept packet.
    """
    return prepare_survey(dataset, config)[0]


def train_model(features: FeatureSet, config: Config) -> BaseEstimator:
    """Fit ``config.model`` on every fingerprint.

    Early stopping of the MLP holds out random packets here, not whole
    reference points, because the saved model should be fitted on every
    surveyed point.

    Parameters
    ----------
    features : FeatureSet
        Training fingerprints and positions.
    config : Config
        ``model`` and ``seed`` are used.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The fitted model; store it with
        :func:`~wifi_fingerprint_locator.models.save_model`.
    """
    model = build_model(config)
    model.fit(features.X, features.y)
    _LOGGER.info("%s model trained on %d fingerprint(s)", config.model.name, features.n_items)
    return model


def predict_inputs(inputs: Sequence[str | Path], bundle: Mapping[str, Any]) -> pd.DataFrame:
    """Estimate the position of every packet of new capture files.

    Only the configuration stored in ``bundle`` is used, so the fingerprints
    are built exactly as for training. Every report of every input is used
    (the survey packet window does not apply), and the cleaning policies are
    applied to all inputs together.

    Parameters
    ----------
    inputs : sequence of str or pathlib.Path
        Capture files.
    bundle : mapping
        A model bundle from :func:`~wifi_fingerprint_locator.models.load_model`.

    Returns
    -------
    pandas.DataFrame
        ``predictions.csv``: ``path``, ``segment``, ``predicted_x_m`` and
        ``predicted_y_m``, inputs in the given order.
    """
    if not inputs:
        raise ValueError("no input capture file given")
    config = Config.from_dict(bundle["config"])
    estimator = bundle["estimator"]
    paths = [Path(item) for item in inputs]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"capture file not found: {path}")
    captures = [
        read_capture_channels(path, config.survey, config.scaling, use_window=False)
        for path in paths
    ]
    keep, cleaned, _, bandwidth = clean_captures(captures, config)
    settings = config.features
    X = fingerprint_matrix(
        cleaned,
        settings.representation,
        subcarrier_positions(settings.subcarrier_layout, bandwidth),
        phase_fit=settings.phase_fit,
        amplitude_floor_db=settings.amplitude_floor_db,
    )
    estimate = np.asarray(estimator.predict(X), dtype=np.float64).reshape(X.shape[0], -1)
    names = np.concatenate(
        [np.full(int(mask.sum()), path.as_posix()) for path, mask in zip(paths, keep, strict=True)]
    )
    segments = np.concatenate(
        [capture.segments[mask] for capture, mask in zip(captures, keep, strict=True)]
    )
    table = pd.DataFrame(
        {
            "path": names,
            "segment": segments,
            PREDICTED_COLUMNS[0]: estimate[:, 0],
            PREDICTED_COLUMNS[1]: estimate[:, 1],
        }
    )
    for path, mask in zip(paths, keep, strict=True):
        rows = table["path"] == path.as_posix()
        if not mask.any():
            _LOGGER.warning("%s: every packet was removed by cleaning; no estimate", path.name)
            continue
        _LOGGER.info(
            "%s: %d packet(s), mean estimate (%.2f, %.2f) m",
            path.name,
            int(mask.sum()),
            table.loc[rows, PREDICTED_COLUMNS[0]].mean(),
            table.loc[rows, PREDICTED_COLUMNS[1]].mean(),
        )
    return table


def run_comparison(dataset: str | Path, config: Config) -> ComparisonResult:
    """Read a survey once and evaluate every representation under every split strategy.

    Parameters
    ----------
    dataset : str or pathlib.Path
        ``dataset.csv``.
    config : Config
        ``evaluation.compare_representations`` and
        ``evaluation.compare_strategies`` select the matrix.

    Returns
    -------
    ComparisonResult
        Per-fold metrics, summary and out-of-fold errors.
    """
    survey = read_survey(dataset, config)
    return compare_representations(survey, config)
