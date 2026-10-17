"""The comparison matrix: fingerprint representation x split strategy x fold.

What matters here is less how good one model is than which representation
localizes best, and whether that ranking still holds when the test locations
were never surveyed. :func:`compare_representations` cross-validates the same
model on every representation in ``evaluation.compare_representations`` under
every strategy in ``evaluation.compare_strategies``. The captures are read
once, and every representation gets the same seed and the same folds, so two
rows of the matrix differ in one factor only.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .evaluation import cross_validate
from .metrics import METRIC_NAMES
from .site_survey import SurveyChannels, survey_features

if TYPE_CHECKING:
    from .config import Config

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class ComparisonResult:
    """Fold-level metrics, their summary and the out-of-fold errors of every combination.

    Attributes
    ----------
    folds : pandas.DataFrame
        One row per representation, strategy and fold: ``representation``,
        ``strategy``, ``fold``, ``n_train``, ``n_test`` and the scalar metrics.
    summary : pandas.DataFrame
        One row per representation and strategy: ``n_folds``, then
        ``<metric>_mean`` and ``<metric>_std`` (``ddof=1``) of every metric.
    errors : pandas.DataFrame
        Long table of every out-of-fold Euclidean error: ``representation``,
        ``strategy``, ``error_m``.

    Examples
    --------
    >>> summary = pd.DataFrame({"representation": ["a"], "strategy": ["s"], "n_folds": [2]})
    >>> ComparisonResult(pd.DataFrame(), summary, pd.DataFrame()).summary.shape
    (1, 3)
    """

    folds: pd.DataFrame
    summary: pd.DataFrame
    errors: pd.DataFrame

    def as_dict(self) -> dict[str, Any]:
        """``{strategy: {representation: {metric: {"mean", "std"}}}}`` for JSON."""
        nested: dict[str, Any] = {}
        for row in self.summary.to_dict(orient="records"):
            cell = {
                name: {"mean": row[f"{name}_mean"], "std": row[f"{name}_std"]}
                for name in METRIC_NAMES
            }
            nested.setdefault(row["strategy"], {})[row["representation"]] = {
                "n_folds": int(row["n_folds"]),
                **cell,
            }
        return nested

    def best(self, metric: str = "mean_error_m") -> dict[str, Any]:
        """The combination with the lowest mean of ``metric``.

        Examples
        --------
        >>> summary = pd.DataFrame({"representation": ["a", "b"], "strategy": ["s", "s"],
        ...                         "n_folds": [2, 2], "mean_error_m_mean": [0.5, 0.2]})
        >>> ComparisonResult(pd.DataFrame(), summary, pd.DataFrame()).best()["representation"]
        'b'
        """
        row = self.summary.loc[self.summary[f"{metric}_mean"].idxmin()]
        return {
            "representation": row["representation"],
            "strategy": row["strategy"],
            metric: float(row[f"{metric}_mean"]),
        }


def compare_representations(
    survey: SurveyChannels,
    config: Config,
    *,
    representations: Sequence[str] | None = None,
    strategies: Sequence[str] | None = None,
) -> ComparisonResult:
    """Cross-validate every representation under every split strategy.

    Parameters
    ----------
    survey : SurveyChannels
        Cleaned channels from :func:`~wifi_fingerprint_locator.site_survey.read_survey`.
    config : Config
        Model, folds and seed; ``evaluation.split_strategy`` and
        ``features.representation`` are overridden per combination.
    representations, strategies : sequence of str, optional
        Override ``evaluation.compare_representations`` and
        ``evaluation.compare_strategies``.

    Returns
    -------
    ComparisonResult
        Per-fold metrics, summary and errors.

    Examples
    --------
    >>> import tempfile
    >>> from wifi_fingerprint_locator.config import Config
    >>> from wifi_fingerprint_locator.site_survey import read_survey
    >>> from wifi_fingerprint_locator.synthetic import write_synthetic_dataset
    >>> config = Config().override({"synthetic.grid_shape": [3, 2], "synthetic.n_packets": 6,
    ...                             "survey.n_packets": 6, "model.name": "knn",
    ...                             "model.n_neighbors": 3, "evaluation.n_splits": 2})
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     survey = read_survey(write_synthetic_dataset(folder, config.synthetic, rng=0), config)
    >>> result = compare_representations(survey, config, representations=("amplitude",),
    ...                                  strategies=("packet_random", "location_holdout"))
    >>> result.summary[["representation", "strategy", "n_folds"]].values.tolist()
    [['amplitude', 'packet_random', 2], ['amplitude', 'location_holdout', 2]]
    """
    names = tuple(representations or config.evaluation.compare_representations)
    splits = tuple(strategies or config.evaluation.compare_strategies)
    fold_frames = []
    summary_rows = []
    error_frames = []
    for representation in names:
        features = survey_features(survey, config, representation)
        for strategy in splits:
            run_config = config.override(
                {
                    "features.representation": representation,
                    "evaluation.split_strategy": strategy,
                }
            )
            result = cross_validate(features, run_config)
            folds = result.folds.copy()
            folds.insert(0, "strategy", strategy)
            folds.insert(0, "representation", representation)
            fold_frames.append(folds)
            row: dict[str, Any] = {
                "representation": representation,
                "strategy": strategy,
                "n_folds": len(folds),
            }
            for name in METRIC_NAMES:
                stats = result.metrics["summary"][name]
                row[f"{name}_mean"] = stats["mean"]
                row[f"{name}_std"] = stats["std"]
            summary_rows.append(row)
            error_frames.append(
                pd.DataFrame(
                    {
                        "representation": representation,
                        "strategy": strategy,
                        "error_m": result.errors,
                    }
                )
            )
            _LOGGER.info(
                "%s under %s: mean error %.3f m over %d fold(s)",
                representation,
                strategy,
                row["mean_error_m_mean"],
                len(folds),
            )
    return ComparisonResult(
        folds=pd.concat(fold_frames, ignore_index=True),
        summary=pd.DataFrame(summary_rows),
        errors=pd.concat(error_frames, ignore_index=True).astype({"error_m": np.float64}),
    )
