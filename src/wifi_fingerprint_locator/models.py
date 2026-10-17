"""Registry of position estimators and the model file format.

Every model is a scikit-learn :class:`~sklearn.pipeline.Pipeline` with the same
front end, fitted on training data only:

1. ``drop_constant`` -- :class:`~sklearn.feature_selection.VarianceThreshold`
   removes features that are constant in the training data;
2. ``scale`` -- :class:`~sklearn.preprocessing.MinMaxScaler` maps every
   remaining feature to ``[-1, 1]``;
3. ``regressor`` -- the estimator named by ``model.name``:

   =====================  =========================================================
   name                   estimator
   =====================  =========================================================
   ``mlp``                :class:`~wifi_fingerprint_locator.position_regressor.PositionRegressor`
                          inside a :class:`~sklearn.compose.TransformedTargetRegressor`
                          that also maps the coordinates to ``[-1, 1]``
   ``knn``                :class:`~sklearn.neighbors.KNeighborsRegressor`, uniform weights
   ``weighted_knn``       :class:`~sklearn.neighbors.KNeighborsRegressor`, inverse distance
   ``nearest_reference``  :class:`~wifi_fingerprint_locator.baselines.NearestReferencePoint`
   =====================  =========================================================

A trained model is stored with :func:`save_model` as a joblib bundle that also
carries the complete configuration, so ``predict`` can rebuild the features
exactly as they were built for training.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from .baselines import NearestReferencePoint
from .errors import ConfigError, ModelFileError
from .position_regressor import PositionRegressor

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

    from .config import Config

#: Format tag of the model bundles written by :func:`save_model`.
MODEL_FORMAT: str = "wifi_fingerprint_locator/model-v1"


def _front_end(regressor: Any) -> Pipeline:
    return Pipeline(
        [
            ("drop_constant", VarianceThreshold(threshold=0.0)),
            ("scale", MinMaxScaler(feature_range=(-1.0, 1.0))),
            ("regressor", regressor),
        ]
    )


def _mlp(config: Config) -> Pipeline:
    settings = config.model
    network = PositionRegressor(
        hidden_layer_sizes=tuple(settings.hidden_layer_sizes),
        solver=settings.solver,
        alpha=settings.alpha,
        alpha_grid=tuple(settings.alpha_grid),
        max_epochs=settings.max_epochs,
        epochs_per_check=settings.epochs_per_check,
        patience=settings.patience,
        validation_fraction=settings.validation_fraction,
        learning_rate=settings.learning_rate,
        min_gradient=settings.min_gradient,
        lm_initial_mu=settings.lm_initial_mu,
        lm_mu_decrease=settings.lm_mu_decrease,
        lm_mu_increase=settings.lm_mu_increase,
        lm_mu_max=settings.lm_mu_max,
        lm_memory_limit_mb=settings.lm_memory_limit_mb,
        random_state=config.seed,
    )
    return _front_end(
        TransformedTargetRegressor(
            regressor=network,
            transformer=MinMaxScaler(feature_range=(-1.0, 1.0)),
            check_inverse=False,
        )
    )


def _knn(config: Config) -> Pipeline:
    return _front_end(KNeighborsRegressor(n_neighbors=config.model.n_neighbors, weights="uniform"))


def _weighted_knn(config: Config) -> Pipeline:
    return _front_end(KNeighborsRegressor(n_neighbors=config.model.n_neighbors, weights="distance"))


def _nearest_reference(config: Config) -> Pipeline:
    return _front_end(NearestReferencePoint())


_FACTORIES: dict[str, Callable[[Config], Pipeline]] = {
    "mlp": _mlp,
    "knn": _knn,
    "weighted_knn": _weighted_knn,
    "nearest_reference": _nearest_reference,
}
#: Model names accepted by :func:`build_model` and ``model.name``.
MODEL_NAMES: tuple[str, ...] = tuple(_FACTORIES)


def build_model(config: Config) -> BaseEstimator:
    """Unfitted pipeline for ``config.model.name``.

    Parameters
    ----------
    config : Config
        Run configuration; ``model`` and ``seed`` are used.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Constant-feature removal, scaling to ``[-1, 1]`` and the estimator.

    Raises
    ------
    ConfigError
        For an unknown model name.

    Examples
    --------
    >>> from wifi_fingerprint_locator.config import Config
    >>> [name for name, _ in build_model(Config()).steps]
    ['drop_constant', 'scale', 'regressor']
    """
    name = config.model.name
    if name not in _FACTORIES:
        raise ConfigError(f"unknown model {name!r}; choose one of {list(MODEL_NAMES)}")
    return _FACTORIES[name](config)


def save_model(path: str | Path, estimator: Any, config: Config) -> Path:
    """Store a fitted pipeline and its configuration as a joblib bundle.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination, usually ``model.joblib``.
    estimator : sklearn.pipeline.Pipeline
        The fitted model.
    config : Config
        The configuration it was trained with.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    from . import __version__  # deferred: the package __init__ imports this module

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, Any] = {
        "format": MODEL_FORMAT,
        "package_version": __version__,
        "config": config.to_dict(),
        "estimator": estimator,
    }
    joblib.dump(bundle, destination)
    return destination


def load_model(path: str | Path) -> dict[str, Any]:
    """Load a bundle written by :func:`save_model`.

    Only load model files from sources you trust: joblib files can run code
    when they are opened.

    Parameters
    ----------
    path : str or pathlib.Path
        The model file.

    Returns
    -------
    dict
        ``format``, ``package_version``, ``config`` and ``estimator``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ModelFileError
        If it is not a bundle of this package.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"model file not found: {source}. Run 'train' first")
    try:
        bundle = joblib.load(source)
    except Exception as exc:  # joblib raises many unrelated types for foreign files
        raise ModelFileError(f"{source} is not a readable model file: {exc}") from exc
    if not isinstance(bundle, dict) or bundle.get("format") != MODEL_FORMAT:
        found = bundle.get("format") if isinstance(bundle, dict) else type(bundle).__name__
        raise ModelFileError(f"{source} has format {found!r}; expected {MODEL_FORMAT!r}")
    missing = [key for key in ("config", "estimator") if key not in bundle]
    if missing:
        raise ModelFileError(f"{source} lacks {missing}; write it again with 'train'")
    return bundle
