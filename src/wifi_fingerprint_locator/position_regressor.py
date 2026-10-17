"""Multilayer-perceptron regression of a position from a fingerprint.

:class:`PositionRegressor` is a scikit-learn compatible estimator: a fully
connected network with tanh hidden layers and a linear output layer, trained
on the squared error with an L2 penalty ``alpha``. ``solver`` picks one of
three optimizers:

``lbfgs``
    Quasi-Newton L-BFGS (Liu & Nocedal, 1989) through scikit-learn's
    :class:`~sklearn.neural_network.MLPRegressor`. Fast and memory-light; the
    default.
``adam``
    Stochastic gradient descent with Adam (Kingma & Ba, 2015), also through
    scikit-learn, run one epoch at a time with ``partial_fit`` so that the
    validation checks never reset the moment estimates.
``levenberg_marquardt``
    The damped Gauss-Newton method of
    :mod:`~wifi_fingerprint_locator.levenberg_marquardt`, the classic choice
    for small networks; it needs memory quadratic in the number of weights.

All three stop early on a validation set held out from the training data.
Training runs in chunks of ``epochs_per_check`` iterations (every epoch for
Levenberg-Marquardt), the validation error is measured after each chunk,
training stops after ``patience`` chunks without improvement, and the weights
of the best chunk are restored.

The validation set is a random share of the items, or a random share of
whole reference points when :meth:`PositionRegressor.fit` gets the reference
point of every item. Cross-validation passes them under the location splits,
whose test points never appear in training. Validating there on other packets
of points that are also being fitted would pick the weights that best
recognize places already seen.

When ``alpha_grid`` lists several penalties, one network is trained per value
and the one with the lowest validation error is kept. I use this as a
validation-based substitute for Bayesian regularization.

The estimator expects inputs and targets already scaled to about ``[-1, 1]``;
:func:`~wifi_fingerprint_locator.models.build_model` wraps it in the scaling
pipeline.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.utils.validation import check_is_fitted

from .levenberg_marquardt import network_forward, train_levenberg_marquardt
from .splits import group_codes

_LOGGER = logging.getLogger(__name__)

#: Solvers accepted by :class:`PositionRegressor` and ``model.solver``.
SOLVERS: tuple[str, ...] = ("lbfgs", "adam", "levenberg_marquardt")


class PositionRegressor(RegressorMixin, BaseEstimator):
    """Tanh MLP regression with validation early stopping and best-weight restore.

    Parameters
    ----------
    hidden_layer_sizes : tuple of int, optional
        Width of each hidden layer.
    solver : {"lbfgs", "adam", "levenberg_marquardt"}, optional
        Optimizer.
    alpha : float, optional
        L2 penalty on the weights.
    alpha_grid : tuple of float, optional
        When not empty, candidate penalties chosen among by validation error;
        ``alpha`` is then ignored. Choosing needs a validation set, so
        ``patience`` and ``validation_fraction`` must not be 0.
    max_epochs : int, optional
        Upper bound on iterations (epochs for ``adam`` and Levenberg-Marquardt).
    epochs_per_check : int, optional
        Iterations between validation checks (``lbfgs`` and ``adam``).
    patience : int, optional
        Checks without validation improvement before stopping; 0 disables
        early stopping and the validation split.
    validation_fraction : float, optional
        Share of the training items held out for early stopping; a share of
        the reference points when ``fit`` gets ``groups``.
    learning_rate : float, optional
        Initial step size of ``adam``.
    min_gradient : float, optional
        Gradient tolerance of ``lbfgs`` and Levenberg-Marquardt.
    lm_initial_mu, lm_mu_decrease, lm_mu_increase, lm_mu_max : float, optional
        Damping schedule of Levenberg-Marquardt.
    lm_memory_limit_mb : float, optional
        Memory ceiling of Levenberg-Marquardt.
    random_state : int, optional
        Seed of the weight initialization and of the validation split.

    Attributes
    ----------
    coefs_, intercepts_ : list of numpy.ndarray
        Weights and biases of the kept network.
    alpha_ : float
        Penalty of the kept network.
    n_epochs_ : int
        Iterations run for the kept network.
    best_epoch_ : int
        Iteration whose weights were restored.
    stop_reason_ : str
        Why training of the kept network ended.
    validation_curve_ : list of float
        Validation mean squared error after each check.
    validation_scores_ : dict
        Best validation error of every candidate penalty.

    References
    ----------
    .. [1] D. C. Liu and J. Nocedal, "On the limited memory BFGS method for
       large scale optimization," Mathematical Programming, vol. 45,
       pp. 503-528, 1989.
    .. [2] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization,"
       in Proc. International Conference on Learning Representations (ICLR),
       2015.
    .. [3] L. Prechelt, "Early stopping -- but when?," in Neural Networks:
       Tricks of the Trade, G. B. Orr and K.-R. Müller, Eds., Lecture Notes in
       Computer Science, vol. 1524. Berlin, Germany: Springer, 1998, pp. 55-69.
    .. [4] F. Pedregosa et al., "Scikit-learn: Machine learning in Python,"
       Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(-1.0, 1.0, size=(200, 3))
    >>> y = np.column_stack([X[:, 0] - 0.5 * X[:, 1], 0.8 * X[:, 2]])
    >>> model = PositionRegressor((8,), max_epochs=200, random_state=0).fit(X, y)
    >>> bool(np.mean((model.predict(X) - y) ** 2) < 1e-3), model.coefs_[0].shape
    (True, (3, 8))
    """

    def __init__(
        self,
        hidden_layer_sizes: Sequence[int] = (60, 30, 15),
        *,
        solver: str = "lbfgs",
        alpha: float = 1e-4,
        alpha_grid: Sequence[float] = (),
        max_epochs: int = 1000,
        epochs_per_check: int = 10,
        patience: int = 6,
        validation_fraction: float = 0.15,
        learning_rate: float = 1e-3,
        min_gradient: float = 1e-7,
        lm_initial_mu: float = 1e-3,
        lm_mu_decrease: float = 0.1,
        lm_mu_increase: float = 10.0,
        lm_mu_max: float = 1e10,
        lm_memory_limit_mb: float = 512.0,
        random_state: int = 0,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.solver = solver
        self.alpha = alpha
        self.alpha_grid = alpha_grid
        self.max_epochs = max_epochs
        self.epochs_per_check = epochs_per_check
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.learning_rate = learning_rate
        self.min_gradient = min_gradient
        self.lm_initial_mu = lm_initial_mu
        self.lm_mu_decrease = lm_mu_decrease
        self.lm_mu_increase = lm_mu_increase
        self.lm_mu_max = lm_mu_max
        self.lm_memory_limit_mb = lm_memory_limit_mb
        self.random_state = random_state

    def _split(
        self,
        n_items: int,
        rng: np.random.Generator,
        groups: NDArray[np.int64] | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Indices of the fitting set and of the validation set.

        With integer group codes, whole groups are held out: about
        ``validation_fraction`` of them and at least one, provided two or more
        groups stay for fitting. With fewer groups, random items are held out.
        """
        everything = np.arange(n_items)
        if self.patience <= 0 or self.validation_fraction <= 0:
            return everything, everything[:0]
        if groups is not None:
            n_groups = int(groups.max()) + 1
            n_held = max(round(self.validation_fraction * n_groups), 1)
            if n_groups - n_held >= 2:
                held = np.isin(groups, rng.permutation(n_groups)[:n_held])
                return np.flatnonzero(~held), np.flatnonzero(held)
            _LOGGER.warning(
                "%d reference point(s) are too few to hold whole ones out for early stopping; "
                "validating on random packets instead",
                n_groups,
            )
        n_validation = round(self.validation_fraction * n_items)
        if n_validation < 1 or n_items - n_validation < 2:
            return everything, everything[:0]
        order = rng.permutation(n_items)
        return np.sort(order[n_validation:]), np.sort(order[:n_validation])

    def _fit_sklearn(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_val: NDArray[np.float64],
        y_val: NDArray[np.float64],
        alpha: float,
    ) -> dict[str, Any]:
        validating = X_val.shape[0] > 0
        chunk = self.epochs_per_check if validating else self.max_epochs
        adam = self.solver == "adam"
        network = MLPRegressor(
            hidden_layer_sizes=tuple(int(size) for size in self.hidden_layer_sizes),
            activation="tanh",
            solver=self.solver,
            alpha=alpha,
            learning_rate_init=self.learning_rate,
            max_iter=int(chunk),
            tol=self.min_gradient,
            n_iter_no_change=self.max_epochs + 1,
            warm_start=True,
            # Adam runs epoch by epoch through partial_fit, which re-reads random_state
            # on every call: a RandomState instance keeps advancing, so each epoch gets
            # a new shuffle instead of replaying the first one.
            random_state=np.random.RandomState(self.random_state) if adam else self.random_state,
        )
        best: dict[str, Any] = {}
        best_score = np.inf
        curve: list[float] = []
        failures = 0
        epochs = 0
        stop_reason = "maximum epochs reached"
        # A single output is passed as a vector, which MLPRegressor expects.
        fit_targets = y[:, 0] if y.shape[1] == 1 else y
        while epochs < self.max_epochs:
            n_chunk = int(min(chunk, self.max_epochs - epochs))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                if adam:
                    # fit() would build a new optimizer for every chunk and restart the
                    # moment estimates and the bias correction; partial_fit keeps them,
                    # so the chunks add up to one uninterrupted Adam run.
                    for _ in range(n_chunk):
                        network.partial_fit(X, fit_targets)
                    ran = n_chunk
                else:
                    network.max_iter = n_chunk
                    network.fit(X, fit_targets)
                    ran = int(network.n_iter_)
            epochs += max(ran, 1)
            if not validating:
                best = {
                    "coefs": [c.copy() for c in network.coefs_],
                    "intercepts": [b.copy() for b in network.intercepts_],
                    "epoch": epochs,
                }
                if not adam and ran < n_chunk:
                    stop_reason = "minimum gradient reached"
                    break
                continue
            score = float(np.mean((network.predict(X_val).reshape(y_val.shape) - y_val) ** 2))
            curve.append(score)
            if score < best_score:
                best_score, failures = score, 0
                best = {
                    "coefs": [c.copy() for c in network.coefs_],
                    "intercepts": [b.copy() for b in network.intercepts_],
                    "epoch": epochs,
                }
            else:
                failures += 1
                if failures >= self.patience:
                    stop_reason = "validation stop"
                    break
            if not adam and ran < n_chunk:
                stop_reason = "minimum gradient reached"
                break
        return {
            "coefs": [np.array(c, dtype=np.float64) for c in best["coefs"]],
            "intercepts": [np.array(b, dtype=np.float64) for b in best["intercepts"]],
            "n_epochs": epochs,
            "best_epoch": int(best["epoch"]),
            "stop_reason": stop_reason,
            "validation_curve": curve,
            "validation_score": best_score,
        }

    def _fit_levenberg_marquardt(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_val: NDArray[np.float64],
        y_val: NDArray[np.float64],
        alpha: float,
    ) -> dict[str, Any]:
        validating = X_val.shape[0] > 0
        result = train_levenberg_marquardt(
            X,
            y,
            tuple(int(size) for size in self.hidden_layer_sizes),
            X_val=X_val if validating else None,
            Y_val=y_val if validating else None,
            alpha=alpha,
            max_epochs=self.max_epochs,
            patience=max(int(self.patience), 1),
            min_gradient=self.min_gradient,
            initial_mu=self.lm_initial_mu,
            mu_decrease=self.lm_mu_decrease,
            mu_increase=self.lm_mu_increase,
            mu_max=self.lm_mu_max,
            memory_limit_mb=self.lm_memory_limit_mb,
            rng=self.random_state,
        )
        curve = result.validation_curve
        return {
            "coefs": result.coefs,
            "intercepts": result.intercepts,
            "n_epochs": result.n_epochs,
            "best_epoch": result.best_epoch,
            "stop_reason": result.stop_reason,
            "validation_curve": curve,
            "validation_score": min(curve) if curve else np.inf,
        }

    def fit(self, X: ArrayLike, y: ArrayLike, groups: ArrayLike | None = None) -> PositionRegressor:
        """Train the network(s) and keep the best.

        Parameters
        ----------
        X : array_like, shape (n_items, n_features)
            Scaled fingerprints.
        y : array_like, shape (n_items,) or (n_items, n_outputs)
            Scaled targets.
        groups : array_like, shape (n_items,), optional
            Reference point of every item. When given, early stopping holds
            out whole reference points instead of random items.

        Returns
        -------
        PositionRegressor
            The fitted estimator.

        Raises
        ------
        ValueError
            For bad inputs, or when ``alpha_grid`` has several candidates but
            there is no validation set to choose among them.
        """
        if self.solver not in SOLVERS:
            raise ValueError(f"unknown solver {self.solver!r}; choose one of {list(SOLVERS)}")
        inputs = np.asarray(X, dtype=np.float64)
        targets = np.asarray(y, dtype=np.float64)
        if inputs.ndim != 2:
            raise ValueError(f"X must be two-dimensional, got shape {inputs.shape}")
        self.target_ndim_ = targets.ndim
        targets = targets.reshape(inputs.shape[0], -1)
        if not (np.isfinite(inputs).all() and np.isfinite(targets).all()):
            raise ValueError("X and y must be finite")
        codes = None
        if groups is not None:
            labels = np.asarray(groups)
            if labels.shape != (inputs.shape[0],):
                raise ValueError(
                    f"groups must hold one entry per item ({inputs.shape[0]}), "
                    f"got shape {labels.shape}"
                )
            codes = group_codes(labels)
        rng = np.random.default_rng(self.random_state)
        fit_rows, val_rows = self._split(inputs.shape[0], rng, codes)
        X_fit, y_fit = inputs[fit_rows], targets[fit_rows]
        X_val, y_val = inputs[val_rows], targets[val_rows]

        candidates = tuple(float(a) for a in self.alpha_grid) or (float(self.alpha),)
        if len(candidates) > 1 and val_rows.size == 0:
            raise ValueError(
                f"alpha_grid lists {len(candidates)} penalties but there is no validation set "
                f"to choose among them (patience={self.patience}, "
                f"validation_fraction={self.validation_fraction}, {inputs.shape[0]} items)"
            )
        trainer = (
            self._fit_levenberg_marquardt
            if self.solver == "levenberg_marquardt"
            else self._fit_sklearn
        )
        outcomes = {alpha: trainer(X_fit, y_fit, X_val, y_val, alpha) for alpha in candidates}
        chosen = min(candidates, key=lambda a: outcomes[a]["validation_score"])
        outcome = outcomes[chosen]
        self.coefs_ = outcome["coefs"]
        self.intercepts_ = outcome["intercepts"]
        self.alpha_ = chosen
        self.n_epochs_ = outcome["n_epochs"]
        self.best_epoch_ = outcome["best_epoch"]
        self.stop_reason_ = outcome["stop_reason"]
        self.validation_curve_ = outcome["validation_curve"]
        self.validation_scores_ = {a: outcomes[a]["validation_score"] for a in candidates}
        self.n_features_in_ = inputs.shape[1]
        _LOGGER.debug(
            "%s network: alpha %g, %d iteration(s), best %d, %s",
            self.solver,
            chosen,
            self.n_epochs_,
            self.best_epoch_,
            self.stop_reason_,
        )
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Network output for scaled fingerprints.

        Parameters
        ----------
        X : array_like, shape (n_items, n_features)
            Scaled fingerprints.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_items, n_outputs)``, or ``(n_items,)`` when the
            estimator was fitted on a one-dimensional target.
        """
        check_is_fitted(self, "coefs_")
        inputs = np.asarray(X, dtype=np.float64)
        if inputs.ndim != 2 or inputs.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must be shaped (n_items, {self.n_features_in_}), got {inputs.shape}"
            )
        output = network_forward(inputs, self.coefs_, self.intercepts_)
        return output.ravel() if self.target_ndim_ == 1 else output
