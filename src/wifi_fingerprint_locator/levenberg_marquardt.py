"""A tanh multilayer perceptron trained by Levenberg-Marquardt, in NumPy.

The network maps an input :math:`x` through hidden layers
:math:`a_l = \\tanh(a_{l-1} W_l + b_l)` to a linear output
:math:`f(x) = a_{L-1} W_L + b_L`. With residuals :math:`e = f(X) - Y` over all
training items and outputs and the Jacobian :math:`J = \\partial e / \\partial w`,
each epoch solves

.. math::

    (J^\\top J + \\alpha K D + \\mu I)\\,\\Delta w = -(J^\\top e + \\alpha K D w)

for the step (Levenberg, 1944; Marquardt, 1963; Hagan & Menhaj, 1994), where
:math:`D` selects the weights (not the biases) of the L2 penalty
:math:`\\alpha K \\lVert W \\rVert^2` and :math:`K` is the number of outputs.
The factor :math:`K` puts :math:`\\alpha` on the scale of scikit-learn's
``MLPRegressor``, whose loss is this objective divided by :math:`2NK` for
:math:`N` items. A step that lowers the objective is accepted
and :math:`\\mu` is multiplied by ``mu_decrease``; otherwise :math:`\\mu` grows by
``mu_increase`` and the step is retried, until :math:`\\mu` exceeds ``mu_max``.
The Jacobian comes from back-propagation, one output at a time, so it is
exact. Training stops early when the validation error has not improved for
``patience`` consecutive epochs (Prechelt, 1998) and the weights of the best
validation epoch are returned.

The method needs the dense Jacobian (``n_items * n_outputs`` by ``n_parameters``)
and the normal matrix (``n_parameters`` squared) in memory, so it suits small
networks and moderate data sets. :func:`lm_memory_mb` estimates the need, and
:func:`train_levenberg_marquardt` refuses to start above a limit instead of
exhausting memory. Initial weights follow Nguyen & Widrow (1990).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray

from .errors import ConfigError

_LOGGER = logging.getLogger(__name__)

_Layers = list[NDArray[np.float64]]


@dataclass(frozen=True, eq=False)
class LevenbergMarquardtResult:
    """Outcome of :func:`train_levenberg_marquardt`.

    Attributes
    ----------
    coefs, intercepts : list of numpy.ndarray
        Weights ``(n_in, n_out)`` and biases ``(n_out,)`` of every layer, from
        the best epoch.
    n_epochs : int
        Epochs run.
    best_epoch : int
        Epoch whose weights were kept (0 = the initial weights).
    stop_reason : str
        Why training ended.
    training_curve : list of float
        Training objective per epoch, starting with the initial weights: the
        squared error plus the weight penalty, divided by
        ``n_items * n_outputs``. Only with ``alpha=0`` is it the mean squared
        error.
    validation_curve : list of float
        Validation mean squared error per epoch, starting with the initial
        weights; empty without validation data.

    Examples
    --------
    >>> result = LevenbergMarquardtResult([], [], 0, 0, "maximum epochs reached", [], [])
    >>> result.stop_reason
    'maximum epochs reached'
    """

    coefs: _Layers
    intercepts: _Layers
    n_epochs: int
    best_epoch: int
    stop_reason: str
    training_curve: list[float]
    validation_curve: list[float]


def layer_sizes(n_inputs: int, hidden_layer_sizes: Sequence[int], n_outputs: int) -> list[int]:
    """Width of every layer, input and output included.

    Examples
    --------
    >>> layer_sizes(90, (60, 30, 15), 2)
    [90, 60, 30, 15, 2]
    """
    sizes = [int(n_inputs), *(int(size) for size in hidden_layer_sizes), int(n_outputs)]
    if min(sizes) < 1:
        raise ValueError(f"every layer needs at least one unit, got {sizes}")
    return sizes


def n_parameters(sizes: Sequence[int]) -> int:
    """Number of weights and biases of a network with these layer sizes.

    Examples
    --------
    >>> n_parameters([90, 60, 30, 15, 2])
    7787
    """
    return int(sum((n_in + 1) * n_out for n_in, n_out in pairwise(sizes)))


def lm_memory_mb(n_residuals: int, n_params: int) -> float:
    """Approximate memory, in MB, of one Levenberg-Marquardt epoch.

    Counts the Jacobian twice (back-propagated blocks and the assembled
    matrix) and the normal matrix twice (it and its damped copy).

    Examples
    --------
    >>> round(lm_memory_mb(15_680, 7_787))
    2924
    """
    return 8.0 * (2.0 * n_residuals * n_params + 2.0 * n_params**2) / 1e6


def nguyen_widrow_weights(
    sizes: Sequence[int], rng: np.random.Generator | int | None = None
) -> tuple[_Layers, _Layers]:
    """Initial weights: Nguyen-Widrow for the tanh layers, Glorot-uniform for the output.

    In a hidden layer with :math:`n` inputs and :math:`h` units every weight
    vector is drawn uniformly, then rescaled to the length
    :math:`\\beta = 0.7\\,h^{1/n}`, and the biases are drawn from
    :math:`[-\\beta, \\beta]`, which spreads the active regions of the units over
    inputs scaled to :math:`[-1, 1]`.

    Parameters
    ----------
    sizes : sequence of int
        Layer widths from :func:`layer_sizes`.
    rng : numpy.random.Generator or int, optional
        Randomness.

    Returns
    -------
    coefs, intercepts : list of numpy.ndarray
        Weights ``(n_in, n_out)`` and biases of every layer.

    References
    ----------
    .. [1] D. Nguyen and B. Widrow, "Improving the learning speed of 2-layer
       neural networks by choosing initial values of the adaptive weights,"
       in Proc. International Joint Conference on Neural Networks (IJCNN),
       vol. 3, 1990, pp. 21-26.

    Examples
    --------
    >>> coefs, intercepts = nguyen_widrow_weights([4, 3, 2], rng=0)
    >>> [c.shape for c in coefs], np.linalg.norm(coefs[0], axis=0).round(3).tolist()
    ([(4, 3), (3, 2)], [0.921, 0.921, 0.921])
    """
    rng = np.random.default_rng(rng)
    coefs: _Layers = []
    intercepts: _Layers = []
    n_layers = len(sizes) - 1
    for index, (n_in, n_out) in enumerate(pairwise(sizes)):
        if index < n_layers - 1:
            beta = 0.7 * n_out ** (1.0 / n_in)
            weights = rng.uniform(-1.0, 1.0, size=(n_in, n_out))
            weights *= beta / np.maximum(np.linalg.norm(weights, axis=0), 1e-12)
            biases = rng.uniform(-beta, beta, size=n_out)
        else:
            limit = np.sqrt(6.0 / (n_in + n_out))
            weights = rng.uniform(-limit, limit, size=(n_in, n_out))
            biases = rng.uniform(-limit, limit, size=n_out)
        coefs.append(weights)
        intercepts.append(biases)
    return coefs, intercepts


def _activations(X: NDArray[np.float64], coefs: _Layers, intercepts: _Layers) -> _Layers:
    layers = [X]
    last = len(coefs) - 1
    for index, (weights, biases) in enumerate(zip(coefs, intercepts, strict=True)):
        z = layers[-1] @ weights + biases
        layers.append(z if index == last else np.tanh(z))
    return layers


def network_forward(X: ArrayLike, coefs: _Layers, intercepts: _Layers) -> NDArray[np.float64]:
    """Output of the network: tanh hidden layers, linear output layer.

    Examples
    --------
    >>> coefs = [np.ones((2, 1)), np.full((1, 1), 2.0)]
    >>> intercepts = [np.zeros(1), np.zeros(1)]
    >>> network_forward([[0.0, 0.0]], coefs, intercepts).tolist()
    [[0.0]]
    """
    values = np.asarray(X, dtype=np.float64)
    return _activations(values, coefs, intercepts)[-1]


def pack_parameters(coefs: _Layers, intercepts: _Layers) -> NDArray[np.float64]:
    """Flatten weights and biases into one vector, layer by layer (weights first).

    Examples
    --------
    >>> pack_parameters([np.ones((2, 1))], [np.zeros(1)]).tolist()
    [1.0, 1.0, 0.0]
    """
    return np.concatenate(
        [part for w, b in zip(coefs, intercepts, strict=True) for part in (w.ravel(), b.ravel())]
    )


def unpack_parameters(vector: ArrayLike, sizes: Sequence[int]) -> tuple[_Layers, _Layers]:
    """Inverse of :func:`pack_parameters`.

    Examples
    --------
    >>> coefs, intercepts = unpack_parameters(np.arange(3.0), [2, 1])
    >>> coefs[0].ravel().tolist(), intercepts[0].tolist()
    ([0.0, 1.0], [2.0])
    """
    values = np.asarray(vector, dtype=np.float64)
    needed = n_parameters(sizes)
    if values.size != needed:
        raise ValueError(f"vector holds {values.size} values, the layers need {needed}")
    coefs: _Layers = []
    intercepts: _Layers = []
    start = 0
    for n_in, n_out in pairwise(sizes):
        coefs.append(values[start : start + n_in * n_out].reshape(n_in, n_out).copy())
        start += n_in * n_out
        intercepts.append(values[start : start + n_out].copy())
        start += n_out
    return coefs, intercepts


def network_jacobian(
    X: ArrayLike, coefs: _Layers, intercepts: _Layers
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Network output and its exact Jacobian with respect to all parameters.

    Parameters
    ----------
    X : array_like, shape (n_items, n_inputs)
        Inputs.
    coefs, intercepts : list of numpy.ndarray
        Network parameters.

    Returns
    -------
    output : numpy.ndarray, shape (n_items, n_outputs)
        Network output.
    jacobian : numpy.ndarray, shape (n_items * n_outputs, n_parameters)
        Row ``i * n_outputs + o`` holds the derivatives of output ``o`` of
        item ``i``, columns ordered as :func:`pack_parameters`.

    Examples
    --------
    >>> coefs, intercepts = nguyen_widrow_weights([3, 4, 2], rng=1)
    >>> X = np.random.default_rng(2).normal(size=(5, 3))
    >>> output, jacobian = network_jacobian(X, coefs, intercepts)
    >>> theta = pack_parameters(coefs, intercepts)
    >>> step = np.zeros_like(theta)
    >>> step[7] = 1e-6
    >>> moved = network_forward(X, *unpack_parameters(theta + step, [3, 4, 2]))
    >>> bool(np.allclose((moved - output).ravel() / 1e-6, jacobian[:, 7], atol=1e-5))
    True
    """
    values = np.asarray(X, dtype=np.float64)
    layers = _activations(values, coefs, intercepts)
    output = layers[-1]
    n_items, n_outputs = output.shape
    # delta[i, o, j]: derivative of output o of item i with respect to the
    # pre-activation j of the current layer.
    delta = np.broadcast_to(np.eye(n_outputs), (n_items, n_outputs, n_outputs))
    blocks: list[list[NDArray[np.float64]]] = [[] for _ in coefs]
    for index in range(len(coefs) - 1, -1, -1):
        previous = layers[index]
        weight_block = np.einsum("ni,noj->noij", previous, delta)
        blocks[index] = [
            weight_block.reshape(n_items * n_outputs, -1),
            np.ascontiguousarray(delta).reshape(n_items * n_outputs, -1),
        ]
        if index > 0:
            delta = (delta @ coefs[index].T) * (1.0 - previous**2)[:, None, :]
    jacobian = np.hstack([part for layer in blocks for part in layer])
    return output, jacobian


def train_levenberg_marquardt(
    X: ArrayLike,
    Y: ArrayLike,
    hidden_layer_sizes: Sequence[int],
    *,
    X_val: ArrayLike | None = None,
    Y_val: ArrayLike | None = None,
    alpha: float = 0.0,
    max_epochs: int = 1000,
    patience: int = 6,
    min_gradient: float = 1e-7,
    initial_mu: float = 1e-3,
    mu_decrease: float = 0.1,
    mu_increase: float = 10.0,
    mu_max: float = 1e10,
    memory_limit_mb: float = 512.0,
    rng: np.random.Generator | int | None = None,
) -> LevenbergMarquardtResult:
    """Train a tanh MLP with Levenberg-Marquardt and validation early stopping.

    Parameters
    ----------
    X, Y : array_like, shapes (n_items, n_inputs) and (n_items, n_outputs)
        Training inputs and targets (both best scaled to about ``[-1, 1]``).
    hidden_layer_sizes : sequence of int
        Width of each hidden layer.
    X_val, Y_val : array_like, optional
        Validation data for early stopping; without them training runs until
        another criterion stops it and the final weights are kept.
    alpha : float, optional
        L2 penalty on the weights, on the scale of scikit-learn's ``alpha``:
        the objective is ``sum(e**2) + alpha * n_outputs * sum(W**2)``.
    max_epochs : int, optional
        Upper bound on epochs.
    patience : int, optional
        Epochs without validation improvement before stopping.
    min_gradient : float, optional
        Stop when the gradient norm falls below this.
    initial_mu, mu_decrease, mu_increase, mu_max : float, optional
        Damping schedule.
    memory_limit_mb : float, optional
        Refuse to train when :func:`lm_memory_mb` exceeds this.
    rng : numpy.random.Generator or int, optional
        Randomness of the initial weights.

    Returns
    -------
    LevenbergMarquardtResult
        Best weights and the training history.

    Raises
    ------
    ConfigError
        When the problem is too large for ``memory_limit_mb``.

    References
    ----------
    .. [1] K. Levenberg, "A method for the solution of certain non-linear
       problems in least squares," Quarterly of Applied Mathematics, vol. 2,
       no. 2, pp. 164-168, 1944.
    .. [2] D. W. Marquardt, "An algorithm for least-squares estimation of
       nonlinear parameters," Journal of the Society for Industrial and
       Applied Mathematics, vol. 11, no. 2, pp. 431-441, 1963.
    .. [3] M. T. Hagan and M. B. Menhaj, "Training feedforward networks with
       the Marquardt algorithm," IEEE Transactions on Neural Networks, vol. 5,
       no. 6, pp. 989-993, 1994.
    .. [4] L. Prechelt, "Early stopping -- but when?," in Neural Networks:
       Tricks of the Trade, G. B. Orr and K.-R. Müller, Eds., Lecture Notes in
       Computer Science, vol. 1524. Berlin, Germany: Springer, 1998, pp. 55-69.

    Examples
    --------
    >>> X = np.linspace(-1.0, 1.0, 40)[:, None]
    >>> result = train_levenberg_marquardt(X, np.sin(2.5 * X), (6,), max_epochs=200, rng=0)
    >>> result.training_curve[-1] < 1e-4, result.stop_reason in ("maximum epochs reached",
    ...     "minimum gradient reached", "maximum mu reached")
    (True, True)
    """
    rng = np.random.default_rng(rng)
    inputs = np.asarray(X, dtype=np.float64)
    targets = np.asarray(Y, dtype=np.float64)
    if inputs.ndim != 2 or targets.ndim != 2 or inputs.shape[0] != targets.shape[0]:
        raise ValueError(f"X {inputs.shape} and Y {targets.shape} must be 2-D with matching rows")
    sizes = layer_sizes(inputs.shape[1], hidden_layer_sizes, targets.shape[1])
    n_params = n_parameters(sizes)
    needed = lm_memory_mb(targets.size, n_params)
    if needed > memory_limit_mb:
        raise ConfigError(
            f"Levenberg-Marquardt needs about {needed:.0f} MB for {targets.size} residuals and "
            f"{n_params} parameters, above model.lm_memory_limit_mb={memory_limit_mb:g}; use a "
            "smaller network, fewer training items or model.solver='lbfgs'"
        )
    validating = X_val is not None and Y_val is not None and len(np.asarray(X_val)) > 0
    validation_inputs = np.asarray(X_val, dtype=np.float64) if validating else None
    validation_targets = np.asarray(Y_val, dtype=np.float64) if validating else None

    coefs, intercepts = nguyen_widrow_weights(sizes, rng)
    theta = pack_parameters(coefs, intercepts)
    weight_mask = pack_parameters(
        [np.ones_like(w) for w in coefs], [np.zeros_like(b) for b in intercepts]
    )
    # scikit-learn's loss is sum(e**2) / (2 N K) + alpha * sum(W**2) / (2 N); scaled
    # by 2 N K, the penalty weight is alpha * K.
    penalty = alpha * targets.shape[1] * weight_mask

    def objective(vector: NDArray[np.float64]) -> float:
        residual = network_forward(inputs, *unpack_parameters(vector, sizes)) - targets
        return float(np.sum(residual**2) + np.sum(penalty * vector**2))

    def validation_mse(vector: NDArray[np.float64]) -> float:
        assert validation_inputs is not None and validation_targets is not None
        output = network_forward(validation_inputs, *unpack_parameters(vector, sizes))
        return float(np.mean((output - validation_targets) ** 2))

    mu = float(initial_mu)
    training_curve = [objective(theta) / targets.size]
    validation_curve = [validation_mse(theta)] if validating else []
    best_theta, best_epoch = theta.copy(), 0
    best_validation = validation_curve[0] if validating else np.inf
    failures = 0
    stop_reason = "maximum epochs reached"
    epoch = 0
    diagonal = np.diag_indices(n_params)
    while epoch < max_epochs:
        output, jacobian = network_jacobian(inputs, *unpack_parameters(theta, sizes))
        residual = (output - targets).ravel()
        current = float(residual @ residual + np.sum(penalty * theta**2))
        gradient = jacobian.T @ residual + penalty * theta
        if float(np.linalg.norm(gradient)) < min_gradient:
            stop_reason = "minimum gradient reached"
            break
        normal = jacobian.T @ jacobian
        normal[diagonal] += penalty
        del jacobian
        accepted = False
        while mu <= mu_max:
            # Damp a copy on its diagonal: no dense identity matrix, so the normal
            # matrix and its damped copy are the only n_params**2 arrays alive.
            damped = normal.copy()
            damped[diagonal] += mu
            try:
                step = scipy.linalg.solve(damped, -gradient, assume_a="pos", overwrite_a=True)
            except (np.linalg.LinAlgError, ValueError):
                mu *= mu_increase
                continue
            candidate = theta + step
            value = objective(candidate)
            if value < current:
                theta, accepted = candidate, True
                mu *= mu_decrease
                break
            mu *= mu_increase
        if not accepted:
            stop_reason = "maximum mu reached"
            break
        epoch += 1
        training_curve.append(value / targets.size)
        if validating:
            score = validation_mse(theta)
            validation_curve.append(score)
            if score < best_validation:
                best_validation, best_theta, best_epoch, failures = score, theta.copy(), epoch, 0
            else:
                failures += 1
                if failures >= patience:
                    stop_reason = "validation stop"
                    break
    if not validating:
        best_theta, best_epoch = theta, epoch
    best_coefs, best_intercepts = unpack_parameters(best_theta, sizes)
    _LOGGER.debug("levenberg-marquardt: %d epoch(s), best %d, %s", epoch, best_epoch, stop_reason)
    return LevenbergMarquardtResult(
        coefs=best_coefs,
        intercepts=best_intercepts,
        n_epochs=epoch,
        best_epoch=best_epoch,
        stop_reason=stop_reason,
        training_curve=training_curve,
        validation_curve=validation_curve,
    )
