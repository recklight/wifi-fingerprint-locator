"""Levenberg-Marquardt training: exact Jacobian, convergence, early stopping, memory guard."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.neural_network import MLPRegressor

from wifi_fingerprint_locator import levenberg_marquardt
from wifi_fingerprint_locator.errors import ConfigError
from wifi_fingerprint_locator.levenberg_marquardt import (
    layer_sizes,
    lm_memory_mb,
    n_parameters,
    network_forward,
    network_jacobian,
    nguyen_widrow_weights,
    pack_parameters,
    train_levenberg_marquardt,
    unpack_parameters,
)


def test_jacobian_matches_central_finite_differences(rng):
    sizes = [4, 5, 3, 2]
    coefs, intercepts = nguyen_widrow_weights(sizes, rng=0)
    X = rng.normal(size=(7, 4))
    _, jacobian = network_jacobian(X, coefs, intercepts)
    theta = pack_parameters(coefs, intercepts)
    assert jacobian.shape == (7 * 2, n_parameters(sizes))
    step = 1e-6
    numeric = np.empty_like(jacobian)
    for k in range(theta.size):
        delta = np.zeros_like(theta)
        delta[k] = step
        plus = network_forward(X, *unpack_parameters(theta + delta, sizes))
        minus = network_forward(X, *unpack_parameters(theta - delta, sizes))
        numeric[:, k] = ((plus - minus) / (2 * step)).ravel()
    assert np.allclose(jacobian, numeric, atol=1e-7)


def test_parameter_packing_round_trips():
    coefs, intercepts = nguyen_widrow_weights([3, 4, 2], rng=1)
    again = unpack_parameters(pack_parameters(coefs, intercepts), [3, 4, 2])
    for left, right in zip(coefs + intercepts, again[0] + again[1], strict=True):
        assert np.array_equal(left, right)
    with pytest.raises(ValueError, match="holds"):
        unpack_parameters(np.zeros(5), [3, 4, 2])


def test_nguyen_widrow_hidden_weights_have_the_prescribed_length():
    coefs, intercepts = nguyen_widrow_weights([6, 10, 2], rng=2)
    beta = 0.7 * 10 ** (1 / 6)
    assert np.allclose(np.linalg.norm(coefs[0], axis=0), beta)
    assert np.all(np.abs(intercepts[0]) <= beta)


def test_small_network_fits_a_smooth_two_output_function():
    X = np.linspace(-1.0, 1.0, 60)[:, None]
    Y = np.column_stack([np.sin(2.0 * X[:, 0]), X[:, 0] ** 2])
    result = train_levenberg_marquardt(X, Y, (8,), max_epochs=150, rng=0)
    assert result.training_curve[-1] < 1e-4
    assert result.training_curve[-1] < result.training_curve[0]
    prediction = network_forward(X, result.coefs, result.intercepts)
    assert np.mean((prediction - Y) ** 2) < 1e-4


def test_validation_early_stopping_restores_the_best_epoch(rng):
    X = rng.uniform(-1.0, 1.0, (40, 3))
    Y = rng.normal(size=(40, 1))  # pure noise: validation error soon stops improving
    X_val = rng.uniform(-1.0, 1.0, (20, 3))
    Y_val = rng.normal(size=(20, 1))
    result = train_levenberg_marquardt(
        X, Y, (12,), X_val=X_val, Y_val=Y_val, max_epochs=300, patience=4, rng=0
    )
    assert result.stop_reason == "validation stop"
    assert result.n_epochs == result.best_epoch + 4
    best = network_forward(X_val, result.coefs, result.intercepts)
    assert np.isclose(np.mean((best - Y_val) ** 2), min(result.validation_curve))


def test_weight_penalty_shrinks_the_weights(rng):
    X = rng.uniform(-1.0, 1.0, (50, 2))
    Y = (X[:, :1] * 3.0) ** 3
    plain = train_levenberg_marquardt(X, Y, (6,), max_epochs=60, rng=0)
    penalized = train_levenberg_marquardt(X, Y, (6,), alpha=1.0, max_epochs=60, rng=0)
    norm = lambda result: sum(float(np.sum(w**2)) for w in result.coefs)  # noqa: E731
    assert norm(penalized) < norm(plain)


def test_penalty_is_on_the_scale_of_scikit_learn_alpha(rng, monkeypatch):
    # At the same weights the objective must equal scikit-learn's MLP loss times
    # 2 * n_items * n_outputs, so one model.alpha means the same for every solver.
    # The training curve stores the objective divided by n_items * n_outputs,
    # which leaves the factor 2.
    X = rng.uniform(-1.0, 1.0, (30, 3))
    Y = rng.normal(size=(30, 2))
    reference = MLPRegressor(
        hidden_layer_sizes=(4,),
        activation="tanh",
        solver="lbfgs",
        alpha=0.5,
        max_iter=5,
        random_state=0,
    ).fit(X, Y)
    monkeypatch.setattr(
        levenberg_marquardt,
        "nguyen_widrow_weights",
        lambda sizes, rng: (reference.coefs_, reference.intercepts_),
    )
    result = train_levenberg_marquardt(X, Y, (4,), alpha=0.5, max_epochs=0)
    assert np.isclose(result.training_curve[0], 2.0 * reference.loss_, rtol=1e-9)


def test_memory_guard_refuses_oversized_problems():
    sizes = layer_sizes(90, (60, 30, 15), 2)
    assert n_parameters(sizes) == 7787
    assert lm_memory_mb(11_200 * 2, 7787) > 512
    X = np.zeros((300, 90))
    with pytest.raises(ConfigError, match="lm_memory_limit_mb"):
        train_levenberg_marquardt(X, np.zeros((300, 2)), (60, 30, 15), memory_limit_mb=10.0)
