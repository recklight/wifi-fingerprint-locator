"""The MLP position regressor: learning, early stopping, penalty selection, estimator contract."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from sklearn.base import clone

from wifi_fingerprint_locator.position_regressor import SOLVERS, PositionRegressor
from wifi_fingerprint_locator.splits import group_codes


@pytest.fixture
def smooth_problem(rng):
    X = rng.uniform(-1.0, 1.0, (240, 4))
    y = np.column_stack([np.tanh(X[:, 0] + 0.5 * X[:, 1]), 0.6 * X[:, 2] - 0.3 * X[:, 3]])
    return X, y


@pytest.mark.parametrize("solver", SOLVERS)
def test_every_solver_learns_a_smooth_mapping(smooth_problem, solver):
    X, y = smooth_problem
    model = PositionRegressor((10,), solver=solver, max_epochs=300, random_state=0)
    model.fit(X, y)
    assert np.mean((model.predict(X) - y) ** 2) < 0.02, solver
    assert model.n_epochs_ >= model.best_epoch_ >= 0


def test_patience_stops_training_and_restores_the_best_check(rng):
    X = rng.uniform(-1.0, 1.0, (200, 6))
    y = rng.normal(size=(200, 2))
    model = PositionRegressor((40, 20), max_epochs=1000, epochs_per_check=5, patience=2)
    model.fit(X, y)
    assert model.stop_reason_ == "validation stop"
    assert (
        len(model.validation_curve_)
        == model.validation_curve_.index(min(model.validation_curve_)) + 3
    )
    assert model.best_epoch_ < model.n_epochs_


def test_without_validation_the_full_budget_is_used(smooth_problem):
    X, y = smooth_problem
    model = PositionRegressor((6,), max_epochs=20, patience=0).fit(X, y)
    assert model.validation_curve_ == [] and model.n_epochs_ <= 20


def test_alpha_grid_keeps_the_lowest_validation_error(smooth_problem):
    X, y = smooth_problem
    model = PositionRegressor((8,), alpha_grid=(10.0, 1e-4), max_epochs=100).fit(X, y)
    assert set(model.validation_scores_) == {10.0, 1e-4}
    assert model.alpha_ == min(model.validation_scores_, key=model.validation_scores_.get)
    assert model.alpha_ == 1e-4


def test_alpha_grid_without_a_validation_set_is_refused(smooth_problem):
    X, y = smooth_problem
    with pytest.raises(ValueError, match="alpha_grid"):
        PositionRegressor((4,), alpha_grid=(1.0, 1e-4), patience=0).fit(X, y)


def test_groups_hold_whole_reference_points_out_for_early_stopping(grid_features):
    codes = group_codes(grid_features.groups)
    model = PositionRegressor(validation_fraction=0.25)
    fit_rows, val_rows = model._split(codes.size, np.random.default_rng(7), codes)
    assert set(codes[val_rows]).isdisjoint(codes[fit_rows])
    assert np.unique(codes[val_rows]).size == 3  # a quarter of the 12 points
    assert np.array_equal(np.sort(np.concatenate([fit_rows, val_rows])), np.arange(codes.size))
    again = model._split(codes.size, np.random.default_rng(7), codes)
    assert np.array_equal(again[0], fit_rows) and np.array_equal(again[1], val_rows)


def test_fit_accepts_one_group_per_item(grid_features):
    X, y, groups = grid_features.X, grid_features.y, grid_features.groups
    model = PositionRegressor((6,), max_epochs=30).fit(X, y, groups=groups)
    assert model.validation_curve_
    with pytest.raises(ValueError, match="one entry per item"):
        PositionRegressor((6,), max_epochs=30).fit(X, y, groups=groups[:-1])


def test_too_few_reference_points_fall_back_to_random_packets(caplog):
    codes = np.repeat([0, 1], 10)
    with caplog.at_level(logging.WARNING):
        fit_rows, val_rows = PositionRegressor()._split(20, np.random.default_rng(0), codes)
    assert val_rows.size == 3 and fit_rows.size == 17
    assert "too few" in caplog.text


def test_adam_is_one_uninterrupted_run_whatever_the_check_interval(smooth_problem):
    # The optimizer state (moments, bias-correction step) must survive the
    # validation checks, so checking every epoch or every fourth epoch has to
    # trace the same trajectory; a restarted optimizer would not.
    X, y = smooth_problem
    curves = {}
    for interval in (1, 4):
        model = PositionRegressor(
            (8,),
            solver="adam",
            max_epochs=12,
            epochs_per_check=interval,
            patience=100,
            learning_rate=0.01,
            random_state=2,
        ).fit(X, y)
        curves[interval] = model.validation_curve_
    assert len(curves[1]) == 12 and len(curves[4]) == 3
    assert np.allclose(curves[4], curves[1][3::4], rtol=1e-12)


def test_adam_and_lbfgs_runs_are_reproducible(smooth_problem):
    X, y = smooth_problem
    for solver in ("adam", "lbfgs"):
        runs = [
            PositionRegressor((6,), solver=solver, max_epochs=20, random_state=4).fit(X, y)
            for _ in range(2)
        ]
        assert np.array_equal(runs[0].predict(X), runs[1].predict(X)), solver


def test_the_same_seed_gives_the_same_network(smooth_problem):
    X, y = smooth_problem
    first = PositionRegressor((8,), max_epochs=50, random_state=3).fit(X, y).predict(X)
    second = PositionRegressor((8,), max_epochs=50, random_state=3).fit(X, y).predict(X)
    assert np.array_equal(first, second)


def test_one_dimensional_targets_stay_one_dimensional(smooth_problem):
    X, y = smooth_problem
    model = PositionRegressor((6,), max_epochs=30).fit(X, y[:, 0])
    assert model.predict(X).shape == (X.shape[0],)


def test_estimator_contract(smooth_problem):
    X, y = smooth_problem
    model = PositionRegressor((5,), solver="adam", alpha=0.01)
    assert clone(model).get_params()["alpha"] == 0.01
    with pytest.raises(ValueError, match="unknown solver"):
        PositionRegressor(solver="sgd").fit(X, y)
    fitted = PositionRegressor((5,), max_epochs=10).fit(X, y)
    with pytest.raises(ValueError, match="shaped"):
        fitted.predict(X[:, :2])
    with pytest.raises(ValueError, match="finite"):
        PositionRegressor().fit(np.full((10, 2), np.nan), np.zeros((10, 2)))
