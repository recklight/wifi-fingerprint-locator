"""Fingerprint-matching baselines: nearest mean fingerprint and k nearest neighbors."""

from __future__ import annotations

import numpy as np
import pytest

from wifi_fingerprint_locator import Config, build_model
from wifi_fingerprint_locator.baselines import NearestReferencePoint


def test_nearest_reference_answers_with_a_surveyed_position(grid_features):
    model = NearestReferencePoint().fit(grid_features.X, grid_features.y)
    estimate = model.predict(grid_features.X)
    surveyed = {tuple(point) for point in grid_features.y}
    assert all(tuple(point) in surveyed for point in estimate)
    assert np.mean(np.all(estimate == grid_features.y, axis=1)) > 0.95


def test_nearest_reference_uses_mean_fingerprints():
    X = np.array([[0.0], [2.0], [10.0]])
    y = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0]])
    model = NearestReferencePoint().fit(X, y)
    assert model.centroids_.ravel().tolist() == [1.0, 10.0]
    assert model.predict([[6.0]]).tolist() == [[5.0, 5.0]]


def test_nearest_reference_checks_its_input(grid_features):
    model = NearestReferencePoint().fit(grid_features.X, grid_features.y)
    with pytest.raises(ValueError, match="shaped"):
        model.predict(grid_features.X[:, :3])


@pytest.mark.parametrize("name", ["knn", "weighted_knn", "nearest_reference"])
def test_baselines_locate_known_reference_points(grid_features, name):
    config = Config().override({"model.name": name, "model.n_neighbors": 3})
    model = build_model(config).fit(grid_features.X, grid_features.y)
    errors = np.hypot(*(model.predict(grid_features.X) - grid_features.y).T)
    assert np.median(errors) < 0.3, name
