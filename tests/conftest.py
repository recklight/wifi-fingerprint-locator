"""Shared fixtures. Every fixture is synthesized in code; no recorded data is read."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wifi_fingerprint_locator import (
    Config,
    FeatureSet,
    prepare_features,
    read_survey,
    write_synthetic_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def fast_config():
    return Config.from_yaml(PROJECT_ROOT / "configs" / "quick_demo.yaml")


@pytest.fixture
def synthetic_dataset(tmp_path, fast_config):
    return write_synthetic_dataset(tmp_path / "data", fast_config.synthetic, rng=0)


@pytest.fixture
def survey(synthetic_dataset, fast_config):
    return read_survey(synthetic_dataset, fast_config)


@pytest.fixture
def features(synthetic_dataset, fast_config):
    return prepare_features(synthetic_dataset, fast_config)


@pytest.fixture
def grid_features(rng):
    """A smooth, learnable fingerprint field on a 4 x 3 grid, 15 packets per point."""
    xs, ys = np.meshgrid(np.arange(4.0), np.arange(3.0), indexing="ij")
    points = np.column_stack([xs.ravel(), ys.ravel()])
    coordinates = np.repeat(points, 15, axis=0)
    anchors = rng.uniform(-1.0, 4.0, size=(6, 2))
    distances = np.linalg.norm(coordinates[:, None, :] - anchors[None, :, :], axis=-1)
    X = np.hstack([distances, np.log1p(distances)]) + rng.normal(0.0, 0.02, (len(coordinates), 12))
    groups = np.repeat([f"point_{k:02d}" for k in range(len(points))], 15)
    paths = np.repeat([f"items/item_{k:03d}.dat" for k in range(len(points))], 15)
    segments = np.tile(np.arange(15), len(points))
    return FeatureSet(X, coordinates, groups, paths, Config().to_dict(), segments)
