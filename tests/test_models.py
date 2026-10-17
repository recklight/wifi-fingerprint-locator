"""Model registry and model files."""

from __future__ import annotations

import joblib
import numpy as np
import pytest

from wifi_fingerprint_locator import (
    MODEL_NAMES,
    Config,
    ConfigError,
    ModelFileError,
    build_model,
    load_model,
    save_model,
)
from wifi_fingerprint_locator.models import MODEL_FORMAT


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_is_a_scaling_pipeline(name):
    model = build_model(Config().override({"model.name": name}))
    assert [step for step, _ in model.steps] == ["drop_constant", "scale", "regressor"]


def test_mlp_settings_reach_the_network(fast_config):
    config = fast_config.override({"model.alpha": 0.5, "seed": 9})
    network = build_model(config).named_steps["regressor"].regressor
    assert network.alpha == 0.5 and network.random_state == 9
    assert tuple(network.hidden_layer_sizes) == (16, 8)


def test_constant_features_are_dropped_and_targets_scaled(grid_features, fast_config):
    X = np.hstack([grid_features.X, np.ones((grid_features.n_items, 1))])
    model = build_model(fast_config).fit(X, grid_features.y)
    assert model.named_steps["drop_constant"].get_support().sum() == grid_features.n_features
    errors = np.hypot(*(model.predict(X) - grid_features.y).T)
    assert np.median(errors) < 0.5


def test_unknown_model_name_is_reported():
    config = Config()
    object.__setattr__(config.model, "name", "forest")
    with pytest.raises(ConfigError, match="unknown model"):
        build_model(config)


def test_model_files_round_trip(tmp_path, grid_features, fast_config):
    model = build_model(fast_config).fit(grid_features.X, grid_features.y)
    path = save_model(tmp_path / "model.joblib", model, fast_config)
    bundle = load_model(path)
    assert bundle["format"] == MODEL_FORMAT
    assert Config.from_dict(bundle["config"]) == fast_config
    assert np.array_equal(
        bundle["estimator"].predict(grid_features.X), model.predict(grid_features.X)
    )


def test_foreign_or_missing_model_files_are_rejected(tmp_path):
    joblib.dump({"format": "something-else"}, tmp_path / "other.joblib")
    with pytest.raises(ModelFileError, match="expected"):
        load_model(tmp_path / "other.joblib")
    (tmp_path / "garbage.joblib").write_bytes(b"not a model")
    with pytest.raises(ModelFileError, match="not a readable model"):
        load_model(tmp_path / "garbage.joblib")
    with pytest.raises(FileNotFoundError, match="Run 'train' first"):
        load_model(tmp_path / "absent.joblib")
