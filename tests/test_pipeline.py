"""High-level pipeline: prepare, train, predict and compare on a synthetic survey."""

from __future__ import annotations

import numpy as np
import pytest

from wifi_fingerprint_locator import (
    Config,
    load_model,
    predict_inputs,
    prepare_features,
    prepare_survey,
    run_comparison,
    save_model,
    train_model,
)
from wifi_fingerprint_locator.dataset import load_dataset_index


def test_prepare_builds_one_fingerprint_per_kept_packet(synthetic_dataset, fast_config):
    features, report = prepare_survey(synthetic_dataset, fast_config)
    assert features.n_items == report.n_output <= report.n_input == 240
    assert features.n_features == 90
    assert np.unique(features.groups).size == 12
    assert features.config == fast_config.to_dict()


def test_prepare_respects_the_representation(synthetic_dataset, fast_config):
    config = fast_config.override({"features.representation": "amplitude_phase"})
    assert prepare_features(synthetic_dataset, config).n_features == 180


def test_trained_model_locates_known_points(features, fast_config):
    model = train_model(features, fast_config)
    errors = np.hypot(*(model.predict(features.X) - features.y).T)
    assert np.median(errors) < 0.5


def test_prediction_uses_only_the_stored_configuration(tmp_path, synthetic_dataset, features):
    config = Config.from_dict(features.config)
    model = train_model(features, config)
    bundle = load_model(save_model(tmp_path / "model.joblib", model, config))
    index = load_dataset_index(synthetic_dataset)
    inputs = index["path"].tolist()[:2]
    table = predict_inputs(inputs, bundle)
    assert table.columns.tolist() == ["path", "segment", "predicted_x_m", "predicted_y_m"]
    assert list(dict.fromkeys(table["path"])) == [p.replace("\\", "/") for p in inputs]
    first = table[table["path"] == table["path"].iloc[0]]
    estimate = first[["predicted_x_m", "predicted_y_m"]].mean().to_numpy()
    assert np.linalg.norm(estimate - index[["x_m", "y_m"]].to_numpy()[0]) < 0.5


def test_prediction_needs_existing_inputs(tmp_path, features, fast_config):
    bundle = {"config": fast_config.to_dict(), "estimator": train_model(features, fast_config)}
    with pytest.raises(FileNotFoundError, match="capture file not found"):
        predict_inputs([tmp_path / "absent.dat"], bundle)
    with pytest.raises(ValueError, match="no input"):
        predict_inputs([], bundle)


def test_run_comparison_reads_the_survey_once(synthetic_dataset, fast_config):
    config = fast_config.override(
        {
            "model.name": "knn",
            "evaluation.compare_representations": ["amplitude", "sanitized_phase"],
            "evaluation.compare_strategies": ["temporal"],
        }
    )
    result = run_comparison(synthetic_dataset, config)
    assert result.summary["representation"].tolist() == ["amplitude", "sanitized_phase"]
