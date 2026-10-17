"""Cross-validation: out-of-fold coverage, group integrity, metric definitions, reproducibility."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error

from wifi_fingerprint_locator import Config, PositionRegressor, cross_validate
from wifi_fingerprint_locator.evaluation import aggregate_recordings
from wifi_fingerprint_locator.metrics import METRIC_NAMES


@pytest.fixture
def knn_config():
    return Config().override(
        {"model.name": "knn", "model.n_neighbors": 3, "evaluation.n_splits": 3}
    )


def test_every_item_has_exactly_one_out_of_fold_estimate(grid_features, knn_config):
    result = cross_validate(grid_features, knn_config)
    predictions = result.predictions
    assert len(predictions) == grid_features.n_items
    assert predictions["fold"].between(0, 2).all()
    assert predictions[["predicted_x_m", "predicted_y_m"]].notna().all().all()
    assert predictions["path"].tolist() == grid_features.paths.tolist()


@pytest.mark.parametrize("strategy", ["location_holdout", "location_block"])
def test_location_folds_never_split_a_group(grid_features, knn_config, strategy):
    config = knn_config.override({"evaluation.split_strategy": strategy})
    predictions = cross_validate(grid_features, config).predictions
    assert (predictions.groupby("group")["fold"].nunique() == 1).all()


def test_fold_metrics_match_the_reference_implementation(grid_features, knn_config):
    result = cross_validate(grid_features, knn_config)
    predictions = result.predictions
    for fold, row in result.folds.iterrows():
        part = predictions[predictions["fold"] == fold]
        truth = part[["x_m", "y_m"]].to_numpy()
        estimate = part[["predicted_x_m", "predicted_y_m"]].to_numpy()
        assert np.isclose(row["mae"], mean_absolute_error(truth, estimate), rtol=1e-10)
        assert np.isclose(row["mean_error_m"], part["error_m"].mean(), rtol=1e-10)
        assert row["n_test"] == len(part)


def test_summary_uses_the_sample_standard_deviation(grid_features, knn_config):
    result = cross_validate(grid_features, knn_config)
    values = result.folds["mean_error_m"].to_numpy()
    summary = result.metrics["summary"]["mean_error_m"]
    assert np.isclose(summary["mean"], values.mean())
    assert np.isclose(summary["std"], values.std(ddof=1))


def test_metrics_document_has_the_fixed_layout(grid_features, knn_config):
    metrics = cross_validate(grid_features, knn_config).metrics
    assert list(metrics) == [
        "task",
        "n_items",
        "n_splits",
        "seed",
        "folds",
        "summary",
        "pooled",
        "recordings",
    ]
    assert metrics["task"] == "regression" and metrics["n_splits"] == 3
    assert list(metrics["summary"]) == list(METRIC_NAMES)
    assert list(metrics["folds"][0])[:3] == ["fold", "n_train", "n_test"]
    assert list(metrics["recordings"]) == ["n_items", "folds", "summary", "pooled"]
    assert metrics["recordings"]["n_items"] == 12


def test_packet_split_is_more_optimistic_than_location_split(grid_features, knn_config):
    packet = cross_validate(
        grid_features, knn_config.override({"evaluation.split_strategy": "packet_random"})
    )
    location = cross_validate(grid_features, knn_config)
    assert packet.metrics["pooled"]["mean_error_m"] < location.metrics["pooled"]["mean_error_m"]


def test_location_splits_also_hold_reference_points_out_for_early_stopping(
    features, fast_config, monkeypatch
):
    splits = []
    split = PositionRegressor._split

    def recording_split(self, n_items, rng, groups=None):
        fit_rows, val_rows = split(self, n_items, rng, groups)
        splits.append((groups, fit_rows, val_rows))
        return fit_rows, val_rows

    monkeypatch.setattr(PositionRegressor, "_split", recording_split)
    config = fast_config.override({"model.max_epochs": 20})
    cross_validate(features, config)  # default split: location_holdout
    assert len(splits) == config.evaluation.n_splits
    for groups, fit_rows, val_rows in splits:
        assert groups is not None and val_rows.size > 0
        assert set(groups[val_rows]).isdisjoint(groups[fit_rows])
    splits.clear()
    cross_validate(features, config.override({"evaluation.split_strategy": "packet_random"}))
    assert splits and all(groups is None for groups, _, _ in splits)


def test_the_same_seed_gives_identical_results(features, fast_config):
    first = cross_validate(features, fast_config)
    second = cross_validate(features, fast_config)
    assert first.predictions.equals(second.predictions)
    assert first.metrics == second.metrics


def test_recordings_average_the_packets_of_each_capture(grid_features, knn_config):
    result = cross_validate(grid_features, knn_config)
    recordings = result.recording_predictions
    first = result.predictions[result.predictions["path"] == recordings["path"].iloc[0]]
    assert np.isclose(recordings["predicted_x_m"].iloc[0], first["predicted_x_m"].mean())
    assert recordings["n_packets"].iloc[0] == len(first)
    assert aggregate_recordings(result.predictions).equals(recordings)
