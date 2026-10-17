"""The representation x split-strategy comparison matrix."""

from __future__ import annotations

import pytest

from wifi_fingerprint_locator import compare_representations
from wifi_fingerprint_locator.metrics import METRIC_NAMES


@pytest.fixture
def comparison(survey, fast_config):
    config = fast_config.override({"model.name": "weighted_knn"})
    return compare_representations(
        survey,
        config,
        representations=("amplitude", "real_imag"),
        strategies=("packet_random", "location_holdout"),
    )


def test_every_combination_has_every_fold(comparison, fast_config):
    assert len(comparison.summary) == 4
    assert len(comparison.folds) == 4 * fast_config.evaluation.n_splits
    assert set(comparison.folds["representation"]) == {"amplitude", "real_imag"}
    for name in METRIC_NAMES:
        assert f"{name}_mean" in comparison.summary and f"{name}_std" in comparison.summary


def test_errors_cover_every_packet_once_per_combination(comparison, survey):
    counts = comparison.errors.groupby(["representation", "strategy"]).size()
    assert (counts == survey.n_items).all()


def test_packet_split_beats_location_split_and_amplitude_beats_raw_real_imag(comparison):
    summary = comparison.summary.set_index(["representation", "strategy"])["mean_error_m_mean"]
    assert summary[("amplitude", "packet_random")] < summary[("amplitude", "location_holdout")]
    assert summary[("amplitude", "packet_random")] < summary[("real_imag", "packet_random")]
    assert comparison.best()["strategy"] == "packet_random"


def test_json_view_is_nested_by_strategy_then_representation(comparison):
    nested = comparison.as_dict()
    assert set(nested) == {"packet_random", "location_holdout"}
    cell = nested["packet_random"]["amplitude"]
    assert cell["n_folds"] == 3 and set(cell["mean_error_m"]) == {"mean", "std"}


def test_matrix_defaults_come_from_the_configuration(survey, fast_config):
    config = fast_config.override(
        {
            "model.name": "knn",
            "evaluation.compare_representations": ["sanitized_phase"],
            "evaluation.compare_strategies": ["temporal"],
        }
    )
    result = compare_representations(survey, config)
    assert result.summary[["representation", "strategy"]].values.tolist() == [
        ["sanitized_phase", "temporal"]
    ]
