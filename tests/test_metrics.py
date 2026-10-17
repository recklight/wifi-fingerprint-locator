"""Localization metrics: Euclidean versus per-axis errors, the CDF, scikit-learn references."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from wifi_fingerprint_locator.metrics import (
    METRIC_NAMES,
    empirical_cdf,
    localization_scores,
    nearest_reference,
    position_errors,
    summarize_errors,
    summarize_folds,
)


def test_euclidean_error_is_not_the_per_axis_error():
    truth = np.array([[0.0, 0.0]])
    estimate = np.array([[3.0, 4.0]])
    assert position_errors(truth, estimate).tolist() == [5.0]
    scores = localization_scores(truth, estimate)
    assert scores["mean_error_m"] == 5.0
    assert scores["mae"] == 3.5 and scores["mae_x_m"] == 3.0 and scores["mae_y_m"] == 4.0


def test_scores_agree_with_scikit_learn_and_scipy(rng):
    truth = rng.uniform(0.0, 7.0, (60, 2))
    estimate = truth + rng.normal(0.0, 0.5, (60, 2))
    scores = localization_scores(truth, estimate)
    assert np.isclose(scores["mae"], mean_absolute_error(truth, estimate), rtol=1e-10)
    assert np.isclose(scores["rmse"], np.sqrt(mean_squared_error(truth, estimate)), rtol=1e-10)
    assert np.isclose(scores["r2"], r2_score(truth, estimate), rtol=1e-10)
    pearson = np.mean([stats.pearsonr(truth[:, k], estimate[:, k])[0] for k in range(2)])
    assert np.isclose(scores["pearson_r"], pearson, rtol=1e-10)
    errors = np.hypot(*(estimate - truth).T)
    assert np.isclose(scores["p90_error_m"], np.percentile(errors, 90), rtol=1e-10)
    assert np.isclose(scores["rmse_error_m"], np.sqrt(np.mean(errors**2)), rtol=1e-10)
    assert list(scores) == list(METRIC_NAMES)


def test_cdf_is_monotone_and_ends_at_one(rng):
    errors = rng.exponential(1.0, 500)
    thresholds, fractions = empirical_cdf(errors, step_m=0.2)
    assert np.all(np.diff(fractions) >= 0)
    assert fractions[0] >= 0 and fractions[-1] == 1.0
    assert np.allclose(np.diff(thresholds), 0.2)


def test_cdf_counts_errors_at_or_below_each_threshold():
    _, fractions = empirical_cdf([0.0, 0.2, 0.4, 0.4], step_m=0.2)
    assert fractions.tolist() == [0.25, 0.5, 1.0]


def test_cdf_can_stop_early():
    thresholds, fractions = empirical_cdf([0.1, 5.0], step_m=0.5, max_m=1.0)
    assert thresholds.tolist() == [0.0, 0.5, 1.0] and fractions.tolist() == [0.0, 0.5, 0.5]


def test_grid_scores_snap_estimates_to_reference_points():
    references = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    truth = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    estimate = np.array([[0.2, 0.1], [1.6, 0.0], [2.1, 0.0], [1.9, 0.0]])
    scores = localization_scores(truth, estimate, references)
    assert scores["grid_accuracy"] == 0.75
    # Recall per true point: 1, 0, 1 -> unweighted mean 2/3.
    assert scores["grid_uar"] == pytest.approx(2 / 3)
    assert nearest_reference(estimate, references).tolist() == [0, 2, 2, 2]


def test_constant_coordinates_give_nan_instead_of_crashing():
    truth = np.array([[1.0, 1.0], [1.0, 1.0]])
    scores = localization_scores(truth, truth + 0.1)
    assert np.isnan(scores["r2"]) and np.isnan(scores["pearson_r"])


def test_summary_statistics_of_errors():
    summary = summarize_errors(np.arange(101.0))
    assert summary["median_error_m"] == 50.0 and summary["p90_error_m"] == 90.0
    assert summary["max_error_m"] == 100.0


def test_fold_summary_uses_the_sample_standard_deviation():
    summary = summarize_folds([{"mae": 1.0}, {"mae": 2.0}, {"mae": 6.0}])
    assert summary["mae"]["mean"] == 3.0
    assert summary["mae"]["std"] == pytest.approx(np.std([1.0, 2.0, 6.0], ddof=1))


def test_empty_or_misshapen_inputs_are_rejected():
    with pytest.raises(ValueError, match="empty"):
        summarize_errors([])
    with pytest.raises(ValueError, match="shaped"):
        position_errors(np.zeros((3, 3)), np.zeros((3, 3)))
