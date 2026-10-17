"""Figures: every plot returns a Figure, tables carry the plotted numbers, file names keep dots."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from wifi_fingerprint_locator.figures import (
    SERIES_COLORS,
    SERIES_MARKERS,
    comparison_cdf_table,
    error_cdf_table,
    location_errors_table,
    new_figure,
    phase_sanitization_table,
    plot_comparison_cdf,
    plot_error_cdf,
    plot_location_errors,
    plot_phase_sanitization,
    plot_residuals,
    residuals_table,
    save_figure,
)


@pytest.fixture
def predictions(rng):
    truth = np.repeat(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), 5, axis=0)
    estimate = truth + rng.normal(0.0, 0.3, truth.shape)
    return pd.DataFrame(
        {
            "path": np.repeat(["a.dat", "b.dat", "c.dat"], 5),
            "segment": np.tile(np.arange(5), 3),
            "group": np.repeat(["p", "q", "r"], 5),
            "fold": np.repeat([0, 1, 2], 5),
            "x_m": truth[:, 0],
            "y_m": truth[:, 1],
            "predicted_x_m": estimate[:, 0],
            "predicted_y_m": estimate[:, 1],
        }
    )


@pytest.fixture
def comparison_errors(rng):
    return pd.DataFrame(
        {
            "representation": np.tile(np.repeat(["amplitude", "real_imag"], 20), 3),
            "strategy": np.repeat(["packet_random", "location_block", "temporal"], 40),
            "error_m": rng.exponential(1.0, 120),
        }
    )


def test_every_plot_returns_a_figure(predictions, comparison_errors):
    figures = [
        plot_error_cdf({"packets": [0.1, 0.5, 1.2], "captures": [0.3]}),
        plot_residuals(predictions),
        plot_location_errors(predictions),
        plot_comparison_cdf(comparison_errors),
        plot_phase_sanitization(np.zeros((2, 30)), np.ones((2, 30)), np.arange(30)),
    ]
    assert all(isinstance(figure, Figure) for figure in figures)


def test_comparison_figure_has_a_panel_per_strategy(comparison_errors):
    visible = [axes for axes in plot_comparison_cdf(comparison_errors).axes if axes.get_visible()]
    assert [axes.get_title() for axes in visible] == ["packet_random", "location_block", "temporal"]


def test_tables_have_the_documented_columns(predictions, comparison_errors):
    assert error_cdf_table({"a": [0.1]}).columns.tolist() == ["curve", "threshold_m", "fraction"]
    assert residuals_table(predictions).columns.tolist() == [
        "path",
        "segment",
        "fold",
        "dx_m",
        "dy_m",
        "error_m",
    ]
    assert location_errors_table(predictions).columns.tolist() == [
        "group",
        "x_m",
        "y_m",
        "n_items",
        "mean_error_m",
        "median_error_m",
        "mean_dx_m",
        "mean_dy_m",
    ]
    assert comparison_cdf_table(comparison_errors).columns.tolist() == [
        "strategy",
        "representation",
        "threshold_m",
        "fraction",
    ]
    table = phase_sanitization_table(np.zeros((2, 3)), np.ones((2, 3)), [0, 1, 2])
    assert table.columns.tolist() == [
        "packet",
        "subcarrier",
        "raw_phase_rad",
        "sanitized_phase_rad",
    ]


def test_cdf_curves_share_thresholds_and_end_at_one(comparison_errors):
    table = comparison_cdf_table(comparison_errors, step_m=0.5)
    sizes = table.groupby(["strategy", "representation"]).size()
    assert sizes.nunique() == 1
    assert (table.groupby(["strategy", "representation"])["fraction"].max() == 1.0).all()


def test_location_table_averages_per_reference_point(predictions):
    table = location_errors_table(predictions)
    assert table["n_items"].tolist() == [5, 5, 5]
    part = predictions[predictions["group"] == "q"]
    expected = np.hypot(part["predicted_x_m"] - 1.0, part["predicted_y_m"]).mean()
    assert np.isclose(table.loc[1, "mean_error_m"], expected)


def test_save_figure_keeps_dots_in_the_stem(tmp_path):
    path = save_figure(new_figure(3, 2, 80), tmp_path / "error.v2", "png")
    assert path.name == "error.v2.png" and path.stat().st_size > 0


def test_series_styles_have_the_same_length():
    assert len(SERIES_COLORS) == len(SERIES_MARKERS) == 5
