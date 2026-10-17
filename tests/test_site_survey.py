"""Survey assembly: packet windows, stream and antenna selection, cleaning and indexing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wifi_fingerprint_locator import Config
from wifi_fingerprint_locator.dataset import load_dataset_index, write_dataset_index
from wifi_fingerprint_locator.errors import DatasetError
from wifi_fingerprint_locator.site_survey import (
    index_capture_folder,
    read_capture_channels,
    read_survey,
    survey_features,
)
from wifi_fingerprint_locator.synthetic import make_capture, make_room


@pytest.fixture
def capture(tmp_path, fast_config):
    room = make_room(fast_config.synthetic, rng=0)
    path = tmp_path / "capture.dat"
    make_capture(path, room, (1.0, 1.0), fast_config.synthetic, n_packets=30, rng=1)
    return path


def test_packet_window_selects_reports_by_position(capture, fast_config):
    config = fast_config.override({"survey.packet_start": 10, "survey.n_packets": 15})
    channels = read_capture_channels(capture, config.survey, config.scaling)
    assert channels.segments.tolist() == list(range(10, 25))
    assert channels.csi.shape == (15, 3, 30)


def test_a_short_capture_is_used_with_a_warning(capture, fast_config, caplog):
    config = fast_config.override({"survey.n_packets": 100})
    channels = read_capture_channels(capture, config.survey, config.scaling)
    assert channels.n_packets == 30
    assert "window asks for 100" in caplog.text


def test_window_past_the_end_is_an_error(capture, fast_config):
    config = fast_config.override({"survey.packet_start": 40})
    with pytest.raises(DatasetError, match="no report in the window"):
        read_capture_channels(capture, config.survey, config.scaling)


def test_prediction_reads_every_report(capture, fast_config):
    config = fast_config.override({"survey.n_packets": 5})
    channels = read_capture_channels(capture, config.survey, config.scaling, use_window=False)
    assert channels.n_packets == 30


def test_antenna_selection_and_raw_units(capture, fast_config):
    config = fast_config.override({"survey.rx_indices": [2, 0], "scaling.to_snr_units": False})
    channels = read_capture_channels(capture, config.survey, config.scaling)
    assert channels.csi.shape[1] == 2
    assert np.allclose(channels.csi, np.rint(channels.csi.real) + 1j * np.rint(channels.csi.imag))


def test_second_stream_selection_keeps_only_two_stream_packets(capture, fast_config):
    config = fast_config.override({"survey.tx_index": 1, "synthetic.two_stream_fraction": 0.5})
    room = make_room(config.synthetic, rng=0)
    make_capture(capture, room, (0.0, 0.0), config.synthetic, n_packets=40, rng=5)
    channels = read_capture_channels(capture, config.survey, config.scaling)
    assert 0 < channels.n_packets < 40 and set(channels.n_tx.tolist()) == {2}


def test_survey_rows_follow_the_index(survey, synthetic_dataset):
    index = load_dataset_index(synthetic_dataset)
    assert survey.n_items == survey.report.n_output
    assert list(dict.fromkeys(survey.paths)) == index["relative_path"].tolist()
    first = survey.paths == survey.paths[0]
    assert np.all(survey.coordinates[first] == index[["x_m", "y_m"]].to_numpy()[0])
    assert survey.report.kept_n_tx == 1


def test_every_representation_comes_from_one_survey(survey, fast_config):
    amplitude = survey_features(survey, fast_config)
    real_imag = survey_features(survey, fast_config, "real_imag")
    assert amplitude.n_features == 90 and real_imag.n_features == 180
    assert real_imag.config["features"]["representation"] == "real_imag"
    assert np.array_equal(amplitude.y, real_imag.y)


def test_a_capture_emptied_by_cleaning_is_reported(tmp_path, fast_config):
    room = make_room(fast_config.synthetic, rng=0)
    single = fast_config.override({"synthetic.two_stream_fraction": 0.0}).synthetic
    double = fast_config.override({"synthetic.two_stream_fraction": 1.0}).synthetic
    make_capture(tmp_path / "a.dat", room, (0.0, 0.0), single, n_packets=10, rng=1)
    make_capture(tmp_path / "b.dat", room, (1.0, 0.0), double, n_packets=5, rng=2)
    rows = pd.DataFrame({"path": ["a.dat", "b.dat"], "x_m": [0.0, 1.0], "y_m": [0.0, 0.0]})
    index = write_dataset_index(tmp_path / "dataset.csv", rows)
    with pytest.raises(DatasetError, match="removed every packet"):
        read_survey(index, fast_config)
    kept = read_survey(index, fast_config.override({"cleaning.n_tx_policy": "keep"}))
    assert kept.n_items == 15
    with pytest.raises(DatasetError, match="transmit streams"):
        read_survey(index, fast_config.override({"cleaning.n_tx_policy": "require_constant"}))


def test_a_missing_stream_makes_a_capture_unusable(capture, fast_config):
    config = fast_config.override({"survey.tx_index": 2})
    with pytest.raises(DatasetError, match="no report carries"):
        read_capture_channels(capture, config.survey, config.scaling)


def test_index_turns_grid_indices_into_meters(tmp_path):
    for name in ("walk1-1.dat", "walk2-3.dat", "walk10-1.dat", "readme.txt"):
        (tmp_path / name).write_bytes(b"")
    rows = index_capture_folder(tmp_path, Config().override({"data.grid_spacing_m": 0.6}).data)
    assert rows["path"].tolist() == ["walk1-1.dat", "walk2-3.dat", "walk10-1.dat"]
    assert rows["x_m"].tolist() == pytest.approx([0.0, 0.6, 5.4])
    assert rows["y_m"].tolist() == pytest.approx([0.0, 1.2, 0.0])
    assert rows["campaign"].tolist() == ["walk"] * 3


def test_index_without_matches_is_an_error(tmp_path):
    (tmp_path / "notes.txt").write_bytes(b"")
    with pytest.raises(DatasetError, match="file_pattern"):
        index_capture_folder(tmp_path, Config().data)
    with pytest.raises(FileNotFoundError):
        index_capture_folder(tmp_path / "absent", Config().data)
