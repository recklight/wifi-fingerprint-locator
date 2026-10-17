"""Synthetic surveys: reproducibility, on-disk format and the designed structure."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wifi_fingerprint_locator import read_beamform_file, read_survey
from wifi_fingerprint_locator.config import SyntheticConfig
from wifi_fingerprint_locator.phase_sanitization import unwrap_phase
from wifi_fingerprint_locator.synthetic import (
    make_capture,
    make_room,
    path_geometry,
    reference_grid,
    write_synthetic_dataset,
)

SMALL = SyntheticConfig(grid_shape=(3, 2), n_packets=8)


def folder_bytes(folder):
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def test_the_same_seed_gives_identical_bytes(tmp_path):
    write_synthetic_dataset(tmp_path / "a", SMALL, rng=4)
    write_synthetic_dataset(tmp_path / "b", SMALL, rng=4)
    assert folder_bytes(tmp_path / "a") == folder_bytes(tmp_path / "b")


def test_different_seeds_give_different_surveys(tmp_path):
    write_synthetic_dataset(tmp_path / "a", SMALL, rng=1)
    write_synthetic_dataset(tmp_path / "b", SMALL, rng=2)
    assert folder_bytes(tmp_path / "a") != folder_bytes(tmp_path / "b")


def test_index_lists_one_capture_per_point_and_campaign(tmp_path):
    config = SyntheticConfig(grid_shape=(2, 2), n_packets=3, n_campaigns=2)
    index = write_synthetic_dataset(tmp_path, config, rng=0)
    rows = pd.read_csv(index)
    assert len(rows) == 8
    assert rows["path"].iloc[-1] == "items/item_007.dat"
    assert rows["group"].nunique() == 4 and rows["campaign"].nunique() == 2
    assert rows[["x_m", "y_m"]].iloc[:4].to_numpy().tolist() == reference_grid((2, 2), 1.0).tolist()


def test_reports_carry_the_designed_impairments(tmp_path):
    config = SyntheticConfig(two_stream_fraction=0.3, zero_entry_rate=0.01)
    room = make_room(config, rng=0)
    make_capture(tmp_path / "a.dat", room, (2.0, 2.0), config, n_packets=200, rng=1)
    reports, statistics = read_beamform_file(tmp_path / "a.dat")
    assert statistics.n_reports == 200 and statistics.n_malformed == 0
    streams = np.array([r.meta.n_tx for r in reports])
    assert 0.15 < np.mean(streams == 2) < 0.45
    assert sum(int(np.sum(r.csi == 0)) for r in reports) > 0
    noise = np.array([r.meta.noise_dbm for r in reports])
    agc = np.array([r.meta.agc_db for r in reports])
    assert noise.min() >= -96 and noise.max() <= -84
    assert agc.min() >= 38 and agc.max() <= 45
    assert all(abs(r.csi).max() <= 128 for r in reports)


def test_raw_phase_slope_changes_from_packet_to_packet(tmp_path):
    config = SyntheticConfig(zero_entry_rate=0.0)
    room = make_room(config, rng=0)
    reports = make_capture(tmp_path / "a.dat", room, (1.0, 1.0), config, n_packets=40, rng=2)
    phases = np.array([unwrap_phase(r.csi[0, 0]) for r in reports])
    slopes = phases[:, -1] - phases[:, 0]
    assert np.std(slopes) > 0.5


def test_received_power_falls_with_distance_from_the_access_point():
    config = SyntheticConfig(n_scatterers=0)
    room = make_room(config, rng=0)
    near, far = (np.array([[0.0, 0.0]]), np.array([[7.0, 6.0]]))
    near_gain = path_geometry(room, near, 0.0)[1][0, 0, 0]
    far_gain = path_geometry(room, far, 0.0)[1][0, 0, 0]
    assert near_gain > 4 * far_gain


def test_packets_of_one_point_are_more_alike_than_packets_of_different_points(
    tmp_path, fast_config
):
    survey = read_survey(
        write_synthetic_dataset(tmp_path, fast_config.synthetic, rng=3), fast_config
    )
    amplitude = np.abs(survey.csi).reshape(survey.n_items, -1)
    names = list(dict.fromkeys(survey.groups))
    means = np.array([amplitude[survey.groups == name].mean(axis=0) for name in names])
    within = np.mean(
        [np.linalg.norm(amplitude[survey.groups == name] - means[k], axis=1).mean()
         for k, name in enumerate(names)]
    )  # fmt: skip
    between = np.mean(
        np.linalg.norm(means[:, None] - means[None, :], axis=-1)[np.triu_indices(len(names), 1)]
    )
    assert between > 2 * within
