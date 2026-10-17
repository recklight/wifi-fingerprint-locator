"""Dataset index and features file: paths, required columns and consistency checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wifi_fingerprint_locator import Config
from wifi_fingerprint_locator.dataset import (
    FeatureSet,
    load_dataset_index,
    load_features,
    save_features,
    write_dataset_index,
)
from wifi_fingerprint_locator.errors import ConfigError, DatasetError


def write_index(folder, text, files=("a.dat",)):
    folder.mkdir(parents=True, exist_ok=True)
    for name in files:
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(b"")
    path = folder / "dataset.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_paths_are_relative_to_the_index_folder(tmp_path, monkeypatch):
    index = write_index(tmp_path / "survey", "path,x_m,y_m\nsub/a.dat,1,2\n", files=("sub/a.dat",))
    monkeypatch.chdir(tmp_path)
    frame = load_dataset_index(index)
    assert frame["path"].iloc[0] == str((tmp_path / "survey" / "sub" / "a.dat").resolve())
    assert frame["relative_path"].iloc[0] == "sub/a.dat"
    assert frame["x_m"].dtype == np.float64


def test_groups_default_to_the_position_and_blanks_are_filled(tmp_path):
    index = write_index(
        tmp_path, "path,x_m,y_m,group\na.dat,1,2,\nb.dat,0,0,door\n", files=("a.dat", "b.dat")
    )
    assert load_dataset_index(index)["group"].tolist() == ["x1_y2", "door"]


def test_missing_columns_are_listed(tmp_path):
    index = write_index(tmp_path, "path,x\na.dat,1\n")
    with pytest.raises(
        DatasetError, match=r"lacks column\(s\) \['x_m', 'y_m'\]; found \['path', 'x'\]"
    ):
        load_dataset_index(index)


def test_missing_capture_files_are_listed_and_counted(tmp_path):
    rows = "path,x_m,y_m\n" + "".join(f"c{k}.dat,0,0\n" for k in range(7))
    index = write_index(tmp_path, rows, files=())
    with pytest.raises(DatasetError, match=r"7 capture file\(s\).*c0.dat.*and 2 more"):
        load_dataset_index(index)


def test_non_numeric_coordinates_are_rejected(tmp_path):
    index = write_index(tmp_path, "path,x_m,y_m\na.dat,left,2\n")
    with pytest.raises(DatasetError, match="finite numbers"):
        load_dataset_index(index)


def test_missing_index_explains_the_next_step(tmp_path):
    with pytest.raises(FileNotFoundError, match="synthesize"):
        load_dataset_index(tmp_path / "dataset.csv")


def test_written_index_uses_lf_line_endings(tmp_path):
    path = write_dataset_index(tmp_path / "dataset.csv", pd.DataFrame({"path": ["a"], "x_m": [1]}))
    assert b"\r\n" not in path.read_bytes()


def make_features(config):
    return FeatureSet(
        X=np.arange(12.0).reshape(4, 3),
        y=np.zeros((4, 2)),
        groups=np.array(["p", "p", "q", "q"]),
        paths=np.array(["a.dat", "a.dat", "b.dat", "b.dat"]),
        config=config.to_dict(),
        segments=np.array([0, 1, 0, 1]),
    )


def test_features_round_trip_byte_for_byte(tmp_path):
    features = make_features(Config())
    first = save_features(tmp_path / "one.npz", features)
    second = save_features(tmp_path / "two.npz", features)
    assert first.read_bytes() == second.read_bytes()
    loaded = load_features(first, Config())
    assert np.array_equal(loaded.X, features.X) and loaded.segments.tolist() == [0, 1, 0, 1]
    assert loaded.items_table().columns.tolist() == ["path", "x_m", "y_m", "group", "segment"]


def test_changed_feature_settings_are_detected(tmp_path):
    path = save_features(tmp_path / "features.npz", make_features(Config()))
    changed = Config().override({"features.representation": "real_imag", "model.alpha": 1.0})
    with pytest.raises(ConfigError, match=r"differ.*features\.representation"):
        load_features(path, changed)
    assert load_features(path, Config().override({"model.alpha": 1.0})).n_items == 4


def test_foreign_files_are_not_features(tmp_path):
    other = tmp_path / "other.npz"
    np.savez(other, values=np.zeros(3))
    with pytest.raises(DatasetError, match="lacks key"):
        load_features(other)
    with pytest.raises(FileNotFoundError, match="Run 'prepare' first"):
        load_features(tmp_path / "absent.npz")


@pytest.mark.parametrize(
    ("content", "fragment"),
    [
        (b"PK\x03\x04" + bytes(40), "not a features file"),
        (b"plain text, not an archive", "not a features file"),
        (None, "single array"),
    ],
)
def test_damaged_features_files_raise_a_dataset_error(tmp_path, content, fragment):
    path = tmp_path / "features.npz"
    if content is None:
        with path.open("wb") as stream:
            np.save(stream, np.zeros(3))
    else:
        path.write_bytes(content)
    with pytest.raises(DatasetError, match=fragment):
        load_features(path)


def test_feature_set_validates_shapes():
    with pytest.raises(DatasetError, match="y must be shaped"):
        FeatureSet(np.zeros((3, 2)), np.zeros((3, 3)), np.array(["a"] * 3), np.array(["f"] * 3), {})
    with pytest.raises(DatasetError, match="groups"):
        FeatureSet(np.zeros((3, 2)), np.zeros((3, 2)), np.array(["a"] * 2), np.array(["f"] * 3), {})
