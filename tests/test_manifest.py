"""Run folders and manifests: suffixes, key order, digests and JSON conventions."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

import numpy as np

from wifi_fingerprint_locator import Config, __version__
from wifi_fingerprint_locator.manifest import (
    array_sha256,
    build_manifest,
    file_sha256,
    run_directory,
    write_json,
)


def test_run_directory_adds_a_suffix_when_the_name_exists(tmp_path):
    first = run_directory(tmp_path, "evaluate")
    second = run_directory(tmp_path, "evaluate")
    assert re.fullmatch(r"evaluate_\d{8}T\d{6}Z", first.name)
    assert first != second and first.is_dir() and second.is_dir()
    if second.name.startswith(first.name):  # created within the same second
        assert second.name == f"{first.name}_1"
    third_base = tmp_path / "fixed"
    third_base.mkdir()
    assert run_directory(third_base, "x").parent == third_base


def test_manifest_keys_and_order(tmp_path):
    source = tmp_path / "input.bin"
    source.write_bytes(b"abc")
    manifest = build_manifest(
        command="prepare",
        config=Config().to_dict(),
        inputs=[source, tmp_path / "absent.bin"],
        seed=3,
        extra={"n_items": np.int64(4)},
    )
    assert list(manifest) == [
        "command",
        "created_utc",
        "package_version",
        "git_commit",
        "seed",
        "versions",
        "inputs",
        "config",
        "details",
    ]
    assert manifest["package_version"] == __version__ and manifest["seed"] == 3
    assert list(manifest["versions"])[:2] == ["python", "platform"]
    assert "scikit-learn" in manifest["versions"]
    assert manifest["inputs"][0]["sha256"].startswith("ba7816bf8f01")
    assert manifest["inputs"][0]["bytes"] == 3
    assert manifest["inputs"][1] == {"path": (tmp_path / "absent.bin").as_posix(), "missing": True}
    assert manifest["details"] == {"n_items": 4}
    created = datetime.fromisoformat(manifest["created_utc"])
    assert created.utcoffset() == timedelta(0) and created.microsecond == 0


def test_known_digests(tmp_path):
    path = tmp_path / "abc.txt"
    path.write_bytes(b"abc")
    assert file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert array_sha256(np.zeros(3)) != array_sha256(np.zeros(3, dtype=np.float32))


def test_json_writes_nan_as_null_with_a_final_newline(tmp_path):
    path = write_json(tmp_path / "x.json", {"a": float("nan"), "b": np.array([1.0, np.inf])})
    text = path.read_bytes().decode("utf-8")
    assert text.endswith("}\n") and "\r\n" not in text
    assert json.loads(text) == {"a": None, "b": [1.0, None]}
