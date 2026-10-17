"""Command line, end to end on synthetic data."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from wifi_fingerprint_locator.cli import app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUICK = PROJECT_ROOT / "configs" / "quick_demo.yaml"
COMMANDS = ("synthesize", "index", "inspect", "prepare", "evaluate", "train", "predict", "compare")
MANIFEST_KEYS = [
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
RUNNER = CliRunner()


def run(*args):
    return RUNNER.invoke(app, [str(a) for a in args])


@pytest.fixture
def prepared(tmp_path):
    data = tmp_path / "data"
    assert run("synthesize", "--config", QUICK, "--out", data).exit_code == 0
    result = run(
        "prepare", "--config", QUICK, "--dataset", data / "dataset.csv", "--out", tmp_path / "prep"
    )
    assert result.exit_code == 0, result.output
    return tmp_path


def test_full_workflow_writes_every_documented_file(prepared):
    tmp = prepared
    for name in ("dataset.csv", "config.yaml", "manifest.json", "items/item_000.dat"):
        assert (tmp / "data" / name).is_file(), name
    for name in ("features.npz", "items.csv", "config.yaml", "manifest.json"):
        assert (tmp / "prep" / name).is_file(), name

    result = run("evaluate", tmp / "prep" / "features.npz", "--config", QUICK, "--out", tmp / "ev")
    assert result.exit_code == 0, result.output
    assert "mean error :" in result.output and "+/-" in result.output
    for name in (
        "metrics.json",
        "folds.csv",
        "predictions.csv",
        "recording_predictions.csv",
        "residuals.csv",
        "residuals.png",
        "error_cdf.csv",
        "error_cdf.png",
        "location_errors.csv",
        "location_errors.png",
        "config.yaml",
        "manifest.json",
    ):
        assert (tmp / "ev" / name).is_file(), name
    manifest = json.loads((tmp / "ev" / "manifest.json").read_text(encoding="utf-8"))
    assert list(manifest) == MANIFEST_KEYS and manifest["command"] == "evaluate"
    columns = pd.read_csv(tmp / "ev" / "predictions.csv").columns.tolist()
    assert columns == [
        "path",
        "segment",
        "group",
        "fold",
        "x_m",
        "y_m",
        "predicted_x_m",
        "predicted_y_m",
        "error_m",
    ]

    result = run("train", tmp / "prep" / "features.npz", "--config", QUICK, "--out", tmp / "tr")
    assert result.exit_code == 0, result.output
    assert (tmp / "tr" / "model.joblib").is_file()

    capture = tmp / "data" / "items" / "item_003.dat"
    result = run("predict", capture, "--model", tmp / "tr" / "model.joblib", "--out", tmp / "pr")
    assert result.exit_code == 0, result.output
    table = pd.read_csv(tmp / "pr" / "predictions.csv")
    assert table.columns.tolist() == ["path", "segment", "predicted_x_m", "predicted_y_m"]
    assert result.output.strip().splitlines()[-1].startswith("written to ")


def test_inspect_index_and_compare(prepared):
    tmp = prepared
    captures = sorted((tmp / "data" / "items").glob("*.dat"))[:2]
    result = run("inspect", *captures, "--out", tmp / "inspect")
    assert result.exit_code == 0, result.output
    assert len(pd.read_csv(tmp / "inspect" / "captures.csv")) == 2

    result = run(
        "index",
        tmp / "data" / "items",
        "--set",
        r"data.file_pattern=item_(?P<x>\d)(?P<y>\d)\d\.dat",
    )
    assert result.exit_code == 0, result.output
    assert len(pd.read_csv(tmp / "data" / "items" / "dataset.csv")) == 12

    result = run(
        "compare",
        "--config",
        QUICK,
        "--dataset",
        tmp / "data" / "dataset.csv",
        "--set",
        "model.name=knn",
        "--set",
        'evaluation.compare_representations=["amplitude", "real_imag"]',
        "--out",
        tmp / "cmp",
        "--no-figures",
    )
    assert result.exit_code == 0, result.output
    summary = pd.read_csv(tmp / "cmp" / "comparison_summary.csv")
    assert len(summary) == 2 * 4
    for name in ("comparison_folds.csv", "comparison.json", "error_cdf.csv", "manifest.json"):
        assert (tmp / "cmp" / name).is_file(), name
    assert not (tmp / "cmp" / "error_cdf.png").exists()


def test_the_same_seed_reproduces_evaluate_byte_for_byte(prepared):
    tmp = prepared
    features = tmp / "prep" / "features.npz"
    for name in ("a", "b"):
        result = run("evaluate", features, "--config", QUICK, "--seed", 5, "--out", tmp / name)
        assert result.exit_code == 0, result.output
    for name in ("predictions.csv", "metrics.json", "folds.csv"):
        assert (tmp / "a" / name).read_bytes() == (tmp / "b" / name).read_bytes(), name


def test_without_out_a_time_stamped_run_folder_is_created(prepared):
    tmp = prepared
    root = tmp / "runs"
    result = run(
        "evaluate",
        tmp / "prep" / "features.npz",
        "--config",
        QUICK,
        "--set",
        f"output.root={root.as_posix()}",
        "--no-figures",
    )
    assert result.exit_code == 0, result.output
    folders = list(root.iterdir())
    assert len(folders) == 1 and folders[0].name.startswith("evaluate_")
    assert folders[0].name.endswith("Z")


def test_help_version_and_bare_invocation():
    result = run("--help")
    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in result.output
    version = run("--version")
    assert version.exit_code == 0 and version.output.strip() == "wifi-fingerprint-locator 0.1.0"
    assert "Usage" in run().output


@pytest.mark.parametrize(
    "case",
    [
        "missing_input",
        "unknown_key",
        "invalid_value",
        "set_without_equals",
        "broken_yaml",
        "loud_log_level",
        "changed_feature_settings",
        "missing_dataset",
        "corrupt_features",
    ],
)
def test_expected_failures_exit_with_code_2(prepared, case):
    tmp = prepared
    features = tmp / "prep" / "features.npz"
    broken = tmp / "broken.yaml"
    broken.write_text("model: [unclosed", encoding="utf-8")
    corrupt = tmp / "corrupt.npz"
    corrupt.write_bytes(b"PK\x03\x04" + bytes(40))
    arguments = {
        "missing_input": ("evaluate", tmp / "absent.npz"),
        "unknown_key": ("evaluate", features, "--set", "model.depth=3"),
        "invalid_value": ("evaluate", features, "--set", "evaluation.n_splits=1"),
        "set_without_equals": ("evaluate", features, "--set", "model.alpha"),
        "broken_yaml": ("evaluate", features, "--config", broken),
        "loud_log_level": ("--log-level", "LOUD", "evaluate", features),
        "changed_feature_settings": (
            "evaluate",
            features,
            "--config",
            QUICK,
            "--set",
            "features.representation=real_imag",
        ),
        "missing_dataset": ("prepare", "--dataset", tmp / "absent.csv"),
        "corrupt_features": ("train", corrupt, "--config", QUICK),
    }[case]
    result = run(*arguments)
    assert result.exit_code == 2, result.output
    assert "error:" in result.output or "Usage" in result.output
    assert isinstance(result.exception, SystemExit)


def test_module_entry_point_lists_every_command():
    completed = subprocess.run(
        [sys.executable, "-m", "wifi_fingerprint_locator", "--help"],
        capture_output=True,
        encoding="utf-8",
        env=dict(os.environ, PYTHONPATH=str(PROJECT_ROOT / "src"), PYTHONIOENCODING="utf-8"),
        check=True,
    )
    for command in COMMANDS:
        assert command in completed.stdout
