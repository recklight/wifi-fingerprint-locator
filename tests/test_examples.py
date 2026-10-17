"""Every example script runs end to end on synthetic data with the quick settings."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_example_is_covered():
    covered = {
        "end_to_end_synthetic",
        "sanitize_channel_phase",
        "compare_feature_representations",
        "compare_split_strategies",
    }
    assert {path.stem for path in EXAMPLES.glob("*.py")} == covered


def test_end_to_end_synthetic(tmp_path):
    out = tmp_path / "end_to_end_synthetic"
    assert load("end_to_end_synthetic").main(["--quick", "--out", str(out)]) == 0
    for name in (
        "data/dataset.csv",
        "features.npz",
        "metrics.json",
        "predictions.csv",
        "model.joblib",
        "located.csv",
    ):
        assert (out / name).is_file(), name


def test_sanitize_channel_phase(tmp_path):
    out = tmp_path / "sanitize_channel_phase"
    assert load("sanitize_channel_phase").main(["--quick", "--out", str(out)]) == 0
    for name in ("phase_sanitization.csv", "phase_sanitization.png", "phase_stability.csv"):
        assert (out / name).is_file(), name
    stability = pd.read_csv(out / "phase_stability.csv")
    raw = stability.loc[stability["method"] == "unwrapped only", "spread_rad"].iloc[0]
    assert (stability["spread_rad"].iloc[1:] < raw).all()


def test_compare_feature_representations(tmp_path):
    out = tmp_path / "compare_feature_representations"
    assert load("compare_feature_representations").main(["--quick", "--out", str(out)]) == 0
    for name in ("summary.csv", "error_cdf.csv", "error_cdf.png"):
        assert (out / name).is_file(), name
    assert len(pd.read_csv(out / "summary.csv")) == 5


def test_compare_split_strategies(tmp_path):
    out = tmp_path / "compare_split_strategies"
    arguments = ["--quick", "--seed", "1", "--out", str(out)]
    assert load("compare_split_strategies").main(arguments) == 0
    for name in ("summary.csv", "error_cdf.csv", "error_cdf.png"):
        assert (out / name).is_file(), name
    summary = pd.read_csv(out / "summary.csv")
    assert len(summary) == 8 and set(summary["model"]) == {"mlp", "weighted_knn"}
