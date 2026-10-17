"""Configuration: defaults, YAML round trips, unknown keys and value checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from wifi_fingerprint_locator.config import FEATURE_SECTIONS, Config
from wifi_fingerprint_locator.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_equals_the_dataclass_defaults():
    assert Config.from_yaml(PROJECT_ROOT / "configs" / "default.yaml") == Config()


@pytest.mark.parametrize("path", sorted((PROJECT_ROOT / "configs").glob("*.yaml")), ids=str)
def test_every_shipped_configuration_loads(path):
    assert isinstance(Config.from_yaml(path), Config)


def test_quick_demo_points_at_the_synthesize_output():
    config = Config.from_yaml(PROJECT_ROOT / "configs" / "quick_demo.yaml")
    assert config.data.dataset == "data/synthetic/dataset.csv"
    first_line = (PROJECT_ROOT / "configs" / "quick_demo.yaml").read_text(encoding="utf-8")
    assert first_line.startswith("# Small, fast settings for synthetic data")


def test_yaml_round_trip(tmp_path):
    config = Config().override({"model.hidden_layer_sizes": [8, 4], "seed": 3})
    path = config.to_yaml(tmp_path / "config.yaml")
    assert Config.from_yaml(path) == config
    assert config.model.hidden_layer_sizes == (8, 4)


def test_configurations_are_hashable():
    assert hash(Config()) == hash(Config())


def test_feature_sections_are_sections():
    assert set(FEATURE_SECTIONS) <= set(Config().to_dict())
    assert "model" not in FEATURE_SECTIONS


def test_unknown_keys_are_rejected_with_the_valid_ones_listed():
    with pytest.raises(ConfigError, match=r"valid keys are \[.*'representation'"):
        Config.from_dict({"features": {"representaton": "amplitude"}})
    with pytest.raises(ConfigError, match="valid top-level keys"):
        Config.from_dict({"modle": {}})
    with pytest.raises(ConfigError, match="unknown configuration key"):
        Config().override({"model.depth": 3})
    with pytest.raises(ConfigError, match="unknown configuration section"):
        Config().override({"network.depth": 3})


@pytest.mark.parametrize(
    ("key", "value", "fragment"),
    [
        ("seed", -1, "seed"),
        ("evaluation.n_splits", 1, "evaluation.n_splits"),
        ("evaluation.split_strategy", "random", "evaluation.split_strategy"),
        ("evaluation.compare_representations", ["amplitude", "amplitude"], "twice"),
        ("evaluation.compare_strategies", ["sideways"], "evaluation.compare_strategies"),
        ("features.representation", "phase", "features.representation"),
        ("features.subcarrier_layout", "dense", "features.subcarrier_layout"),
        ("features.phase_fit", "cubic", "features.phase_fit"),
        ("cleaning.n_tx_policy", "first", "cleaning.n_tx_policy"),
        ("cleaning.zero_entry_policy", "fill", "cleaning.zero_entry_policy"),
        ("survey.n_packets", 0, "survey.n_packets"),
        ("survey.rx_indices", [0, 0], "survey.rx_indices"),
        ("survey.tx_index", 3, "survey.tx_index"),
        ("model.name", "forest", "model.name"),
        ("model.solver", "sgd", "model.solver"),
        ("model.hidden_layer_sizes", [], "model.hidden_layer_sizes"),
        ("model.alpha", -0.1, "model.alpha"),
        ("model.alpha", "1e-4", "model.alpha"),
        ("model.validation_fraction", 1.0, "model.validation_fraction"),
        ("model.lm_mu_increase", 1.0, "model.lm_mu_increase"),
        ("synthetic.grid_shape", [3], "synthetic.grid_shape"),
        ("synthetic.n_rx", 4, "synthetic.n_rx"),
        ("synthetic.zero_entry_rate", 1.5, "synthetic.zero_entry_rate"),
        ("data.file_pattern", "(?P<x>\\d+)", "data.file_pattern"),
        ("data.file_pattern", "(", "data.file_pattern"),
        ("output.figure_format", "gif", "output.figure_format"),
        ("output.dpi", 0, "output.dpi"),
    ],
)
def test_invalid_values_name_the_dotted_key(key, value, fragment):
    with pytest.raises(ConfigError, match=fragment):
        Config().override({key: value})


def test_levenberg_marquardt_only_applies_to_the_network():
    with pytest.raises(ConfigError, match="only applies"):
        Config().override({"model.solver": "levenberg_marquardt", "model.name": "knn"})


def test_missing_and_malformed_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        Config.from_yaml(tmp_path / "absent.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("model: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        Config.from_yaml(broken)
    listing = tmp_path / "list.yaml"
    listing.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        Config.from_yaml(listing)


@pytest.mark.parametrize("key", ["model.patience", "model.validation_fraction"])
def test_alpha_grid_needs_a_validation_set(key):
    with pytest.raises(ConfigError, match=r"model\.alpha_grid"):
        Config().override({"model.alpha_grid": [0.1, 0.0001], key: 0})
