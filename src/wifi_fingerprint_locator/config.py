"""Run configuration: frozen dataclasses validated on construction, stored as YAML.

Every tunable number of the pipeline is a field here, with its unit in the name
where it has one (``grid_spacing_m``, ``max_timing_offset_ns``).
Each section validates itself in ``__post_init__`` and raises
:class:`~wifi_fingerprint_locator.errors.ConfigError` naming the dotted key, the value it got and
what is allowed. Unknown keys are rejected rather than ignored, so a typo
cannot silently fall back to a default.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .channel_cleaning import N_TX_POLICIES, ZERO_ENTRY_POLICIES
from .errors import ConfigError
from .fingerprints import REPRESENTATIONS
from .models import MODEL_NAMES
from .phase_sanitization import PHASE_FIT_METHODS
from .position_regressor import SOLVERS
from .splits import SPLIT_STRATEGIES
from .subcarriers import SUBCARRIER_LAYOUTS

#: Image formats accepted by ``output.figure_format``.
FIGURE_FORMATS: tuple[str, ...] = ("png", "pdf", "svg")


def _choice(name: str, value: object, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ConfigError(f"{name}={value!r} is not supported; choose one of {list(allowed)}")


def _number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{name} must be a number, got {value!r}")


def _positive(name: str, value: float) -> None:
    _number(name, value)
    if not value > 0:
        raise ConfigError(f"{name} must be strictly positive, got {value!r}")


def _integer(name: str, value: object, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{name} must be an integer of at least {minimum}, got {value!r}")


def _fraction(name: str, value: object, *, upper_open: bool = True) -> None:
    _number(name, value)
    assert isinstance(value, int | float)
    if not (0.0 <= value < 1.0 if upper_open else 0.0 <= value <= 1.0):
        bound = "[0, 1)" if upper_open else "[0, 1]"
        raise ConfigError(f"{name} must lie in {bound}, got {value!r}")


def _boolean(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true or false, got {value!r}")


def _pair(name: str, value: object) -> None:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ConfigError(f"{name} must be a list of two numbers, got {value!r}")
    for item in value:
        _number(name, item)


@dataclass(frozen=True)
class DataConfig:
    """Where the data lives and how a folder of captures is indexed.

    Attributes
    ----------
    dataset : str
        Dataset index (CSV, see :mod:`wifi_fingerprint_locator.dataset`). A
        placeholder until you point it at your own data.
    file_pattern : str
        Regular expression the ``index`` command matches against capture file
        names; it needs the named groups ``x`` and ``y`` (grid indices) and may
        have ``campaign``.
    grid_spacing_m : float
        Distance between neighboring grid points, used by ``index`` to turn
        grid indices into meters.
    grid_origin : int
        Grid index that ``index`` places at 0 m.
    """

    dataset: str = "data/dataset.csv"
    file_pattern: str = r"(?P<campaign>\D*)(?P<x>\d+)-(?P<y>\d+)\.dat"
    grid_spacing_m: float = 1.0
    grid_origin: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset:
            raise ConfigError(f"data.dataset must be a non-empty path, got {self.dataset!r}")
        try:
            pattern = re.compile(self.file_pattern)
        except (re.error, TypeError) as exc:
            raise ConfigError(
                f"data.file_pattern={self.file_pattern!r} is not a regular expression: {exc}"
            ) from None
        missing = {"x", "y"} - set(pattern.groupindex)
        if missing:
            raise ConfigError(
                f"data.file_pattern={self.file_pattern!r} lacks the named group(s) "
                f"{sorted(missing)}"
            )
        _positive("data.grid_spacing_m", self.grid_spacing_m)
        _integer("data.grid_origin", self.grid_origin, -1_000_000)


@dataclass(frozen=True)
class SyntheticConfig:
    """Synthetic survey written by ``synthesize`` (see :mod:`wifi_fingerprint_locator.synthetic`).

    Attributes
    ----------
    grid_shape : tuple of int
        Reference points along x and y.
    grid_spacing_m : float
        Distance between neighboring reference points.
    n_packets : int
        Packets recorded at every reference point per campaign.
    n_campaigns : int
        Repeated surveys of the same grid (one capture file per point each).
    n_rx : int
        Receive antennas, 1 to 3.
    room_margin_m : float
        Distance from the outermost reference points to the walls.
    access_point_m : tuple of float
        Position of the transmitter, relative to the first reference point.
    n_scatterers : int
        Fixed point scatterers (furniture) placed at random in the room.
    wall_reflectivity : float
        Amplitude reflection coefficient of the four walls.
    center_frequency_hz : float
        Carrier frequency.
    antenna_spacing_m : float
        Spacing of the receive antennas (a linear array along x).
    tx_power_dbm : float
        Transmit power.
    snr_db : float
        Signal-to-noise ratio of the measured channel.
    gain_jitter_db : float
        Standard deviation of the per-packet gain fluctuation.
    scatterer_jitter : float
        Relative per-packet fluctuation of the scatterer paths (moving people).
    max_timing_offset_ns : float
        Per-packet sampling-time offset, uniform in plus or minus this value.
    two_stream_fraction : float
        Share of packets reported with two transmit streams.
    zero_entry_rate : float
        Share of CSI entries reported as exactly zero.
    packet_interval_s : float
        Time between packets.
    """

    grid_shape: tuple[int, int] = (8, 7)
    grid_spacing_m: float = 1.0
    n_packets: int = 200
    n_campaigns: int = 1
    n_rx: int = 3
    room_margin_m: float = 1.5
    access_point_m: tuple[float, float] = (-1.0, -1.0)
    n_scatterers: int = 6
    wall_reflectivity: float = 0.5
    center_frequency_hz: float = 5.32e9
    antenna_spacing_m: float = 0.028
    tx_power_dbm: float = 15.0
    snr_db: float = 30.0
    gain_jitter_db: float = 1.0
    scatterer_jitter: float = 0.1
    max_timing_offset_ns: float = 50.0
    two_stream_fraction: float = 0.04
    zero_entry_rate: float = 0.001
    packet_interval_s: float = 1.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.grid_shape, tuple)
            or len(self.grid_shape) != 2
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in self.grid_shape)
        ):
            raise ConfigError(
                f"synthetic.grid_shape must be two positive integers, got {self.grid_shape!r}"
            )
        _positive("synthetic.grid_spacing_m", self.grid_spacing_m)
        _integer("synthetic.n_packets", self.n_packets, 1)
        _integer("synthetic.n_campaigns", self.n_campaigns, 1)
        _integer("synthetic.n_rx", self.n_rx, 1)
        if self.n_rx > 3:
            raise ConfigError(f"synthetic.n_rx must be 1, 2 or 3, got {self.n_rx!r}")
        _positive("synthetic.room_margin_m", self.room_margin_m)
        _pair("synthetic.access_point_m", self.access_point_m)
        _integer("synthetic.n_scatterers", self.n_scatterers, 0)
        _fraction("synthetic.wall_reflectivity", self.wall_reflectivity, upper_open=False)
        _positive("synthetic.center_frequency_hz", self.center_frequency_hz)
        _positive("synthetic.antenna_spacing_m", self.antenna_spacing_m)
        _number("synthetic.tx_power_dbm", self.tx_power_dbm)
        _number("synthetic.snr_db", self.snr_db)
        _number("synthetic.gain_jitter_db", self.gain_jitter_db)
        if self.gain_jitter_db < 0:
            raise ConfigError(
                f"synthetic.gain_jitter_db must not be negative, got {self.gain_jitter_db!r}"
            )
        _number("synthetic.scatterer_jitter", self.scatterer_jitter)
        if self.scatterer_jitter < 0:
            raise ConfigError(
                f"synthetic.scatterer_jitter must not be negative, got {self.scatterer_jitter!r}"
            )
        _number("synthetic.max_timing_offset_ns", self.max_timing_offset_ns)
        if self.max_timing_offset_ns < 0:
            raise ConfigError(
                "synthetic.max_timing_offset_ns must not be negative, "
                f"got {self.max_timing_offset_ns!r}"
            )
        _fraction("synthetic.two_stream_fraction", self.two_stream_fraction, upper_open=False)
        _fraction("synthetic.zero_entry_rate", self.zero_entry_rate)
        _positive("synthetic.packet_interval_s", self.packet_interval_s)


@dataclass(frozen=True)
class SurveyConfig:
    """Which packets of each capture are used.

    Attributes
    ----------
    packet_start : int
        First report of each capture used for the survey (0-based).
    n_packets : int or None
        Reports per capture from ``packet_start`` on; ``None`` uses every
        remaining report. A capture with fewer reports is used as it is, with
        a warning. ``predict`` always uses every report of its inputs.
    tx_index : int
        Transmit stream used (0 = the first).
    rx_indices : tuple of int or None
        Receive antennas used (0-based); ``None`` uses all of them.
    apply_permutation : bool
        Reorder receive columns into physical antenna order.
    strict : bool
        Fail on the first malformed or truncated report instead of skipping it.
    """

    packet_start: int = 0
    n_packets: int | None = 200
    tx_index: int = 0
    rx_indices: tuple[int, ...] | None = None
    apply_permutation: bool = True
    strict: bool = False

    def __post_init__(self) -> None:
        _integer("survey.packet_start", self.packet_start, 0)
        if self.n_packets is not None:
            _integer("survey.n_packets", self.n_packets, 1)
        _integer("survey.tx_index", self.tx_index, 0)
        if self.tx_index > 2:
            raise ConfigError(f"survey.tx_index must be 0, 1 or 2, got {self.tx_index!r}")
        if self.rx_indices is not None:
            values = self.rx_indices
            if (
                not isinstance(values, tuple)
                or not values
                or any(isinstance(v, bool) or not isinstance(v, int) for v in values)
                or len(set(values)) != len(values)
                or min(values) < 0
                or max(values) > 2
            ):
                raise ConfigError(
                    "survey.rx_indices must be null or distinct antenna indices in 0..2, "
                    f"got {values!r}"
                )
        _boolean("survey.apply_permutation", self.apply_permutation)
        _boolean("survey.strict", self.strict)


@dataclass(frozen=True)
class ScalingConfig:
    """Conversion of raw CSI to sqrt(SNR) units (see :mod:`wifi_fingerprint_locator.csi_scaling`).

    Attributes
    ----------
    to_snr_units : bool
        Rescale with the report's RSSI, AGC and noise; when false the raw
        8-bit values are used.
    noise_floor_fallback_dbm : float
        Noise floor assumed when a report gives none (-127 dBm).
    three_stream_gain_db : float
        Power compensation of three-stream reports.
    compensate_stream_power : bool
        Apply the per-stream power compensation.
    """

    to_snr_units: bool = True
    noise_floor_fallback_dbm: float = -92.0
    three_stream_gain_db: float = 4.5
    compensate_stream_power: bool = True

    def __post_init__(self) -> None:
        _boolean("scaling.to_snr_units", self.to_snr_units)
        _number("scaling.noise_floor_fallback_dbm", self.noise_floor_fallback_dbm)
        _number("scaling.three_stream_gain_db", self.three_stream_gain_db)
        _boolean("scaling.compensate_stream_power", self.compensate_stream_power)


@dataclass(frozen=True)
class CleaningConfig:
    """Validation and repair of the measurements (see :mod:`~.channel_cleaning`).

    Attributes
    ----------
    n_tx_policy : str
        ``keep``, ``require_constant`` or ``drop_minority`` (keep the most
        common transmit-stream count of the survey).
    zero_entry_policy : str
        ``interpolate``, ``drop`` or ``keep`` exact-zero CSI entries.
    drop_nonfinite : bool
        Drop packets with undefined values instead of failing.
    """

    n_tx_policy: str = "drop_minority"
    zero_entry_policy: str = "interpolate"
    drop_nonfinite: bool = True

    def __post_init__(self) -> None:
        _choice("cleaning.n_tx_policy", self.n_tx_policy, N_TX_POLICIES)
        _choice("cleaning.zero_entry_policy", self.zero_entry_policy, ZERO_ENTRY_POLICIES)
        _boolean("cleaning.drop_nonfinite", self.drop_nonfinite)


@dataclass(frozen=True)
class FeaturesConfig:
    """Fingerprint representation (see :mod:`wifi_fingerprint_locator.fingerprints`).

    Attributes
    ----------
    representation : str
        ``amplitude``, ``amplitude_db``, ``sanitized_phase``,
        ``amplitude_phase``, ``real_imag`` or ``sanitized_real_imag``.
    subcarrier_layout : str
        ``grouped`` (true subcarrier indices) or ``uniform`` (equally spaced
        approximation) positions for the phase fit and interpolation.
    phase_fit : str
        ``endpoint`` or ``lstsq`` line estimator of the phase sanitization.
    amplitude_floor_db : float
        Smallest amplitude kept by ``amplitude_db``.
    """

    representation: str = "amplitude"
    subcarrier_layout: str = "grouped"
    phase_fit: str = "endpoint"
    amplitude_floor_db: float = -20.0

    def __post_init__(self) -> None:
        _choice("features.representation", self.representation, REPRESENTATIONS)
        _choice("features.subcarrier_layout", self.subcarrier_layout, SUBCARRIER_LAYOUTS)
        _choice("features.phase_fit", self.phase_fit, PHASE_FIT_METHODS)
        _number("features.amplitude_floor_db", self.amplitude_floor_db)


@dataclass(frozen=True)
class ModelConfig:
    """Position estimator (see :mod:`wifi_fingerprint_locator.models`).

    Attributes
    ----------
    name : str
        ``mlp``, ``knn``, ``weighted_knn`` or ``nearest_reference``.
    hidden_layer_sizes : tuple of int
        Width of each hidden layer of the MLP.
    solver : str
        ``lbfgs``, ``adam`` or ``levenberg_marquardt``.
    alpha : float
        L2 penalty on the MLP weights.
    alpha_grid : tuple of float
        When not empty, candidate penalties chosen among by validation error;
        choosing among several needs ``patience`` and ``validation_fraction``
        above 0.
    max_epochs : int
        Upper bound on training iterations.
    epochs_per_check : int
        Iterations between validation checks (``lbfgs`` and ``adam``).
    patience : int
        Validation checks without improvement before stopping; 0 disables
        early stopping.
    validation_fraction : float
        Share of the training items held out for early stopping. Under the
        ``location_block`` and ``location_holdout`` evaluations it is a share
        of the reference points, held out whole.
    learning_rate : float
        Initial step size of ``adam``.
    min_gradient : float
        Gradient tolerance of ``lbfgs`` and ``levenberg_marquardt``.
    lm_initial_mu, lm_mu_decrease, lm_mu_increase, lm_mu_max : float
        Damping schedule of ``levenberg_marquardt``.
    lm_memory_limit_mb : float
        Memory ceiling of ``levenberg_marquardt``.
    n_neighbors : int
        Neighbors of ``knn`` and ``weighted_knn``.
    """

    name: str = "mlp"
    hidden_layer_sizes: tuple[int, ...] = (60, 30, 15)
    solver: str = "lbfgs"
    alpha: float = 0.0001
    alpha_grid: tuple[float, ...] = ()
    max_epochs: int = 1000
    epochs_per_check: int = 10
    patience: int = 6
    validation_fraction: float = 0.15
    learning_rate: float = 0.001
    min_gradient: float = 1e-07
    lm_initial_mu: float = 0.001
    lm_mu_decrease: float = 0.1
    lm_mu_increase: float = 10.0
    lm_mu_max: float = 10000000000.0
    lm_memory_limit_mb: float = 512.0
    n_neighbors: int = 5

    def __post_init__(self) -> None:
        _choice("model.name", self.name, MODEL_NAMES)
        sizes = self.hidden_layer_sizes
        if (
            not isinstance(sizes, tuple)
            or not sizes
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in sizes)
        ):
            raise ConfigError(
                f"model.hidden_layer_sizes must be a list of positive integers, got {sizes!r}"
            )
        _choice("model.solver", self.solver, SOLVERS)
        _number("model.alpha", self.alpha)
        if self.alpha < 0:
            raise ConfigError(f"model.alpha must not be negative, got {self.alpha!r}")
        if not isinstance(self.alpha_grid, tuple):
            raise ConfigError(
                f"model.alpha_grid must be a list of numbers, got {self.alpha_grid!r}"
            )
        for value in self.alpha_grid:
            _number("model.alpha_grid", value)
            if value < 0:
                raise ConfigError(f"model.alpha_grid must not hold negative values, got {value!r}")
        _integer("model.max_epochs", self.max_epochs, 1)
        _integer("model.epochs_per_check", self.epochs_per_check, 1)
        _integer("model.patience", self.patience, 0)
        _fraction("model.validation_fraction", self.validation_fraction)
        if len(self.alpha_grid) > 1 and (self.patience == 0 or self.validation_fraction == 0):
            raise ConfigError(
                "choosing from model.alpha_grid needs a validation set, so model.patience and "
                f"model.validation_fraction must be above 0; got patience={self.patience!r}, "
                f"validation_fraction={self.validation_fraction!r}"
            )
        _positive("model.learning_rate", self.learning_rate)
        _positive("model.min_gradient", self.min_gradient)
        _positive("model.lm_initial_mu", self.lm_initial_mu)
        _fraction("model.lm_mu_decrease", self.lm_mu_decrease)
        _positive("model.lm_mu_decrease", self.lm_mu_decrease)
        _positive("model.lm_mu_increase", self.lm_mu_increase)
        if self.lm_mu_increase <= 1:
            raise ConfigError(
                f"model.lm_mu_increase must be greater than 1, got {self.lm_mu_increase!r}"
            )
        _positive("model.lm_mu_max", self.lm_mu_max)
        _positive("model.lm_memory_limit_mb", self.lm_memory_limit_mb)
        _integer("model.n_neighbors", self.n_neighbors, 1)


@dataclass(frozen=True)
class EvaluationConfig:
    """Cross-validation and the comparison matrix.

    Attributes
    ----------
    n_splits : int
        Number of folds.
    split_strategy : str
        ``packet_random``, ``location_block``, ``location_holdout`` or
        ``temporal`` (see :mod:`wifi_fingerprint_locator.splits`).
    cdf_step_m : float
        Threshold spacing of the error CDF.
    compare_representations : tuple of str
        Representations the ``compare`` command evaluates.
    compare_strategies : tuple of str
        Split strategies the ``compare`` command evaluates.
    """

    n_splits: int = 5
    split_strategy: str = "location_holdout"
    cdf_step_m: float = 0.2
    compare_representations: tuple[str, ...] = (
        "amplitude",
        "sanitized_phase",
        "amplitude_phase",
        "real_imag",
    )
    compare_strategies: tuple[str, ...] = (
        "packet_random",
        "location_block",
        "location_holdout",
        "temporal",
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_splits, bool)
            or not isinstance(self.n_splits, int)
            or self.n_splits < 2
        ):
            raise ConfigError(f"evaluation.n_splits must be at least 2, got {self.n_splits!r}")
        _choice("evaluation.split_strategy", self.split_strategy, SPLIT_STRATEGIES)
        _positive("evaluation.cdf_step_m", self.cdf_step_m)
        for key, values, allowed in (
            ("evaluation.compare_representations", self.compare_representations, REPRESENTATIONS),
            ("evaluation.compare_strategies", self.compare_strategies, SPLIT_STRATEGIES),
        ):
            if not isinstance(values, tuple) or not values:
                raise ConfigError(f"{key} must be a non-empty list, got {values!r}")
            if len(set(values)) != len(values):
                raise ConfigError(f"{key} lists a value twice: {list(values)}")
            for value in values:
                _choice(key, value, allowed)


@dataclass(frozen=True)
class OutputConfig:
    """Where and how results are written.

    Attributes
    ----------
    root : str
        Run folders are created here when ``--out`` is not given.
    figure_format : str
        ``png``, ``pdf`` or ``svg``.
    dpi : int
        Resolution of raster figures.
    """

    root: str = "outputs"
    figure_format: str = "png"
    dpi: int = 150

    def __post_init__(self) -> None:
        _choice("output.figure_format", self.figure_format, FIGURE_FORMATS)
        _positive("output.dpi", self.dpi)


_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "synthetic": SyntheticConfig,
    "survey": SurveyConfig,
    "scaling": ScalingConfig,
    "cleaning": CleaningConfig,
    "features": FeaturesConfig,
    "model": ModelConfig,
    "evaluation": EvaluationConfig,
    "output": OutputConfig,
}

#: Sections whose values change what ``prepare`` writes.
FEATURE_SECTIONS: tuple[str, ...] = ("survey", "scaling", "cleaning", "features")


def _tuplify(value: Any) -> Any:
    """YAML lists become tuples, so configurations stay hashable."""
    if isinstance(value, list):
        return tuple(_tuplify(item) for item in value)
    return value


def _listify(value: Any) -> Any:
    """Tuples become lists again, for YAML and JSON."""
    if isinstance(value, dict):
        return {key: _listify(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_listify(item) for item in value]
    return value


def _build_section(name: str, cls: type, raw: Any) -> Any:
    if raw is None:
        return cls()
    if not isinstance(raw, Mapping):
        raise ConfigError(f"section {name!r} must be a mapping, got {type(raw).__name__}")
    known = sorted(f.name for f in dataclasses.fields(cls))
    unknown = sorted(set(raw) - set(known))
    if unknown:
        raise ConfigError(f"unknown key(s) {unknown} in section {name!r}; valid keys are {known}")
    try:
        return cls(**{key: _tuplify(value) for key, value in raw.items()})
    except TypeError as exc:
        raise ConfigError(f"section {name!r}: {exc}") from None


@dataclass(frozen=True)
class Config:
    """Complete configuration of a run.

    Attributes
    ----------
    seed : int
        Master seed: splits, model initialization and synthetic data derive from it.
    data, synthetic, survey, scaling, cleaning, features, model, evaluation, output
        See the section classes.

    Examples
    --------
    >>> config = Config().override({"evaluation.n_splits": 3})
    >>> config.evaluation.n_splits, Config.from_dict(config.to_dict()) == config
    (3, True)
    """

    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    survey: SurveyConfig = field(default_factory=SurveyConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ConfigError(f"seed must be a non-negative integer, got {self.seed!r}")
        if self.model.solver == "levenberg_marquardt" and self.model.name != "mlp":
            raise ConfigError(
                "model.solver='levenberg_marquardt' only applies to model.name='mlp', "
                f"got model.name={self.model.name!r}"
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> Config:
        """Build a configuration from a nested mapping, rejecting unknown keys."""
        data = dict(raw or {})
        known = sorted(f.name for f in dataclasses.fields(cls))
        unknown = sorted(set(data) - set(known))
        if unknown:
            raise ConfigError(f"unknown setting(s) {unknown}; valid top-level keys are {known}")
        kwargs: dict[str, Any] = {}
        for name, value in data.items():
            if name in _SECTIONS:
                kwargs[name] = _build_section(name, _SECTIONS[name], value)
            else:
                kwargs[name] = _tuplify(value)
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a YAML file; keys it leaves out keep their defaults."""
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"configuration file not found: {source}")
        try:
            raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"{source} is not valid YAML: {exc}") from None
        if raw is not None and not isinstance(raw, Mapping):
            raise ConfigError(f"{source} must contain a mapping of settings")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        """Plain nested mapping (lists instead of tuples), ready for YAML or JSON."""
        return _listify(dataclasses.asdict(self))

    def to_yaml(self, path: str | Path) -> Path:
        """Write the configuration as YAML and return the path."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)
        destination.write_text(text, encoding="utf-8", newline="\n")
        return destination

    def override(self, updates: Mapping[str, Any]) -> Config:
        """Return a validated copy with dotted keys replaced, e.g. ``{"output.dpi": 300}``."""
        tree = self.to_dict()
        for dotted, value in updates.items():
            *parents, leaf = dotted.split(".")
            node = tree
            for part in parents:
                child = node.get(part)
                if not isinstance(child, dict):
                    valid = sorted(key for key, item in node.items() if isinstance(item, dict))
                    raise ConfigError(
                        f"unknown configuration section in {dotted!r}; valid sections are {valid}"
                    )
                node = child
            if leaf not in node:
                raise ConfigError(
                    f"unknown configuration key {dotted!r}; valid keys here are {sorted(node)}"
                )
            node[leaf] = value
        return Config.from_dict(tree)
