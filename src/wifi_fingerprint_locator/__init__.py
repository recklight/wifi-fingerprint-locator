"""WiFi CSI fingerprint indoor localization.

The package decodes Intel 5300 beamforming-report captures, rescales and
cleans the channel measurements, turns them into amplitude, sanitized-phase,
bi-modal or real-imaginary fingerprints, and regresses the position of the
receiver with a neural network or a fingerprint-matching baseline, evaluated
under packet-wise, location-wise and temporal splits.

* :func:`read_beamform_file` / :func:`write_beamform_file` -- capture files;
* :func:`scale_csi` / :func:`total_rss_dbm` -- channel in sqrt(SNR) units;
* :func:`sanitize_phase` / :func:`sanitize_channel_phase` -- linear phase sanitization;
* :func:`clean_channels` / :class:`CleaningReport` -- stream-count, zero-entry and NaN policies;
* :func:`fingerprint_matrix` / :data:`REPRESENTATIONS` -- fingerprint representations;
* :func:`read_survey` / :func:`survey_features` -- a whole survey from its dataset index;
* :func:`fold_indices` / :func:`holdout_indices` -- packet, location and temporal splits;
* :class:`PositionRegressor` -- tanh MLP with validation early stopping (L-BFGS, Adam or
  Levenberg-Marquardt);
* :class:`NearestReferencePoint` -- fingerprint-matching baseline;
* :func:`localization_scores` / :func:`empirical_cdf` -- Euclidean error metrics;
* :func:`cross_validate` / :class:`CrossValidationResult` -- k-fold evaluation;
* :func:`compare_representations` / :class:`ComparisonResult` -- representation x split matrix;
* :func:`prepare_features` / :func:`train_model` / :func:`predict_inputs` -- the pipeline steps;
* :func:`write_synthetic_dataset` -- synthetic surveys in the real on-disk format.
"""

from __future__ import annotations

from .baselines import NearestReferencePoint
from .beamform_report import (
    ChannelSnapshot,
    RadioMeta,
    ReadStatistics,
    read_beamform_file,
    summarize_capture,
    write_beamform_file,
)
from .channel_cleaning import CleaningReport, clean_channels, repair_zero_entries
from .comparison import ComparisonResult, compare_representations
from .config import (
    FEATURE_SECTIONS,
    FIGURE_FORMATS,
    CleaningConfig,
    Config,
    DataConfig,
    EvaluationConfig,
    FeaturesConfig,
    ModelConfig,
    OutputConfig,
    ScalingConfig,
    SurveyConfig,
    SyntheticConfig,
)
from .csi_scaling import scale_csi, total_rss_dbm
from .dataset import (
    DATASET_INDEX_NAME,
    TARGET_COLUMNS,
    FeatureSet,
    load_dataset_index,
    load_features,
    save_features,
    write_dataset_index,
)
from .errors import BeamformFormatError, ConfigError, DatasetError, ModelFileError
from .evaluation import CrossValidationResult, cross_validate
from .figures import (
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
from .fingerprints import REPRESENTATIONS, fingerprint_matrix, fingerprint_names
from .levenberg_marquardt import train_levenberg_marquardt
from .logging_utils import DATE_FORMAT, LOG_FORMAT, configure_logging
from .manifest import (
    array_sha256,
    build_manifest,
    file_sha256,
    git_commit,
    run_directory,
    write_json,
    write_manifest,
)
from .metrics import empirical_cdf, localization_scores, position_errors, summarize_errors
from .models import MODEL_NAMES, build_model, load_model, save_model
from .phase_sanitization import sanitize_channel_phase, sanitize_phase, unwrap_phase
from .pipeline import (
    predict_inputs,
    prepare_features,
    prepare_survey,
    run_comparison,
    train_model,
)
from .position_regressor import PositionRegressor
from .site_survey import SurveyChannels, index_capture_folder, read_survey, survey_features
from .splits import SPLIT_STRATEGIES, fold_indices, holdout_indices
from .subcarriers import reported_subcarrier_indices, subcarrier_positions
from .synthetic import make_capture, make_room, write_synthetic_dataset

__version__ = "0.1.0"

__all__ = [
    "DATASET_INDEX_NAME",
    "DATE_FORMAT",
    "FEATURE_SECTIONS",
    "FIGURE_FORMATS",
    "LOG_FORMAT",
    "MODEL_NAMES",
    "REPRESENTATIONS",
    "SERIES_COLORS",
    "SERIES_MARKERS",
    "SPLIT_STRATEGIES",
    "TARGET_COLUMNS",
    "BeamformFormatError",
    "ChannelSnapshot",
    "CleaningConfig",
    "CleaningReport",
    "ComparisonResult",
    "Config",
    "ConfigError",
    "CrossValidationResult",
    "DataConfig",
    "DatasetError",
    "EvaluationConfig",
    "FeatureSet",
    "FeaturesConfig",
    "ModelConfig",
    "ModelFileError",
    "NearestReferencePoint",
    "OutputConfig",
    "PositionRegressor",
    "RadioMeta",
    "ReadStatistics",
    "ScalingConfig",
    "SurveyChannels",
    "SurveyConfig",
    "SyntheticConfig",
    "__version__",
    "array_sha256",
    "build_manifest",
    "build_model",
    "clean_channels",
    "compare_representations",
    "comparison_cdf_table",
    "configure_logging",
    "cross_validate",
    "empirical_cdf",
    "error_cdf_table",
    "file_sha256",
    "fingerprint_matrix",
    "fingerprint_names",
    "fold_indices",
    "git_commit",
    "holdout_indices",
    "index_capture_folder",
    "load_dataset_index",
    "load_features",
    "load_model",
    "localization_scores",
    "location_errors_table",
    "make_capture",
    "make_room",
    "new_figure",
    "phase_sanitization_table",
    "plot_comparison_cdf",
    "plot_error_cdf",
    "plot_location_errors",
    "plot_phase_sanitization",
    "plot_residuals",
    "position_errors",
    "predict_inputs",
    "prepare_features",
    "prepare_survey",
    "read_beamform_file",
    "read_survey",
    "repair_zero_entries",
    "reported_subcarrier_indices",
    "residuals_table",
    "run_comparison",
    "run_directory",
    "sanitize_channel_phase",
    "sanitize_phase",
    "save_features",
    "save_figure",
    "save_model",
    "scale_csi",
    "subcarrier_positions",
    "summarize_capture",
    "summarize_errors",
    "survey_features",
    "total_rss_dbm",
    "train_levenberg_marquardt",
    "train_model",
    "unwrap_phase",
    "write_beamform_file",
    "write_dataset_index",
    "write_json",
    "write_manifest",
    "write_synthetic_dataset",
]
