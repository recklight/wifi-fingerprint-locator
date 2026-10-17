"""Command line of the package: ``wifi-fingerprint-locator <command> ...``.

Commands
--------
synthesize
    Write a synthetic survey (capture files and ``dataset.csv``).
index
    Write ``FOLDER/dataset.csv`` for capture files whose names encode grid indices.
inspect
    Summarize capture files: reports, stream counts, radio levels and timing.
prepare
    Read the captures listed in ``data.dataset`` and build fingerprints.
evaluate
    Cross-validate the configured model under ``evaluation.split_strategy``.
train
    Train the configured model on every fingerprint.
predict
    Estimate positions for new capture files with a trained model.
compare
    Evaluate every fingerprint representation under every split strategy.

Every command except ``predict`` takes ``--config`` (a YAML file) and any
number of ``--set section.key=value`` overrides. ``inspect``, ``prepare``,
``evaluate``, ``train``, ``predict`` and ``compare`` write their results to
``--out`` or, without it, to a new time-stamped run folder below
``output.root``, together with ``config.yaml`` and ``manifest.json``.
``synthesize`` writes to ``--out`` (``data/synthetic`` by default), and
``index`` writes ``FOLDER/dataset.csv``.

Expected failures end with a one-line message and exit code 2.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer

from . import __version__
from .beamform_report import read_beamform_file, summarize_capture
from .config import Config
from .dataset import DATASET_INDEX_NAME, load_features, save_features, write_dataset_index
from .errors import ConfigError
from .evaluation import cross_validate
from .figures import (
    comparison_cdf_table,
    error_cdf_table,
    location_errors_table,
    plot_comparison_cdf,
    plot_error_cdf,
    plot_location_errors,
    plot_residuals,
    residuals_table,
    save_figure,
)
from .logging_utils import configure_logging
from .manifest import build_manifest, run_directory, write_json, write_manifest
from .models import load_model, save_model
from .pipeline import predict_inputs, prepare_survey, run_comparison, train_model
from .site_survey import index_capture_folder
from .synthetic import write_synthetic_dataset

_LOGGER = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    # Local variables of a failing frame can hold whole datasets; never dump them.
    pretty_exceptions_show_locals=False,
    help="Indoor localization from WiFi CSI fingerprints (amplitude, phase, real-imaginary), "
    "evaluated with packet, location and time splits.",
)

ConfigOption = Annotated[
    Path | None, typer.Option("--config", "-c", help="YAML configuration file.")
]
SetOption = Annotated[
    list[str] | None,
    typer.Option("--set", "-s", help="Override one setting: section.key=value (repeatable)."),
]
SeedOption = Annotated[
    int | None, typer.Option("--seed", help="Master seed; overrides the configuration.")
]
DatasetOption = Annotated[
    Path | None,
    typer.Option("--dataset", "-d", help="Dataset index CSV; overrides data.dataset."),
]
OutOption = Annotated[
    Path | None,
    typer.Option(
        "--out", "-o", help="Output folder (default: a new run folder under output.root)."
    ),
]
ModelOption = Annotated[Path, typer.Option("--model", "-m", help="Model file written by 'train'.")]
FiguresOption = Annotated[
    bool, typer.Option("--figures/--no-figures", help="Also draw the figures of the tables.")
]


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code=2)


@contextmanager
def _reported_errors() -> Iterator[None]:
    """Turn the library's expected failures into a one-line message and exit code 2."""
    try:
        yield
    except (ValueError, FileNotFoundError) as exc:
        _LOGGER.debug("command failed", exc_info=True)
        raise _fail(str(exc)) from None


def _coerce(raw: str) -> Any:
    """Parse a ``--set`` value: true/false, none/null, then JSON, else the plain string."""
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _load_config(
    config_file: Path | None,
    overrides: list[str] | None,
    shortcuts: Mapping[str, Any] | None = None,
) -> Config:
    """Apply ``--config``, then every ``--set``, then the shortcut options.

    A shortcut whose value is ``None`` was not given on the command line and
    leaves the setting alone.
    """
    config = Config.from_yaml(config_file) if config_file is not None else Config()
    changes: dict[str, Any] = {}
    for item in overrides or ():
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"--set expects section.key=value, got {item!r}")
        changes[key.strip()] = _coerce(raw.strip())
    for key, value in (shortcuts or {}).items():
        if value is not None:
            changes[key] = value.as_posix() if isinstance(value, Path) else value
    return config.override(changes) if changes else config


def _report(label: str, value: object) -> None:
    """One line of the summary a command prints when it finishes."""
    typer.echo(f"{label:<11}: {value}")


def _run_folder(config: Config, command: str, out: Path | None) -> Path:
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        return out
    return run_directory(config.output.root, command)


def _finish(
    folder: Path,
    command: str,
    config: Config,
    inputs: Sequence[Path],
    details: Mapping[str, Any],
) -> None:
    """Write ``config.yaml`` and ``manifest.json`` and print the closing line."""
    config.to_yaml(folder / "config.yaml")
    manifest = build_manifest(
        command=command, config=config.to_dict(), inputs=inputs, seed=config.seed, extra=details
    )
    write_manifest(folder / "manifest.json", manifest)
    typer.echo(f"written to {folder.as_posix()}")


def _write_table(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, lineterminator="\n")
    return path


def _mean_std(summary: Mapping[str, Mapping[str, float]], key: str) -> str:
    return f"{summary[key]['mean']:.3f} +/- {summary[key]['std']:.3f} m"


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"wifi-fingerprint-locator {__version__}")
        raise typer.Exit()


@app.callback()
def _global_options(
    log_level: Annotated[
        str, typer.Option("--log-level", help="DEBUG, INFO, WARNING or ERROR.")
    ] = "INFO",
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_show_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    try:
        configure_logging(log_level)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--log-level") from None


@app.command("synthesize")
def synthesize_command(
    config_file: ConfigOption = None,
    set_: SetOption = None,
    seed: SeedOption = None,
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Folder for the synthetic dataset.")
    ] = Path("data/synthetic"),
) -> None:
    """Write a synthetic survey with the on-disk layout of real captures."""
    with _reported_errors():
        config = _load_config(config_file, set_, {"seed": seed})
        index = write_synthetic_dataset(out, config.synthetic, rng=config.seed)
        rows = pd.read_csv(index)
        n_packets = int(len(rows) * config.synthetic.n_packets)
        _report("captures", len(rows))
        _report("locations", rows["group"].nunique())
        _report("packets", n_packets)
        _finish(
            out,
            "synthesize",
            config,
            [],
            {"n_captures": len(rows), "n_packets": n_packets, "dataset": index.as_posix()},
        )


@app.command("index")
def index_command(
    folder: Annotated[Path, typer.Argument(help="Folder of capture files.")],
    config_file: ConfigOption = None,
    set_: SetOption = None,
) -> None:
    """Write FOLDER/dataset.csv for capture files whose names encode grid indices."""
    with _reported_errors():
        config = _load_config(config_file, set_)
        rows = index_capture_folder(folder, config.data)
        destination = write_dataset_index(folder / DATASET_INDEX_NAME, rows)
        _report("captures", len(rows))
        _report("locations", rows["group"].nunique())
        typer.echo(f"written to {destination.as_posix()}")


@app.command("inspect")
def inspect_command(
    inputs: Annotated[list[Path], typer.Argument(help="Capture files to summarize.")],
    config_file: ConfigOption = None,
    set_: SetOption = None,
    out: OutOption = None,
) -> None:
    """Summarize capture files: reports, stream counts, radio levels and timing."""
    with _reported_errors():
        config = _load_config(config_file, set_)
        rows = []
        for path in inputs:
            snapshots, statistics = read_beamform_file(
                path,
                apply_permutation=config.survey.apply_permutation,
                strict=config.survey.strict,
            )
            rows.append({"path": path.as_posix(), **summarize_capture(snapshots, statistics)})
        table = pd.DataFrame(rows)
        folder = _run_folder(config, "inspect", out)
        _write_table(table, folder / "captures.csv")
        _report("captures", len(table))
        _report("reports", int(table["n_reports"].sum()))
        _report("two-stream", int(table["n_multi_stream"].sum()))
        _report("zero values", int(table["n_zero_entries"].sum()))
        _report("skipped", int((table["n_malformed"] + table["n_truncated"]).sum()))
        _finish(folder, "inspect", config, list(inputs), {"n_captures": len(table)})


@app.command("prepare")
def prepare_command(
    config_file: ConfigOption = None,
    set_: SetOption = None,
    dataset: DatasetOption = None,
    out: OutOption = None,
) -> None:
    """Read the captures listed in data.dataset and build fingerprints."""
    with _reported_errors():
        config = _load_config(config_file, set_, {"data.dataset": dataset})
        features, report = prepare_survey(config.data.dataset, config)
        folder = _run_folder(config, "prepare", out)
        save_features(folder / "features.npz", features)
        _write_table(features.items_table(), folder / "items.csv")
        n_locations = int(np.unique(features.groups).size)
        n_captures = int(np.unique(features.paths).size)
        _report("items", features.n_items)
        _report("features", f"{features.n_features} ({config.features.representation})")
        _report("locations", n_locations)
        _report("captures", n_captures)
        _report("dropped", report.n_input - report.n_output)
        _finish(
            folder,
            "prepare",
            config,
            [Path(config.data.dataset)],
            {
                "n_items": features.n_items,
                "n_features": features.n_features,
                "n_locations": n_locations,
                "n_captures": n_captures,
                "cleaning": report.as_dict(),
            },
        )


@app.command("evaluate")
def evaluate_command(
    features: Annotated[Path, typer.Argument(help="features.npz written by 'prepare'.")],
    config_file: ConfigOption = None,
    set_: SetOption = None,
    seed: SeedOption = None,
    out: OutOption = None,
    figures: FiguresOption = True,
) -> None:
    """Cross-validate the configured model; write metrics, predictions and figures."""
    with _reported_errors():
        config = _load_config(config_file, set_, {"seed": seed})
        data = load_features(features, config)
        result = cross_validate(data, config)
        folder = _run_folder(config, "evaluate", out)
        fmt, dpi = config.output.figure_format, config.output.dpi
        write_json(folder / "metrics.json", result.metrics)
        _write_table(result.folds, folder / "folds.csv")
        _write_table(result.predictions, folder / "predictions.csv")
        _write_table(result.recording_predictions, folder / "recording_predictions.csv")
        _write_table(residuals_table(result.predictions), folder / "residuals.csv")
        curves = {
            "packets": result.errors,
            "captures": result.recording_predictions["error_m"].to_numpy(dtype=np.float64),
        }
        step = config.evaluation.cdf_step_m
        _write_table(error_cdf_table(curves, step), folder / "error_cdf.csv")
        _write_table(location_errors_table(result.predictions), folder / "location_errors.csv")
        if figures:
            save_figure(plot_residuals(result.predictions, dpi=dpi), folder / "residuals", fmt)
            title = (
                f"Error CDF ({config.features.representation}, {config.evaluation.split_strategy})"
            )
            save_figure(
                plot_error_cdf(curves, step_m=step, title=title, dpi=dpi),
                folder / "error_cdf",
                fmt,
            )
            save_figure(
                plot_location_errors(result.predictions, dpi=dpi), folder / "location_errors", fmt
            )
        summary = result.metrics["summary"]
        _report("items", data.n_items)
        _report("folds", f"{config.evaluation.n_splits} ({config.evaluation.split_strategy})")
        _report("mean error", _mean_std(summary, "mean_error_m"))
        _report("median", _mean_std(summary, "median_error_m"))
        _report("p90", _mean_std(summary, "p90_error_m"))
        _report("grid acc.", f"{summary['grid_accuracy']['mean']:.3f}")
        _finish(
            folder,
            "evaluate",
            config,
            [features],
            {
                "n_items": data.n_items,
                "split_strategy": config.evaluation.split_strategy,
                "representation": config.features.representation,
            },
        )


@app.command("train")
def train_command(
    features: Annotated[Path, typer.Argument(help="features.npz written by 'prepare'.")],
    config_file: ConfigOption = None,
    set_: SetOption = None,
    seed: SeedOption = None,
    out: OutOption = None,
) -> None:
    """Train the configured model on every fingerprint."""
    with _reported_errors():
        config = _load_config(config_file, set_, {"seed": seed})
        data = load_features(features, config)
        model = train_model(data, config)
        folder = _run_folder(config, "train", out)
        save_model(folder / "model.joblib", model, config)
        details: dict[str, Any] = {"n_items": data.n_items, "model": config.model.name}
        network = getattr(model.named_steps["regressor"], "regressor_", None)
        if network is not None and hasattr(network, "stop_reason_"):
            details.update(
                epochs=int(network.n_epochs_),
                best_epoch=int(network.best_epoch_),
                stop_reason=network.stop_reason_,
                alpha=float(network.alpha_),
            )
        _report("items", data.n_items)
        _report("model", config.model.name)
        if "stop_reason" in details:
            _report("stopped", f"{details['stop_reason']} after {details['epochs']} iterations")
        _finish(folder, "train", config, [features], details)


@app.command("predict")
def predict_command(
    inputs: Annotated[list[Path], typer.Argument(help="Capture files to locate.")],
    model: ModelOption,
    out: OutOption = None,
) -> None:
    """Estimate positions for new capture files with a trained model."""
    with _reported_errors():
        bundle = load_model(model)
        config = Config.from_dict(bundle["config"])
        table = predict_inputs(inputs, bundle)
        folder = _run_folder(config, "predict", out)
        _write_table(table, folder / "predictions.csv")
        _report("captures", len(inputs))
        _report("packets", len(table))
        _finish(folder, "predict", config, [model, *inputs], {"n_items": len(table)})


@app.command("compare")
def compare_command(
    config_file: ConfigOption = None,
    set_: SetOption = None,
    seed: SeedOption = None,
    dataset: DatasetOption = None,
    out: OutOption = None,
    figures: FiguresOption = True,
) -> None:
    """Evaluate every fingerprint representation under every split strategy."""
    with _reported_errors():
        config = _load_config(config_file, set_, {"seed": seed, "data.dataset": dataset})
        result = run_comparison(config.data.dataset, config)
        folder = _run_folder(config, "compare", out)
        step = config.evaluation.cdf_step_m
        _write_table(result.folds, folder / "comparison_folds.csv")
        _write_table(result.summary, folder / "comparison_summary.csv")
        write_json(folder / "comparison.json", result.as_dict())
        _write_table(comparison_cdf_table(result.errors, step), folder / "error_cdf.csv")
        if figures:
            save_figure(
                plot_comparison_cdf(result.errors, step_m=step, dpi=config.output.dpi),
                folder / "error_cdf",
                config.output.figure_format,
            )
        best = result.best()
        worst_row = result.summary.loc[result.summary["mean_error_m_mean"].idxmax()]
        _report("runs", f"{len(result.summary)} ({len(result.folds)} folds)")
        _report(
            "best",
            f"{best['representation']} / {best['strategy']}: {best['mean_error_m']:.3f} m",
        )
        _report(
            "worst",
            f"{worst_row['representation']} / {worst_row['strategy']}: "
            f"{worst_row['mean_error_m_mean']:.3f} m",
        )
        _finish(
            folder,
            "compare",
            config,
            [Path(config.data.dataset)],
            {
                "representations": list(config.evaluation.compare_representations),
                "strategies": list(config.evaluation.compare_strategies),
            },
        )


def main() -> None:
    """Entry point of the ``wifi-fingerprint-locator`` console script."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
