"""Fingerprint representations side by side on one synthetic survey.

    python examples/compare_feature_representations.py  # writes examples/output/<script name>

1. Write a synthetic survey and read it once.
2. Cross-validate the same MLP on amplitude, sanitized phase, amplitude plus
   phase, raw real and imaginary parts, and real and imaginary parts after
   phase sanitization, with the same folds for every representation.
3. Write the summary table and one error CDF per representation.

The split used here (``temporal``) keeps every location in training, so the
numbers compare representations under favorable conditions; see
``compare_split_strategies.py`` for what happens at unseen locations.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from wifi_fingerprint_locator import (
    Config,
    compare_representations,
    comparison_cdf_table,
    configure_logging,
    plot_comparison_cdf,
    read_survey,
    save_figure,
    write_synthetic_dataset,
)

LOGGER = logging.getLogger("compare_feature_representations")
CONFIGS = Path(__file__).resolve().parents[1] / "configs"
REPRESENTATIONS = (
    "amplitude",
    "sanitized_phase",
    "amplitude_phase",
    "real_imag",
    "sanitized_real_imag",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/output/compare_feature_representations"),
        help="Output folder.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Master seed.")
    parser.add_argument(
        "--quick", action="store_true", help="Tiny data and models; finishes in seconds."
    )
    args = parser.parse_args(argv)
    configure_logging("INFO")
    name = "quick_demo.yaml" if args.quick else "synthetic_survey.yaml"
    config = Config.from_yaml(CONFIGS / name).override({"seed": args.seed})
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. One synthetic survey, read once for every representation.
    index = write_synthetic_dataset(args.out / "data", config.synthetic, rng=config.seed)
    survey = read_survey(index, config)

    # 2. Same model, same folds, five representations.
    result = compare_representations(
        survey, config, representations=REPRESENTATIONS, strategies=("temporal",)
    )
    result.summary.to_csv(args.out / "summary.csv", index=False, lineterminator="\n")

    # 3. Error CDFs, with the numbers behind them.
    step = config.evaluation.cdf_step_m
    table = comparison_cdf_table(result.errors, step)
    table.to_csv(args.out / "error_cdf.csv", index=False, lineterminator="\n")
    figure = plot_comparison_cdf(result.errors, step_m=step, dpi=config.output.dpi)
    save_figure(figure, args.out / "error_cdf", config.output.figure_format)
    for row in result.summary.itertuples():
        LOGGER.info(
            "%-20s mean error %.3f +/- %.3f m", row.representation, row.mean_error_m_mean,
            row.mean_error_m_std,
        )  # fmt: skip
    LOGGER.info("written to %s", args.out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
