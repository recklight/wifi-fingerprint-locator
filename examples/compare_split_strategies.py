"""How the training/test split changes the error of the same fingerprints.

    python examples/compare_split_strategies.py --out examples/output/compare_split_strategies

1. Write a synthetic survey and read it once.
2. Evaluate amplitude fingerprints under all four split strategies, with the
   MLP and with a weighted kNN fingerprint-matching baseline.
3. Write the summary table and the error CDFs per strategy.

A packet-wise random split leaves the packets recorded just before and after
each test packet in the training set, so it scores far better than a split
that holds out whole locations. That gap is how much a leaky evaluation
flatters the model.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from wifi_fingerprint_locator import (
    SPLIT_STRATEGIES,
    Config,
    compare_representations,
    comparison_cdf_table,
    configure_logging,
    plot_comparison_cdf,
    read_survey,
    save_figure,
    write_synthetic_dataset,
)

LOGGER = logging.getLogger("compare_split_strategies")
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/output/compare_split_strategies"),
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

    # 1. One synthetic survey.
    index = write_synthetic_dataset(args.out / "data", config.synthetic, rng=config.seed)
    survey = read_survey(index, config)

    # 2. Every split strategy, for the network and for the kNN baseline.
    summaries, errors = [], []
    for model in ("mlp", "weighted_knn"):
        result = compare_representations(
            survey,
            config.override({"model.name": model}),
            representations=("amplitude",),
            strategies=SPLIT_STRATEGIES,
        )
        summaries.append(result.summary.assign(model=model))
        errors.append(result.errors.assign(representation=f"amplitude/{model}"))
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(args.out / "summary.csv", index=False, lineterminator="\n")

    # 3. Error CDFs per strategy, one curve per model.
    step = config.evaluation.cdf_step_m
    combined = pd.concat(errors, ignore_index=True)
    comparison_cdf_table(combined, step).to_csv(
        args.out / "error_cdf.csv", index=False, lineterminator="\n"
    )
    figure = plot_comparison_cdf(combined, step_m=step, dpi=config.output.dpi)
    save_figure(figure, args.out / "error_cdf", config.output.figure_format)
    for row in summary.itertuples():
        LOGGER.info("%-13s %-16s mean error %.3f m", row.model, row.strategy, row.mean_error_m_mean)
    LOGGER.info("written to %s", args.out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
