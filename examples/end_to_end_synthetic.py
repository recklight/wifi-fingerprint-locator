"""End to end on synthetic data: synthesize, prepare, cross-validate, train, predict.

    python examples/end_to_end_synthetic.py --out examples/output/end_to_end_synthetic

1. Write a synthetic survey with the same on-disk layout as real captures.
2. Read, scale and clean the captures and build amplitude fingerprints.
3. Cross-validate the MLP with whole reference points held out.
4. Train on every fingerprint and store the model.
5. Reload the model and locate two capture files.

The scores come from synthetic data and say nothing about accuracy in a real room.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from wifi_fingerprint_locator import (
    Config,
    configure_logging,
    cross_validate,
    load_dataset_index,
    load_model,
    predict_inputs,
    prepare_features,
    save_features,
    save_model,
    train_model,
    write_json,
    write_synthetic_dataset,
)

LOGGER = logging.getLogger("end_to_end_synthetic")
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/output/end_to_end_synthetic"),
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

    # 1. A synthetic survey: one capture file per reference point plus dataset.csv.
    index = write_synthetic_dataset(args.out / "data", config.synthetic, rng=config.seed)

    # 2. Fingerprints, one per kept packet.
    features = prepare_features(index, config)
    save_features(args.out / "features.npz", features)

    # 3. Cross-validation with whole reference points held out.
    result = cross_validate(features, config)
    write_json(args.out / "metrics.json", result.metrics)
    result.predictions.to_csv(args.out / "predictions.csv", index=False, lineterminator="\n")
    summary = result.metrics["summary"]["mean_error_m"]
    LOGGER.info(
        "%s: mean error %.3f +/- %.3f m over %d folds",
        config.evaluation.split_strategy,
        summary["mean"],
        summary["std"],
        config.evaluation.n_splits,
    )

    # 4. Train on everything and store the model with its configuration.
    model_path = save_model(args.out / "model.joblib", train_model(features, config), config)

    # 5. Locate two captures with nothing but the stored model.
    captures = load_dataset_index(index)["path"].tolist()[:2]
    estimates = predict_inputs(captures, load_model(model_path))
    estimates.to_csv(args.out / "located.csv", index=False, lineterminator="\n")
    LOGGER.info("%d packets located", len(estimates))
    LOGGER.info("written to %s", args.out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
