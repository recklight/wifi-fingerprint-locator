"""Phase sanitization on a synthetic capture: raw phase versus sanitized phase.

    python examples/sanitize_channel_phase.py --out examples/output/sanitize_channel_phase

1. Simulate one capture at a single reference point and read it back.
2. Unwrap the phase of the first antenna: every packet has its own random
   slope (timing offset) and offset (carrier phase).
3. Sanitize it with both line estimators and both subcarrier layouts, and
   measure how much the sanitized phase still varies between packets taken at
   the same position. The smaller that spread, the better the phase works as a
   fingerprint.

The figures and tables describe synthetic data, not a real channel.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wifi_fingerprint_locator import (
    Config,
    configure_logging,
    make_capture,
    make_room,
    phase_sanitization_table,
    plot_phase_sanitization,
    read_survey,
    sanitize_channel_phase,
    save_figure,
    subcarrier_positions,
    unwrap_phase,
    write_dataset_index,
)

LOGGER = logging.getLogger("sanitize_channel_phase")
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/output/sanitize_channel_phase"),
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

    # 1. One capture, read back through the regular survey path (scaling and cleaning).
    rng = np.random.default_rng(config.seed)
    room = make_room(config.synthetic, rng)
    make_capture(args.out / "capture.dat", room, (2.0, 1.0), config.synthetic, rng=rng)
    rows = pd.DataFrame({"path": ["capture.dat"], "x_m": [2.0], "y_m": [1.0]})
    index = write_dataset_index(args.out / "dataset.csv", rows)
    survey = read_survey(index, config)
    first_antenna = survey.csi[:, 0, :]

    # 2. Raw unwrapped phase against the true subcarrier positions.
    positions = subcarrier_positions("grouped", survey.bandwidth_mhz)
    raw = unwrap_phase(first_antenna)
    sanitized = sanitize_channel_phase(first_antenna, positions, method="endpoint")
    shown = min(5, raw.shape[0])
    table = phase_sanitization_table(raw[:shown], sanitized[:shown], positions)
    table.to_csv(args.out / "phase_sanitization.csv", index=False, lineterminator="\n")
    figure = plot_phase_sanitization(raw[:shown], sanitized[:shown], positions)
    save_figure(figure, args.out / "phase_sanitization", config.output.figure_format)

    # 3. Spread across packets of the same position, per estimator and layout.
    rows = []
    raw_spread = float(np.mean(np.std(raw - raw.mean(axis=1, keepdims=True), axis=0)))
    rows.append({"layout": "-", "method": "unwrapped only", "spread_rad": raw_spread})
    for layout in ("grouped", "uniform"):
        layout_positions = subcarrier_positions(layout, survey.bandwidth_mhz)
        for method in ("endpoint", "lstsq"):
            phase = sanitize_channel_phase(first_antenna, layout_positions, method=method)
            spread = float(np.mean(np.std(phase, axis=0)))
            rows.append({"layout": layout, "method": method, "spread_rad": spread})
            LOGGER.info("%s positions, %s fit: spread %.3f rad", layout, method, spread)
    stability = pd.DataFrame(rows)
    stability.to_csv(args.out / "phase_stability.csv", index=False, lineterminator="\n")
    LOGGER.info("unwrapped phase without sanitization: spread %.3f rad", raw_spread)
    LOGGER.info("written to %s", args.out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
