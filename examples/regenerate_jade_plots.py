"""Regenerate JADE SVG plots from an existing execution_log.csv."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from franka_llm_drawing.evaluation import ExecutionLogger, ExecutionLogRow, write_jade_plots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Directory containing execution_log.csv.")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    log_path = out_dir / "execution_log.csv"
    logger = ExecutionLogger()
    with log_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            logger.append(
                ExecutionLogRow(
                    t=float(row["t"]),
                    stroke_id=row["stroke_id"],
                    action_type=row["action_type"],
                    desired_position=np.array(
                        [float(row["x_des"]), float(row["y_des"]), float(row["z_des"])],
                        dtype=float,
                    ),
                    actual_position=np.array(
                        [float(row["x_act"]), float(row["y_act"]), float(row["z_act"])],
                        dtype=float,
                    ),
                    q=np.array([float(row[f"q{i}"]) for i in range(1, 8)], dtype=float),
                    qdot=np.array([float(row[f"qdot{i}"]) for i in range(1, 8)], dtype=float),
                    sigma_min=float(row["sigma_min"]),
                    condition_number=float(row["condition_number"]),
                    manipulability=float(row["manipulability"]),
                    lambda_dls=float(row["lambda_dls"]),
                    speed_scale=float(row["speed_scale"]),
                    status=row["status"],
                    joint_limit_margin=_optional_float(row.get("joint_limit_margin")),
                    tip_orientation_error_deg=_optional_float(row.get("tip_orientation_error_deg")),
                    policy_reason=row.get("policy_reason", ""),
                )
            )
    paths = write_jade_plots(logger, out_dir)
    print(f"regenerated {len(paths)} plots in {out_dir}")


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


if __name__ == "__main__":
    main()
