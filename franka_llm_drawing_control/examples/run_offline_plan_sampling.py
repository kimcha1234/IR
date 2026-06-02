"""Sample an LLM DrawingPlan without Isaac Sim."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from franka_llm_drawing.config import load_frames_config
from franka_llm_drawing.frames import samples_to_cartesian_trajectory
from franka_llm_drawing.llm_bridge import load_plan_json
from franka_llm_drawing.trajectory import sample_drawing_plan

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        default=str(Path(__file__).with_name("sample_plan_circle.json")),
        help="Path to a DrawingPlan JSON file.",
    )
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--csv", default=None, help="Optional output CSV path.")
    args = parser.parse_args()

    plan = load_plan_json(args.plan)
    samples = sample_drawing_plan(
        plan,
        dt=args.dt,
        default_speed_m_s=0.03,
        hover_height_m=0.03,
        draw_height_m=0.0,
        default_normal_force_n=1.0,
    )
    frames = load_frames_config(PROJECT_ROOT / "configs" / "frames.yaml")
    cartesian = samples_to_cartesian_trajectory(samples, frames.T_base_board, frames.T_ee_tip)

    print(f"loaded plan: {args.plan}")
    print(f"board-frame samples: {len(samples)}")
    print(f"duration_s: {samples[-1].t:.3f}")
    print(f"first p_base_tip: {cartesian[0].p_base_tip.tolist()}")
    print(f"last p_base_tip: {cartesian[-1].p_base_tip.tolist()}")
    print(f"first T_base_ee translation: {cartesian[0].T_base_ee[:3, 3].tolist()}")

    if args.csv is not None:
        write_csv(args.csv, cartesian)
        print(f"wrote csv: {args.csv}")


def write_csv(path: str | Path, points) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "x_tip", "y_tip", "z_tip", "x_ee", "y_ee", "z_ee", "contact"])
        for point in points:
            writer.writerow(
                [
                    point.t,
                    *point.p_base_tip.tolist(),
                    *point.T_base_ee[:3, 3].tolist(),
                    int(point.pen_contact_desired),
                ]
            )


if __name__ == "__main__":
    main()
