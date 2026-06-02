"""Run a mock Differential IK control loop without Isaac Sim."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from franka_llm_drawing.config import load_frames_config
from franka_llm_drawing.controllers import DifferentialIKConfig, DifferentialIKController
from franka_llm_drawing.evaluation.logger import LogRecord, TrajectoryLogger
from franka_llm_drawing.evaluation.metrics import max_xy_error, xy_rmse
from franka_llm_drawing.frames import samples_to_cartesian_trajectory
from franka_llm_drawing.llm_bridge import load_plan_json
from franka_llm_drawing.robot import MockFrankaBackend
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

    backend = MockFrankaBackend(dt=args.dt)
    controller = DifferentialIKController(
        DifferentialIKConfig(damping=0.05, position_gain=1.0, orientation_gain=0.5)
    )
    logger = TrajectoryLogger()
    ik_success: list[bool] = []

    for point in cartesian:
        state = backend.get_state()
        J = backend.get_end_effector_jacobian()
        p_des = point.T_base_ee[:3, 3]
        R_des = point.T_base_ee[:3, :3]
        q_target, diagnostics = controller.compute_q_target(
            state.q,
            J,
            p_des,
            R_des,
            state.ee_position,
            state.ee_rotation,
        )
        backend.send_joint_position_target(q_target)
        backend.step()
        next_state = backend.get_state()
        logger.append(
            LogRecord(
                t=point.t,
                desired_position=p_des,
                actual_position=next_state.ee_position,
                q=next_state.q,
                qd=next_state.qd,
                diagnostics={
                    "condition_number": diagnostics.condition_number,
                    "delta_q_norm": diagnostics.delta_q_norm,
                },
            )
        )
        ik_success.append(np.isfinite(q_target).all())

    desired = logger.desired_positions()
    actual = logger.actual_positions()
    print(f"mock control samples: {len(logger.records)}")
    print(f"ik_success_rate: {sum(ik_success) / len(ik_success):.3f}")
    print(f"xy_rmse_m: {xy_rmse(desired, actual):.6f}")
    print(f"max_xy_error_m: {max_xy_error(desired, actual):.6f}")


if __name__ == "__main__":
    main()
