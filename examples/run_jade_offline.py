"""Run the JADE pipeline without Isaac Sim.

This validates parsing, trajectory sampling, frame transforms, Jacobian-aware
feasibility checks, adaptive policy selection, CSV logging, and report plots.
The actual path in this script is a deterministic first-order mock tracker; use
``run_jade_isaac.py`` for PhysX/Isaac execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from franka_llm_drawing.config import load_frames_config, load_jade_config
from franka_llm_drawing.evaluation import ExecutionLogger, ExecutionLogRow, write_jade_plots
from franka_llm_drawing.evaluation.metrics import max_xy_error, xy_rmse, z_rmse
from franka_llm_drawing.frames import samples_to_cartesian_trajectory
from franka_llm_drawing.jade import compute_jacobian_metrics, recommend_motion_policy, validate_metrics_by_stroke
from franka_llm_drawing.jade.mock_kinematics import estimate_mock_q_from_tip_position, mock_franka_jacobian
from franka_llm_drawing.llm_bridge import load_plan_json
from franka_llm_drawing.trajectory import sample_drawing_plan
from franka_llm_drawing.trajectory.path_primitives import PoseSample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(PROJECT_ROOT / "examples" / "sample_plans" / "circle.json"))
    parser.add_argument("--frames", default=str(PROJECT_ROOT / "configs" / "frames.yaml"))
    parser.add_argument("--jade", default=str(PROJECT_ROOT / "configs" / "jade.yaml"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "jade_offline"))
    parser.add_argument("--mode", choices=["hover", "contact"], default="hover")
    parser.add_argument("--dry-run", action="store_true", help="Parse/sample/validate only; skip mock tracking.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = load_frames_config(args.frames)
    jade = load_jade_config(args.jade)
    plan = load_plan_json(args.plan)

    draw_height = frames.hover_height_m if args.mode == "hover" else frames.draw_height_m
    samples = sample_drawing_plan(
        plan,
        dt=jade.sampling.dt,
        default_speed_m_s=jade.sampling.default_speed_m_s,
        hover_height_m=frames.hover_height_m,
        draw_height_m=draw_height,
        default_normal_force_n=None if args.mode == "hover" else 1.0,
        time_scaling=jade.sampling.time_scaling,
        max_linear_speed_m_s=jade.sampling.max_linear_speed_m_s,
        max_linear_accel_m_s2=jade.sampling.max_linear_accel_m_s2,
        max_linear_jerk_m_s3=jade.sampling.max_linear_jerk_m_s3,
        contact_force_ramp_duration_s=jade.sampling.contact_force_ramp_duration_s,
        pen_down_settle_duration_s=0.0 if args.mode == "hover" else jade.sampling.pen_down_settle_duration_s,
    )
    if args.mode == "hover":
        samples = force_hover_samples(samples, frames.hover_height_m)
    cartesian = samples_to_cartesian_trajectory(
        samples,
        frames.T_base_board,
        frames.T_ee_tip,
        R_base_tip_override=frames.R_base_tip,
    )

    metrics = []
    q_prev: np.ndarray | None = None
    for index, point in enumerate(cartesian):
        q = estimate_mock_q_from_tip_position(point.p_base_tip)
        qdot = np.zeros(7) if q_prev is None else (q - q_prev) / max(jade.sampling.dt, 1e-12)
        q_prev = q
        metrics.append(
            compute_jacobian_metrics(
                mock_franka_jacobian(q),
                q,
                stroke_id=point.stroke_id or point.source_action_name or "unknown",
                waypoint_index=index,
                qdot_estimate=qdot,
            )
        )

    reports = validate_metrics_by_stroke(
        metrics,
        jade.thresholds,
        jade.adaptive_policy,
        jade.executor,
    )
    write_sampled_trajectory(out_dir / "sampled_trajectory.csv", cartesian)
    write_metrics(out_dir / "jacobian_metrics.csv", metrics)
    write_report(out_dir / "feasibility_report.json", reports)

    if not args.dry_run:
        logger = run_mock_tracking(cartesian, metrics, jade)
        logger.write_csv(out_dir / "execution_log.csv")
        write_jade_plots(logger, out_dir)
        desired = logger.desired_positions()
        actual = logger.actual_positions()
        summary = {
            "samples": len(cartesian),
            "mode": args.mode,
            "overall_status": aggregate_status([report.status for report in reports]),
            "xy_rmse_m": xy_rmse(desired, actual),
            "max_xy_error_m": max_xy_error(desired, actual),
            "z_rmse_m": z_rmse(desired, actual),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"mock_xy_rmse_m: {summary['xy_rmse_m']:.6f}")
        print(f"mock_max_xy_error_m: {summary['max_xy_error_m']:.6f}")

    print(f"plan_actions: {len(plan.actions)}")
    print(f"trajectory_samples: {len(cartesian)}")
    print(f"overall_status: {aggregate_status([report.status for report in reports])}")
    print(f"outputs: {out_dir}")


def force_hover_samples(samples: list[PoseSample], hover_height_m: float) -> list[PoseSample]:
    output: list[PoseSample] = []
    for sample in samples:
        position = sample.position.copy()
        position[2] = float(hover_height_m)
        output.append(
            replace(
                sample,
                position=position,
                pen_contact_desired=False,
                desired_normal_force_n=None,
            )
        )
    return output


def run_mock_tracking(cartesian, metrics, jade) -> ExecutionLogger:
    logger = ExecutionLogger()
    actual = cartesian[0].p_base_tip + np.array([0.0015, -0.0010, 0.0005])
    q_prev: np.ndarray | None = None
    for point, metric in zip(cartesian, metrics):
        policy = recommend_motion_policy(metric, jade.thresholds, jade.adaptive_policy, jade.executor)
        alpha = 0.86 * policy.speed_scale
        actual = actual + alpha * (point.p_base_tip - actual)
        q = estimate_mock_q_from_tip_position(actual)
        qdot = np.zeros(7) if q_prev is None else (q - q_prev) / max(jade.sampling.dt, 1e-12)
        q_prev = q
        logger.append(
            ExecutionLogRow(
                t=point.t,
                stroke_id=point.stroke_id or "unknown",
                action_type=point.source_action_name or "unknown",
                desired_position=point.p_base_tip,
                actual_position=actual.copy(),
                q=q,
                qdot=qdot,
                sigma_min=metric.sigma_min,
                condition_number=metric.condition_number,
                manipulability=metric.manipulability,
                lambda_dls=policy.lambda_dls,
                speed_scale=policy.speed_scale,
                status=policy.status,
            )
        )
    return logger


def write_sampled_trajectory(path: Path, cartesian) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["t", "stroke_id", "action_type", "x_tip", "y_tip", "z_tip", "x_ee", "y_ee", "z_ee"]
        )
        for point in cartesian:
            writer.writerow(
                [
                    point.t,
                    point.stroke_id or "",
                    point.source_action_name or "",
                    *point.p_base_tip.tolist(),
                    *point.T_base_ee[:3, 3].tolist(),
                ]
            )


def write_metrics(path: Path, metrics) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(metrics[0].to_dict().keys()) if metrics else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics:
            writer.writerow(item.to_dict())


def write_report(path: Path, reports) -> None:
    payload = {
        "overall_status": aggregate_status([report.status for report in reports]),
        "reports": [report.to_dict() for report in reports],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def aggregate_status(statuses: list[str]) -> str:
    if "FAIL" in statuses:
        return "FAIL"
    if "WARNING" in statuses:
        return "WARNING"
    return "OK"


if __name__ == "__main__":
    main()
